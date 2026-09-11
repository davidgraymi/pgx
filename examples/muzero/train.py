# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import os
import pickle
import time
from functools import partial
from typing import NamedTuple
from pydantic import ConfigDict

import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import mctx
import optax
import pgx
import wandb
from omegaconf import OmegaConf
from pgx.experimental import auto_reset
from pydantic import BaseModel

from network import AZNet

devices = jax.local_devices()
num_devices = len(devices)
# 1. Create a 1D grid representation of your accelerator devices
mesh = Mesh(np.array(devices), axis_names=("devices",))
# 2. Tell JAX to shard across the newly stacked leading axis (axis 0)
sharding = NamedSharding(mesh, P("devices"))


class Config(BaseModel):
    env_id: pgx.EnvId = "chess"
    seed: int = 0
    max_num_iters: int = 400
    # network params
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 64
    num_simulations: int = 16
    max_num_steps: int = 128
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    # training params
    training_batch_size: int = 2048
    learning_rate: float = 0.001
    load_ckpt: str | None = None
    training_mode: str = "rl"
    # eval params
    eval_interval: int = 5
    model_config = ConfigDict(extra="forbid")
    champion: str | None = None


conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)
print(config)

env = pgx.make(config.env_id)


def forward_fn(x, is_eval=False):
    net = AZNet(
        num_actions=env.num_actions,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
        resnet_v2=config.resnet_v2,
    )
    policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
    return policy_out, value_out


forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
optimizer = optax.adam(learning_rate=config.learning_rate)


def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    # model: params
    # state: embedding
    del rng_key
    model_params, model_state = model

    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    (logits, value), _ = forward.apply(model_params, model_state, state.observation, is_eval=True)
    # mask invalid actions
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.terminated, 0.0, discount)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


@jax.pmap
def selfplay(model, rng_key: jnp.ndarray) -> SelfplayOutput:
    model_params, model_state = model
    batch_size = config.selfplay_batch_size // num_devices

    def step_fn(state, key) -> SelfplayOutput:
        key1, key2, key_noise = jax.random.split(key, 3)
        observation = state.observation

        # 1. Run standard forward pass to gather logits and value predictions
        (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )

        # 2. Sample raw Dirichlet noise using config.root_dirichlet_alpha
        # (Standard default for Chess is 0.3)
        noise_alpha = jnp.full((logits.shape[-1],), config.root_dirichlet_alpha)
        dirichlet_noise = jax.random.dirichlet(key_noise, noise_alpha)
        noise_logits = jnp.log(dirichlet_noise + 1e-8)

        # 3. Mix exploration noise into raw model predictions using config.root_exploration_fraction
        # (Standard default fraction is 0.25)
        mixed_logits = (1.0 - config.root_exploration_fraction) * logits + config.root_exploration_fraction * noise_logits

        # 4. Apply illegal move masking after mixing the noise
        mixed_logits = mixed_logits - jnp.max(mixed_logits, axis=-1, keepdims=True)
        masked_root_logits = jnp.where(state.legal_action_mask, mixed_logits, jnp.finfo(mixed_logits.dtype).min)

        # Pack masked, noise-injected logits into the MCTS search blueprint
        root = mctx.RootFnOutput(prior_logits=masked_root_logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        
        actor = state.current_player
        keys = jax.random.split(key2, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, policy_output.action, keys)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)
        
        return state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
        )

    # Run selfplay for max_num_steps by batch
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    _, data = jax.lax.scan(step_fn, state, key_seq)

    return data


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


@jax.pmap
def compute_loss_input(data: SelfplayOutput) -> Sample:
    batch_size = config.selfplay_batch_size // num_devices
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, value_tgt = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size),
        jnp.arange(config.max_num_steps),
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )


def loss_fn(model_params, model_state, samples: Sample):
    (logits, value), model_state = forward.apply(
        model_params, model_state, samples.obs, is_eval=False
    )

    policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    value_loss = optax.l2_loss(value, samples.value_tgt)
    value_loss = jnp.mean(value_loss * samples.mask)

    return policy_loss + value_loss, (model_state, policy_loss, value_loss)


@partial(jax.pmap, axis_name="i")
def train(model, opt_state, data: Sample):
    model_params, model_state = model
    grads, (model_state, policy_loss, value_loss) = jax.grad(loss_fn, has_aux=True)(
        model_params, model_state, data
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    model = (model_params, model_state)
    return model, opt_state, policy_loss, value_loss


@jax.pmap
def evaluate(rng_key, my_model, baseline_model):
    """Evaluates the live learning model against a stable baseline snapshot model."""
    my_player = 0
    my_model_params, my_model_state = my_model
    base_model_params, base_model_state = baseline_model

    key, subkey = jax.random.split(rng_key)
    batch_size = config.selfplay_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)

    def body_fn(val):
        key, state, R = val
        # Policy output for the active learning agent
        (my_logits, _), _ = forward.apply(
            my_model_params, my_model_state, state.observation, is_eval=True
        )
        # Policy output for the snapshot target baseline agent
        (opp_logits, _), _ = forward.apply(
            base_model_params, base_model_state, state.observation, is_eval=True
        )
        
        is_my_turn = (state.current_player == my_player).reshape((-1, 1))
        logits = jnp.where(is_my_turn, my_logits, opp_logits)
        
        # Mask out illegal moves during evaluation to keep games valid
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits, axis=-1)
        state = jax.vmap(env.step)(state, action)
        R = R + state.rewards[jnp.arange(batch_size), my_player]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()), body_fn, (key, state, jnp.zeros(batch_size))
    )
    return R


def supervised_loss_fn(model_params, model_state, obs, target_actions):
    """Computes categorical cross-entropy and tracking metrics against expert actions."""
    # Forward pass through your network
    (logits, _), model_state = forward.apply(
        model_params, model_state, obs, is_eval=False
    )
    
    # Calculate cross entropy
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_actions)
    loss = jnp.mean(loss)
    
    # --- CALCULATE METRICS ---
    # Top-1 Accuracy: Does the argmax match the target?
    predictions = jnp.argmax(logits, axis=-1)
    top1_acc = jnp.mean(predictions == target_actions)
    
    # Top-5 Accuracy: Is the target action in the top 5 predicted logits?
    # jax.lax.top_k returns values and indices; we just need the indices
    _, top5_indices = jax.lax.top_k(logits, k=5)
    # Check if target matches any of the 5 columns along the last axis
    top5_acc = jnp.mean(jnp.any(top5_indices == target_actions[:, None], axis=-1))
    
    # Policy Entropy: Measures model confidence (-sum(p * log(p)))
    probs = jax.nn.softmax(logits, axis=-1)
    # Add a tiny epsilon to prevent log(0)
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
    mean_entropy = jnp.mean(entropy)
    
    # Bundle metrics into a custom dictionary inside your auxiliary payload
    metrics = {
        "loss": loss,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "entropy": mean_entropy
    }
    
    # Return the loss scalar first, and the updated state + metrics as a tuple for has_aux
    return loss, (model_state, metrics)


@partial(jax.pmap, axis_name="i")
def supervised_train_step(model, opt_state, obs, target_actions):
    """A parallelized parameter update optimization step for supervised learning."""
    model_params, model_state = model
    
    # Capture the nested dictionary from has_aux
    (loss, (model_state, metrics)), grads = jax.value_and_grad(supervised_loss_fn, has_aux=True)(
        model_params, model_state, obs, target_actions
    )
    
    # Average gradients and your custom metric dictionary values across all cores
    grads = jax.lax.pmean(grads, axis_name="i")
    metrics = jax.lax.pmean(metrics, axis_name="i")
    
    # Apply standard optimizer updates
    updates, opt_state = optimizer.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    
    return (model_params, model_state), opt_state, metrics


def make_data_loader(config, num_devices):
    """A clean generator that buffers raw unpadded frames and yields full sharded mini-batches."""
    # Initialize rolling host memory banks
    buffer_obs = np.empty((0,), dtype=np.float32)  # Will resize automatically on first concat
    buffer_policy = np.empty((0,), dtype=np.float32)
    buffer_value = np.empty((0,), dtype=np.float32)
    buffer_mask = np.empty((0,), dtype=bool)

    batch_per_device = config.training_batch_size // num_devices

    while True:
        # 1. Receive incoming raw sharded GPU data from the selfplay collection step
        samples: Sample = yield
        
        # Pull sharded device data back to CPU host RAM
        samples = jax.device_get(samples)

        # Extract only real gameplay steps using the boolean mask array
        valid_obs = samples.obs[samples.mask]
        valid_policy = samples.policy_tgt[samples.mask]
        valid_value = samples.value_tgt[samples.mask]
        valid_mask = samples.mask[samples.mask]

        # Initialize shapes correctly on the very first iteration step
        if buffer_obs.ndim == 1:
            buffer_obs = np.empty((0, *valid_obs.shape[1:]), dtype=valid_obs.dtype)
            buffer_policy = np.empty((0, *valid_policy.shape[1:]), dtype=valid_policy.dtype)
            buffer_value = np.empty((0, *valid_value.shape[1:]), dtype=valid_value.dtype)

        # Append new unpadded frames to our persistent storage banks
        buffer_obs = np.concatenate([buffer_obs, valid_obs], axis=0)
        buffer_policy = np.concatenate([buffer_policy, valid_policy], axis=0)
        buffer_value = np.concatenate([buffer_value, valid_value], axis=0)
        buffer_mask = np.concatenate([buffer_mask, valid_mask], axis=0)

        # Calculate total available complete mini-batches
        num_updates = buffer_obs.shape[0] // config.training_batch_size
        if num_updates == 0:
            # Yield None to tell the outer loop to skip optimization and run another selfplay step
            yield None
            continue

        total_elements_to_train = num_updates * config.training_batch_size

        # Extract complete training slices from the front of the rolling storage buffer
        train_obs = buffer_obs[:total_elements_to_train]
        train_policy = buffer_policy[:total_elements_to_train]
        train_value = buffer_value[:total_elements_to_train]
        train_mask = buffer_mask[:total_elements_to_train]

        # Retain leftovers for subsequent training windows
        buffer_obs = buffer_obs[total_elements_to_train:]
        buffer_policy = buffer_policy[total_elements_to_train:]
        buffer_value = buffer_value[total_elements_to_train:]
        buffer_mask = buffer_mask[total_elements_to_train:]

        # Randomize execution sequences on the CPU
        shuf_idx = np.random.permutation(total_elements_to_train)
        train_obs = train_obs[shuf_idx]
        train_policy = train_policy[shuf_idx]
        train_value = train_value[shuf_idx]
        train_mask = train_mask[shuf_idx]

        # Reshape directly into JAX parallel structures: (num_updates, num_devices, batch_per_device, ...)
        minibatches_obs = train_obs.reshape(num_updates, num_devices, batch_per_device, *train_obs.shape[1:])
        minibatches_policy = train_policy.reshape(num_updates, num_devices, batch_per_device, *train_policy.shape[1:])
        minibatches_value = train_value.reshape(num_updates, num_devices, batch_per_device, *train_value.shape[1:])
        minibatches_mask = train_mask.reshape(num_updates, num_devices, batch_per_device, *train_mask.shape[1:])

        # Build list of ready-to-run Sample namedtuples
        batches_list = [
            Sample(obs=minibatches_obs[i], policy_tgt=minibatches_policy[i], value_tgt=minibatches_value[i], mask=minibatches_mask[i])
            for i in range(num_updates)
        ]

        # Send the processed batches back to the main loop execution thread
        yield batches_list


if __name__ == "__main__":
    wandb.init(project="pgx-chess-muzero", config=config.model_dump())

    # Prepare checkpoint dir
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    now = now.strftime("%Y%m%d%H%M%S")
    ckpt_dir = os.path.join("checkpoints", f"{config.env_id}_{now}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initialize model and opt_state
    dummy_state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    dummy_input = dummy_state.observation
    model = forward.init(jax.random.PRNGKey(0), dummy_input)  # (params, state)
    opt_state = optimizer.init(params=model[0])

    # Load a checkpoint to train from
    if config.load_ckpt is not None:
        if os.path.exists(config.load_ckpt):
            print("Found checkpoint! Loading weights...")
            with open(config.load_ckpt, "rb") as f:
                model = pickle.load(f)
                opt_state = optimizer.init(params=model[0])

    # Train via supervised learning
    if config.training_mode == "supervised":
        sl_dataset_path = os.path.join("data", "sl_dataset.pkl")

        if not os.path.exists(sl_dataset_path):
            print(f"Dataset not found at {sl_dataset_path}. Run python data.py to download the dataset.")
        else:
            print("Found dataset! Commencing Supervised Pre-Training...")
            with open(sl_dataset_path, "rb") as f:
                # Expects a dict containing 'observations' and 'actions' arrays
                dataset = pickle.load(f)

            sl_obs = np.asarray(dataset["observations"])
            sl_actions = np.asarray(dataset["actions"])

            # Temporarily shard the states onto active cores for the optimization step
            sl_model = jax.tree_util.tree_map(
                lambda x: jax.device_put(jnp.stack([x] * num_devices), sharding), model
            )
            sl_opt_state = jax.tree_util.tree_map(
                lambda x: jax.device_put(jnp.stack([x] * num_devices), sharding), opt_state
            )

            # Simple mini-batch loop parameters for SL profiling
            sl_batch_size: int = config.training_batch_size
            num_samples: int = sl_obs.shape[0]
            num_batches: int = num_samples // sl_batch_size
            hours: float = 0.0
            global_step: int = 0
            epoch: int = 0

            while True:
                if epoch % config.eval_interval == 0:
                    # Store checkpoints
                    model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (sl_model, sl_opt_state))
                    chpt_0 = os.path.join(ckpt_dir, f"{epoch:06d}.ckpt")
                    cpu_model_snapshot = jax.device_get(model_0)
                    with open(chpt_0, "wb") as f:
                        dic = {
                            "config": config,
                            "model": cpu_model_snapshot,
                            "opt_state": jax.device_get(opt_state_0),
                            "epoch": epoch,
                            "step": global_step,
                            "hours": hours,
                            "pgx.__version__": pgx.__version__,
                            "env_id": env.id,
                            "env_version": env.version,
                        }
                        pickle.dump(dic, f)
                        print(f"Saved {chpt_0}")

                if epoch >= config.max_num_iters:
                    break

                epoch += 1
                indices = np.random.permutation(num_samples)
                epoch_loss = 0.0

                for b in range(num_batches):
                    st = time.time()

                    batch_idx = indices[b * sl_batch_size : (b + 1) * sl_batch_size]
                    raw_obs = sl_obs[batch_idx]
                    raw_actions = sl_actions[batch_idx]
                    reshaped_obs = raw_obs.reshape(num_devices, sl_batch_size // num_devices, *raw_obs.shape[1:])
                    reshaped_actions = raw_actions.reshape(num_devices, sl_batch_size // num_devices, *raw_actions.shape[1:])
                    batch_obs = jax.device_put(reshaped_obs)
                    batch_actions = jax.device_put(reshaped_actions)
                    sl_model, sl_opt_state, sharded_metrics = supervised_train_step(
                        sl_model, sl_opt_state, batch_obs, batch_actions
                    )

                    processed_metrics = {
                        f"supervised/{k}": float(jax.device_get(jnp.mean(v)))
                        for k, v in sharded_metrics.items()
                    }
                    step_loss = processed_metrics.get("supervised/loss", 0.0)
                    epoch_loss += step_loss
                    global_step += 1
                    et = time.time()
                    hours += (et - st) / 3600

                    log = {
                        "epoch": epoch,
                        "step": global_step,
                        "hours": hours,
                        **processed_metrics
                    }
                    wandb.log(log)

                avg_epoch_loss = epoch_loss / num_batches

                log.update({"supervised/epoch_loss": avg_epoch_loss})
                print(log)
                wandb.log(log)

                current_top1 = log.get("supervised/top1_accuracy", 0.0)
                current_entropy = log.get("supervised/entropy", 99.0)
                
                # Stop if accuracy is too high or entropy collapses too low
                if current_top1 >= 0.55:
                    print(f"\n[Early Stopping] Top-1 Accuracy reached {current_top1:.2%}. "
                          f"Stopping to preserve RL exploration capabilities.")

                    # Store checkpoints
                    model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (sl_model, sl_opt_state))
                    chpt_0 = os.path.join(ckpt_dir, f"{epoch:06d}.ckpt")
                    cpu_model_snapshot = jax.device_get(model_0)
                    with open(chpt_0, "wb") as f:
                        dic = {
                            "config": config,
                            "model": cpu_model_snapshot,
                            "opt_state": jax.device_get(opt_state_0),
                            "epoch": epoch,
                            "step": global_step,
                            "hours": hours,
                            "pgx.__version__": pgx.__version__,
                            "env_id": env.id,
                            "env_version": env.version,
                        }
                        pickle.dump(dic, f)
                        print(f"Saved {chpt_0}")
                    break
                    
                if current_entropy < 1.5:
                    print(f"\n[Early Stopping] Policy Entropy dropped to {current_entropy:.2f}. "
                          f"Stopping to prevent policy collapse before self-play.")

                    # Store checkpoints
                    model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (sl_model, sl_opt_state))
                    chpt_0 = os.path.join(ckpt_dir, f"{epoch:06d}.ckpt")
                    cpu_model_snapshot = jax.device_get(model_0)
                    with open(chpt_0, "wb") as f:
                        dic = {
                            "config": config,
                            "model": cpu_model_snapshot,
                            "opt_state": jax.device_get(opt_state_0),
                            "epoch": epoch,
                            "step": global_step,
                            "hours": hours,
                            "pgx.__version__": pgx.__version__,
                            "env_id": env.id,
                            "env_version": env.version,
                        }
                        pickle.dump(dic, f)
                        print(f"Saved {chpt_0}")
                    break

            jax.effects_barrier()

            # Unwrap parameters back to CPU configurations
            model = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], sl_model))
            opt_state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], sl_opt_state))

            # Save structural checkpoint so this phase can be skipped on future runs
            sl_weights = os.path.join("checkpoints", "sl_weights.pkl")
            with open(sl_weights, "wb") as f:
                pickle.dump(model, f)
            print(f"Supervised training complete! Weights exported to {sl_weights}.")

    # Train via RL
    else:
        # Track the raw un-replicated CPU parameter formats for on-demand evaluation swapping
        # This keeps our memory consumption footprint isolated strictly to host CPU RAM
        champion_model_cpu = jax.device_get(model)

        # Load a champion
        if config.champion is not None:
            if not os.path.exists(config.champion):
                print(f"Champion not found at {config.champion}.")
            else:
                print("Found champion checkpoint! Loading weights...")
                with open(config.champion, "rb") as f:
                    champion_model_cpu = pickle.load(f)

        # Replicate only our active training model and opt_state variables across devices
        model = jax.tree_util.tree_map(
            lambda x: jax.device_put(jnp.stack([x] * num_devices), sharding), model
        )
        opt_state = jax.tree_util.tree_map(
            lambda x: jax.device_put(jnp.stack([x] * num_devices), sharding), opt_state
        )

        # Initialize logging dict
        iteration: int = 0
        hours: float = 0.0
        frames: int = 0
        metrics = {"iteration": 0, "hours": hours, "frames": frames}

        data_loader = make_data_loader(config, num_devices)

        rng_key = jax.random.PRNGKey(config.seed)
        while True:
            # Evaluation
            if iteration % config.eval_interval == 0:
                rng_key, eval_key = jax.random.split(rng_key)
                keys = jax.random.split(eval_key, num_devices)

                # 1. Temporarily mirror the baseline weights to matching multi-device sharded layouts 
                # only for the duration of this evaluation window.
                champion_model_sharded = jax.tree_util.tree_map(
                    lambda x: jax.device_put(jnp.stack([x] * num_devices), sharding), champion_model_cpu
                )

                # 2. Execute the evaluation steps
                R = evaluate(keys, model, champion_model_sharded)

                # 3. Explicitly delete the sharded reference to instantly clear VRAM allocations
                del champion_model_sharded

                win_rate = ((R == 1).sum() / R.size).item()
                draw_rate = ((R == 0).sum() / R.size).item()
                loss_rate = ((R == -1).sum() / R.size).item()

                metrics.update(
                    {
                        "eval/vs_baseline/avg_R": R.mean().item(),
                        "eval/vs_baseline/win_rate": win_rate,
                        "eval/vs_baseline/draw_rate": draw_rate,
                        "eval/vs_baseline/lose_rate": loss_rate,
                    }
                )

                # Store checkpoints
                model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
                chpt_0 = os.path.join(ckpt_dir, f"{iteration:06d}.ckpt")

                cpu_model_snapshot = jax.device_get(model_0)

                with open(chpt_0, "wb") as f:
                    dic = {
                        "config": config,
                        "rng_key": rng_key,
                        "model": cpu_model_snapshot,
                        "opt_state": jax.device_get(opt_state_0),
                        "iteration": iteration,
                        "frames": frames,
                        "hours": hours,
                        "pgx.__version__": pgx.__version__,
                        "env_id": env.id,
                        "env_version": env.version,
                    }
                    pickle.dump(dic, f)

                # Upgrades the baseline when the network shows meaningful improvement
                if win_rate > 0.55:
                    print(f"Model {chpt_0} dethroned the champion!")
                    champion_model_cpu = cpu_model_snapshot
                    champion_path = os.path.join(ckpt_dir, "champion")
                    with open(champion_path, "w", encoding="utf-8") as f:
                        f.write(chpt_0)

            print(metrics)
            wandb.log(metrics)

            if iteration >= config.max_num_iters:
                break

            iteration += 1
            metrics = {"iteration": iteration}
            st = time.time()

            # Selfplay
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            data: SelfplayOutput = selfplay(model, keys)
            samples: Sample = compute_loss_input(data)
            samples = jax.device_get(samples)

            # Total steps per trajectory is the sum of unmasked steps (where value_mask is True)
            # samples.mask shape: (num_devices, batch_size, max_num_steps)
            steps_per_game = jax.device_get(samples.mask.sum(axis=2))  
            avg_game_length = float(steps_per_game.mean())
            min_game_length = float(steps_per_game.min())
            frames += int(steps_per_game.sum())

            next(data_loader)
            minibatches = data_loader.send(samples)

            if minibatches is None:
                # Not enough frames have accumulated yet! Skip training optimization step this iteration
                print(f"Accumulating frames... (Current Global Count: {frames})")
                metrics.update({"hours": hours, "frames": frames, "selfplay/avg_game_length": avg_game_length, "speed/fps": 0})
                wandb.log(metrics)
                continue

            # Training
            policy_losses, value_losses = [], []
            sharded_minibatches = [jax.device_put(b) for b in minibatches]

            for minibatch in sharded_minibatches:
                # Explicitly push the single clean minibatch to the GPU execution memory grids
                model, opt_state, policy_loss, value_loss = train(model, opt_state, minibatch)
                policy_losses.append(policy_loss.mean().item())
                value_losses.append(value_loss.mean().item())

            policy_loss = sum(policy_losses) / len(policy_losses)
            value_loss = sum(value_losses) / len(value_losses)

            et = time.time()
            step_time = et - st
            hours += step_time / 3600

            metrics.update(
                {
                    "hours": hours,
                    "frames": frames,
                    "train/policy_loss": policy_loss,
                    "train/value_loss": value_loss,
                    "selfplay/avg_game_length": avg_game_length,
                    "selfplay/min_game_length": min_game_length,
                    "speed/fps": int(len(minibatches) * config.training_batch_size / (et - st + 1e-8))
                }
            )
