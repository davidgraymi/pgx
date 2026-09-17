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

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

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
    max_num_iters: int = 200
    replay_buffer_capacity: int = 50000
    # network params
    num_channels: int = 64
    num_layers: int = 4
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 16
    num_simulations: int = 64
    max_num_steps: int = 256
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    # training params
    training_batch_size: int = 256
    learning_rate: float = 0.0003
    load_ckpt: str | None = None
    training_mode: str = "pipeline"
    sl_dataset_path: str = "data/sl_dataset.npz"
    sl_validation_path: str = "data/sl_validation.npz"
    supervised_epochs: int = 20
    supervised_validation_patience: int = 5
    supervised_validation_min_delta: float = 0.01
    supervised_label_smoothing: float = 0.05
    supervised_weight_decay: float = 0.0001
    supervised_value_loss_weight: float = 0.5
    supervised_eval_games: int = 512
    supervised_min_random_score: float = 0.55
    require_random_win: bool = True
    # eval params
    eval_interval: int = 10
    eval_games: int = 512
    eval_batch_size: int = 8
    rl_regression_patience: int = 3
    rl_regression_min_delta: float = 0.01
    replay_buffer_path: str | None = None
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
supervised_optimizer = optax.adamw(
    learning_rate=config.learning_rate,
    weight_decay=config.supervised_weight_decay,
)


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


def search_action(model, rng_key, state):
    model_params, model_state = model
    (logits, value), _ = forward.apply(model_params, model_state, state.observation, is_eval=True)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
    policy_output = mctx.gumbel_muzero_policy(
        params=model,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=config.num_simulations,
        invalid_actions=~state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=1.0,
    )
    return jnp.argmax(policy_output.action_weights, axis=-1)


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    bootstrap_value: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray
    max_visits: jnp.ndarray
    legal_pct: jnp.ndarray


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

        summary = policy_output.search_tree.summary()

        max_root_visits = jnp.max(summary.visit_counts, axis=-1)
        legal_percentage = jnp.mean(state.legal_action_mask.astype(jnp.float32), axis=-1)
        
        actor = state.current_player
        keys = jax.random.split(key2, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, policy_output.action, keys)
        (_, bootstrap_value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)
        
        return state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated,
            truncated=state.truncated,
            bootstrap_value=bootstrap_value,
            discount=discount,
            max_visits=max_root_visits,
            legal_pct=legal_percentage
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
    done = data.terminated | data.truncated
    terminals_before = jnp.cumsum(done, axis=0) - done
    value_mask = terminals_before == 0

    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    bootstrap_value = jnp.where(
        data.terminated[-1], 0.0, data.bootstrap_value[-1]
    )
    _, value_tgt = jax.lax.scan(
        body_fn,
        bootstrap_value,
        jnp.arange(config.max_num_steps),
    )
    value_tgt = value_tgt[::-1, :]

    return Sample(
        obs=data.obs,
        policy_tgt=data.action_weights,
        value_tgt=value_tgt,
        mask=value_mask,
    )


def rollout_metrics(data: SelfplayOutput, samples: Sample):
    terminated = np.asarray(jax.device_get(data.terminated))
    truncated = np.asarray(jax.device_get(data.truncated))
    mask = np.asarray(jax.device_get(samples.mask))
    value_targets = np.asarray(jax.device_get(samples.value_tgt))[mask]
    return {
        "selfplay/termination_rate": float(terminated.mean()),
        "selfplay/truncation_rate": float(truncated.mean()),
        "value_target/mean": float(value_targets.mean()) if value_targets.size else 0.0,
        "value_target/std": float(value_targets.std()) if value_targets.size else 0.0,
    }


def loss_fn(model_params, model_state, samples: Sample):
    (logits, value), model_state = forward.apply(
        model_params, model_state, samples.obs, is_eval=False
    )

    policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
    policy_loss = jnp.mean(policy_loss)

    value_loss = optax.l2_loss(value, samples.value_tgt)
    value_loss = jnp.mean(value_loss * samples.mask)

    avg_pred_value_magnitude = jnp.mean(jnp.abs(value))

    return policy_loss + value_loss, (model_state, policy_loss, value_loss, avg_pred_value_magnitude)


@partial(jax.pmap, axis_name="i")
def train(model, opt_state, data: Sample):
    model_params, model_state = model
    grads, (model_state, policy_loss, value_loss, val_magnitude) = jax.grad(loss_fn, has_aux=True)(
        model_params, model_state, data
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    updates, opt_state = optimizer.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    model = (model_params, model_state)
    return model, opt_state, policy_loss, value_loss, val_magnitude


@jax.pmap
def evaluate(rng_key, my_model, baseline_model, my_color):
    """Evaluates the live learning model against a stable baseline snapshot model."""
    my_model_params, my_model_state = my_model
    base_model_params, base_model_state = baseline_model

    key, subkey = jax.random.split(rng_key)
    batch_size = config.eval_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)
    my_player = state._player_order[jnp.arange(batch_size), my_color]

    def body_fn(val):
        key, state, R = val
        key, my_key, opponent_key = jax.random.split(key, 3)
        my_action = search_action((my_model_params, my_model_state), my_key, state)
        opponent_action = search_action((base_model_params, base_model_state), opponent_key, state)
        is_my_turn = state.current_player == my_player
        action = jnp.where(is_my_turn, my_action, opponent_action)
        state = jax.vmap(env.step)(state, action)
        R = R + state.rewards[jnp.arange(batch_size), my_player]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()), body_fn, (key, state, jnp.zeros(batch_size))
    )
    return R


@jax.jit
def evaluate_offline_metrics(forward_apply_fn, model_params, model_state, obs_batch, legal_masks_batch, target_actions_batch):
    """
    Computes policy accuracy and entropy metrics over a batch of human expert data.
    """
    # 1. Run the raw model forward evaluation pass right here
    (logits, _), _ = forward_apply_fn(model_params, model_state, obs_batch, is_eval=True)
    
    # 2. Mask illegal actions to ensure we evaluate valid distributions
    masked_logits = jnp.where(legal_masks_batch, logits, -1e9)
    probs = jax.nn.softmax(masked_logits, axis=-1)
    
    # 3. Calculate Policy Entropy
    safe_probs = jnp.where(legal_masks_batch, probs, 1.0)
    entropy = -jnp.sum(probs * jnp.log(safe_probs + 1e-8), axis=-1)
    mean_entropy = jnp.mean(entropy)
    
    # 4. Total Legal Probability Mass
    raw_probs = jax.nn.softmax(logits, axis=-1)
    legal_mass = jnp.mean(jnp.sum(raw_probs * legal_masks_batch, axis=-1))

    # 5. Top-1 and Top-5 Accuracy Checks
    top_1_predictions = jnp.argmax(probs, axis=-1)
    top_1_acc = jnp.mean(top_1_predictions == target_actions_batch)
    
    top_5_predictions = jnp.argsort(probs, axis=-1)[:, -5:]
    top_5_acc = jnp.mean(jnp.any(top_5_predictions == target_actions_batch[:, None], axis=-1))
    
    return {
        "val_top_1_accuracy": top_1_acc,
        "val_top_5_accuracy": top_5_acc,
        "val_policy_entropy": mean_entropy,
        "val_legal_move_mass": legal_mass
    }


@jax.pmap
def evaluate_vs_random(rng_key, my_model, my_color):
    """Evaluates the live learning model against a uniform random legal opponent."""
    my_model_params, my_model_state = my_model

    key, subkey = jax.random.split(rng_key)
    batch_size = config.eval_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)
    my_player = state._player_order[jnp.arange(batch_size), my_color]

    def body_fn(val):
        key, state, R = val
        key, search_key, random_key = jax.random.split(key, 3)
        model_action = search_action((my_model_params, my_model_state), search_key, state)
        # We assign an equal logit value (0.0) to all moves, then mask out illegal ones.
        # This creates a uniform distribution over only legal actions.
        random_logits = jnp.where(state.legal_action_mask, 0.0, jnp.finfo(jnp.float32).min)
        is_my_turn = state.current_player == my_player
        random_action = jax.random.categorical(random_key, random_logits, axis=-1)
        action = jnp.where(is_my_turn, model_action, random_action)
        state = jax.vmap(env.step)(state, action)
        
        # Accumulate rewards from the perspective of my_player
        R = R + state.rewards[jnp.arange(batch_size), my_player]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()), body_fn, (key, state, jnp.zeros(batch_size))
    )
    return R


def summarize_evaluation(results, colors):
    def summarize(selected_results):
        wins = int(np.sum(selected_results == 1))
        draws = int(np.sum(selected_results == 0))
        losses = int(np.sum(selected_results == -1))
        return {
            "games": int(selected_results.size),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": wins / selected_results.size,
            "draw_rate": draws / selected_results.size,
            "loss_rate": losses / selected_results.size,
            "score": (wins + 0.5 * draws) / selected_results.size,
        }

    stats = summarize(results)
    stats["by_color"] = {
        "white": summarize(results[colors == 0]),
        "black": summarize(results[colors == 1]),
    }
    return stats


def run_random_evaluation(rng_key, model, num_games):
    """Play balanced-color games against a uniform random legal opponent."""
    if config.eval_batch_size % num_devices != 0:
        raise ValueError("eval_batch_size must be divisible by the number of devices")
    if num_games % config.eval_batch_size != 0:
        raise ValueError("num_games must be divisible by eval_batch_size")

    local_batch_size = config.eval_batch_size // num_devices
    sharded_model = jax.tree_util.tree_map(
        lambda x: jax.device_put(jnp.stack([x] * num_devices), sharding), model
    )
    results = []
    result_players = []
    for start in range(0, num_games, config.eval_batch_size):
        rng_key, eval_key = jax.random.split(rng_key)
        keys = jax.random.split(eval_key, num_devices)
        colors = np.arange(start, start + config.eval_batch_size, dtype=np.int32) % 2
        colors = jax.device_put(colors.reshape(num_devices, local_batch_size))
        result = evaluate_vs_random(keys, sharded_model, colors)
        results.append(np.asarray(jax.device_get(result)).reshape(-1))
        result_players.append(np.asarray(colors).reshape(-1))

    del sharded_model
    results = np.concatenate(results)
    result_players = np.concatenate(result_players)

    return summarize_evaluation(results, result_players), rng_key


def run_baseline_evaluation(rng_key, model, baseline_model, num_games):
    if config.eval_batch_size % num_devices != 0:
        raise ValueError("eval_batch_size must be divisible by the number of devices")
    if num_games % config.eval_batch_size != 0:
        raise ValueError("num_games must be divisible by eval_batch_size")

    local_batch_size = config.eval_batch_size // num_devices
    sharded_model = jax.tree_util.tree_map(
        lambda x: jax.device_put(jnp.stack([x] * num_devices), sharding), model
    )
    sharded_baseline = jax.tree_util.tree_map(
        lambda x: jax.device_put(jnp.stack([x] * num_devices), sharding), baseline_model
    )
    results = []
    result_colors = []
    for start in range(0, num_games, config.eval_batch_size):
        rng_key, eval_key = jax.random.split(rng_key)
        keys = jax.random.split(eval_key, num_devices)
        colors = np.arange(start, start + config.eval_batch_size, dtype=np.int32) % 2
        colors = jax.device_put(colors.reshape(num_devices, local_batch_size))
        result = evaluate(keys, sharded_model, sharded_baseline, colors)
        results.append(np.asarray(jax.device_get(result)).reshape(-1))
        result_colors.append(np.asarray(colors).reshape(-1))

    del sharded_model, sharded_baseline
    return summarize_evaluation(np.concatenate(results), np.concatenate(result_colors)), rng_key


def report_random_evaluation(rng_key, model, num_games, label):
    stats, rng_key = run_random_evaluation(rng_key, model, num_games)
    wandb.log({
        f"{label}/games": stats["games"],
        f"{label}/win_rate": stats["win_rate"],
        f"{label}/draw_rate": stats["draw_rate"],
        f"{label}/loss_rate": stats["loss_rate"],
        f"{label}/score": stats["score"],
        f"{label}/white/score": stats["by_color"]["white"]["score"],
        f"{label}/black/score": stats["by_color"]["black"]["score"],
    })
    print(
        f"[{label.upper()}] {stats['games']} games vs random | "
        f"W/D/L {stats['wins']}/{stats['draws']}/{stats['losses']} | "
        f"score {stats['score']:.2%} | "
        f"White {stats['by_color']['white']['score']:.2%} | "
        f"Black {stats['by_color']['black']['score']:.2%}"
    )
    return stats, rng_key


def supervised_loss_fn(model_params, model_state, obs, target_actions, target_values):
    (logits, value), model_state = forward.apply(
        model_params, model_state, obs, is_eval=False
    )
    action_targets = jax.nn.one_hot(target_actions, logits.shape[-1], dtype=logits.dtype)
    smoothing = config.supervised_label_smoothing
    action_targets = (1.0 - smoothing) * action_targets + smoothing / logits.shape[-1]
    policy_loss = jnp.mean(optax.softmax_cross_entropy(logits, action_targets))
    value_loss = jnp.mean(optax.l2_loss(value, target_values))
    loss = policy_loss + config.supervised_value_loss_weight * value_loss
    predictions = jnp.argmax(logits, axis=-1)
    top1_acc = jnp.mean(predictions == target_actions)
    _, top5_indices = jax.lax.top_k(logits, k=5)
    top5_acc = jnp.mean(jnp.any(top5_indices == target_actions[:, None], axis=-1))
    probs = jax.nn.softmax(logits, axis=-1)
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
    metrics = {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "entropy": jnp.mean(entropy),
    }
    return loss, (model_state, metrics)


@partial(jax.pmap, axis_name="i")
def supervised_train_step(model, opt_state, obs, target_actions, target_values):
    model_params, model_state = model
    (loss, (model_state, metrics)), grads = jax.value_and_grad(supervised_loss_fn, has_aux=True)(
        model_params, model_state, obs, target_actions, target_values
    )
    grads = jax.lax.pmean(grads, axis_name="i")
    metrics = jax.lax.pmean(metrics, axis_name="i")
    updates, opt_state = supervised_optimizer.update(grads, opt_state, params=model_params)
    model_params = optax.apply_updates(model_params, updates)
    return (model_params, model_state), opt_state, metrics


@jax.pmap
def supervised_validation_step(model, obs, target_actions, target_values):
    model_params, model_state = model
    (logits, value), _ = forward.apply(model_params, model_state, obs, is_eval=True)
    policy_loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_actions)
    value_loss = optax.l2_loss(value, target_values)
    _, top5_indices = jax.lax.top_k(logits, k=5)
    return {
        "loss": policy_loss + config.supervised_value_loss_weight * value_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "top1_accuracy": jnp.argmax(logits, axis=-1) == target_actions,
        "top5_accuracy": jnp.any(top5_indices == target_actions[:, None], axis=-1),
    }


def load_supervised_dataset(path):
    if path.endswith(".npz"):
        with np.load(path) as dataset:
            observations = np.asarray(dataset["observations"])
            actions = np.asarray(dataset["actions"])
            values = np.asarray(dataset["values"]) if "values" in dataset.files else np.zeros(actions.shape[0], dtype=np.float32)
            phases = np.asarray(dataset["sample_phases"]) if "sample_phases" in dataset.files else np.full(actions.shape[0], "unknown")
            players = np.asarray(dataset["players"]) if "players" in dataset.files else np.full(actions.shape[0], -1, dtype=np.int8)
            return observations, actions, values, phases, players
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    observations = np.asarray(dataset["observations"])
    actions = np.asarray(dataset["actions"])
    values = np.asarray(dataset.get("values", np.zeros(actions.shape[0], dtype=np.float32)))
    phases = np.asarray(dataset.get("sample_phases", np.full(actions.shape[0], "unknown")))
    players = np.asarray(dataset.get("players", np.full(actions.shape[0], -1, dtype=np.int8)))
    return observations, actions, values, phases, players

def run_supervised_training(config, model, opt_state, num_devices, sharding, ckpt_dir):
    """Train the policy and value heads on expert games with held-out validation."""
    sl_dataset_path = config.sl_dataset_path
    if not os.path.exists(sl_dataset_path) and sl_dataset_path.endswith(".npz"):
        sl_dataset_path = sl_dataset_path[:-4] + ".pkl"
    if not os.path.exists(sl_dataset_path):
        print(f"Dataset not found at {sl_dataset_path}. Skipping pre-training.")
        return model, opt_state

    print("Found dataset! Commencing Supervised Pre-Training...")
    sl_obs, sl_actions, sl_values, _, _ = load_supervised_dataset(sl_dataset_path)
    validation_obs = validation_actions = validation_values = None
    validation_phases = validation_players = None
    validation_path = config.sl_validation_path
    if not os.path.exists(validation_path) and validation_path.endswith(".npz"):
        validation_path = validation_path[:-4] + ".pkl"
    if os.path.exists(validation_path):
        (
            validation_obs,
            validation_actions,
            validation_values,
            validation_phases,
            validation_players,
        ) = load_supervised_dataset(validation_path)
        print(f"Found held-out validation set with {validation_obs.shape[0]} positions.")
    else:
        print(f"Validation dataset not found at {validation_path}.")

    sl_model = jax.tree_util.tree_map(
        lambda x: jax.device_put(jnp.stack([x] * num_devices), sharding), model
    )
    supervised_opt_state = supervised_optimizer.init(params=model[0])
    sl_opt_state = jax.tree_util.tree_map(
        lambda x: jax.device_put(jnp.stack([x] * num_devices), sharding), supervised_opt_state
    )

    num_samples = sl_obs.shape[0]
    sl_batch_size = min(config.training_batch_size, num_samples)
    sl_batch_size -= sl_batch_size % num_devices
    if sl_batch_size == 0:
        raise ValueError("The supervised dataset must contain at least one sample per device")
    num_batches = num_samples // sl_batch_size
    sl_rng = np.random.default_rng(config.seed)
    global_step = 0
    hours = 0.0
    best_model = best_opt_state = None
    best_validation_loss = float("inf")
    best_validation_epoch = 0
    validation_epochs_without_improvement = 0

    for epoch in range(config.supervised_epochs):
        indices = sl_rng.permutation(num_samples)
        epoch_loss = 0.0
        for batch_number in range(num_batches):
            st = time.time()
            batch_idx = indices[batch_number * sl_batch_size : (batch_number + 1) * sl_batch_size]
            raw_obs = sl_obs[batch_idx]
            raw_actions = sl_actions[batch_idx]
            raw_values = sl_values[batch_idx]
            batch_obs = jax.device_put(
                raw_obs.reshape(num_devices, sl_batch_size // num_devices, *raw_obs.shape[1:])
            )
            batch_actions = jax.device_put(
                raw_actions.reshape(num_devices, sl_batch_size // num_devices)
            )
            batch_values = jax.device_put(
                raw_values.reshape(num_devices, sl_batch_size // num_devices)
            )
            sl_model, sl_opt_state, sharded_metrics = supervised_train_step(
                sl_model, sl_opt_state, batch_obs, batch_actions, batch_values
            )
            processed_metrics = {
                f"supervised/{key}": float(jax.device_get(jnp.mean(value)))
                for key, value in sharded_metrics.items()
            }
            epoch_loss += processed_metrics["supervised/loss"]
            global_step += 1
            hours += (time.time() - st) / 3600
            wandb.log({
                "supervised/epoch": epoch + 1,
                "supervised/global_step": global_step,
                "hours": hours,
                **processed_metrics,
            })

        avg_epoch_loss = epoch_loss / num_batches
        epoch_log = {
            "supervised/epoch_loss": avg_epoch_loss,
            "supervised/epoch": epoch + 1,
            "hours": hours,
        }
        if validation_obs is None:
            print(f"Supervised Epoch {epoch + 1} Complete | Average Loss: {avg_epoch_loss:.4f}")
            wandb.log(epoch_log)
            continue

        validation_batch_size = min(config.training_batch_size, validation_obs.shape[0])
        validation_batch_size -= validation_batch_size % num_devices
        if validation_batch_size == 0:
            raise ValueError("The validation dataset must contain at least one sample per device")
        validation_num_batches = validation_obs.shape[0] // validation_batch_size
        validation_values_by_metric = {key: [] for key in ("loss", "policy_loss", "value_loss", "top1_accuracy", "top5_accuracy")}
        validation_phase_labels = []
        validation_player_labels = []
        for batch_number in range(validation_num_batches):
            start = batch_number * validation_batch_size
            end = start + validation_batch_size
            batch_obs = jax.device_put(
                validation_obs[start:end].reshape(
                    num_devices, validation_batch_size // num_devices, *validation_obs.shape[1:]
                )
            )
            batch_actions = jax.device_put(
                validation_actions[start:end].reshape(num_devices, validation_batch_size // num_devices)
            )
            batch_values = jax.device_put(
                validation_values[start:end].reshape(num_devices, validation_batch_size // num_devices)
            )
            metrics = supervised_validation_step(sl_model, batch_obs, batch_actions, batch_values)
            for key, value in metrics.items():
                validation_values_by_metric[key].append(np.asarray(jax.device_get(value)).reshape(-1))
            validation_phase_labels.append(validation_phases[start:end])
            validation_player_labels.append(validation_players[start:end])

        validation_arrays = {
            key: np.concatenate(values) for key, values in validation_values_by_metric.items()
        }
        validation_metrics = {
            f"supervised/validation_{key}": float(np.mean(value))
            for key, value in validation_arrays.items()
        }
        validation_phase_labels = np.concatenate(validation_phase_labels)
        validation_player_labels = np.concatenate(validation_player_labels)
        for phase in ("opening", "middlegame", "endgame"):
            mask = validation_phase_labels == phase
            if np.any(mask):
                epoch_log[f"supervised/validation_phase/{phase}/loss"] = float(np.mean(validation_arrays["loss"][mask]))
                epoch_log[f"supervised/validation_phase/{phase}/top1_accuracy"] = float(np.mean(validation_arrays["top1_accuracy"][mask]))
        for player, name in ((0, "white"), (1, "black")):
            mask = validation_player_labels == player
            if np.any(mask):
                epoch_log[f"supervised/validation_color/{name}/loss"] = float(np.mean(validation_arrays["loss"][mask]))
                epoch_log[f"supervised/validation_color/{name}/top1_accuracy"] = float(np.mean(validation_arrays["top1_accuracy"][mask]))
        current_validation_loss = validation_metrics["supervised/validation_loss"]
        if current_validation_loss < best_validation_loss - config.supervised_validation_min_delta:
            best_validation_loss = current_validation_loss
            best_validation_epoch = epoch + 1
            validation_epochs_without_improvement = 0
            best_model = jax.tree_util.tree_map(
                lambda value: np.array(jax.device_get(value[0]), copy=True), sl_model
            )
            best_opt_state = jax.tree_util.tree_map(
                lambda value: np.array(jax.device_get(value[0]), copy=True), sl_opt_state
            )
            epoch_log["supervised/validation_best_loss"] = best_validation_loss
        else:
            validation_epochs_without_improvement += 1
        epoch_log["supervised/validation_epochs_without_improvement"] = validation_epochs_without_improvement
        epoch_log.update(validation_metrics)
        print(
            f"Supervised Epoch {epoch + 1} Complete | Loss: {avg_epoch_loss:.4f} | "
            f"Validation Loss: {validation_metrics['supervised/validation_loss']:.4f} | "
            f"Validation Top-1: {validation_metrics['supervised/validation_top1_accuracy']:.2%}"
        )
        wandb.log(epoch_log)
        if validation_epochs_without_improvement >= config.supervised_validation_patience:
            print(
                f"Validation loss stopped improving for {config.supervised_validation_patience} "
                f"epochs; restoring epoch {best_validation_epoch}."
            )
            break

    jax.effects_barrier()
    if best_model is not None:
        model, opt_state = best_model, best_opt_state
    else:
        model = jax.tree_util.tree_map(lambda x: jax.device_get(x[0]), sl_model)
        opt_state = jax.tree_util.tree_map(lambda x: jax.device_get(x[0]), sl_opt_state)
    sl_weights = os.path.join(ckpt_dir, "sl_weights.pkl")
    with open(sl_weights, "wb") as f:
        pickle.dump(model, f)
    return model, opt_state

def run_rl_training(config, model, opt_state, num_devices, sharding, ckpt_dir, iteration, frames, hours, rng_key, buffer_state, replay_buffer_path):
    """Executes the core MuZero RL continuum using dynamic history sampling."""
    # sl_dataset_path = os.path.join("data", "sl_dataset.pkl")
    # has_validation_data = os.path.exists(sl_dataset_path)
    # val_obs, val_actions, val_masks = None, None, None

    # if has_validation_data:
    #     print("\n=== Loading Offline Lichess Dataset for Periodic Validation ===")
    #     with open(sl_dataset_path, "rb") as f:
    #         val_dataset = pickle.load(f)
        
    #     # Take a fixed subset (e.g., 2048 positions) to keep evaluation extremely fast
    #     val_subset_size = min(2048, val_dataset["observations"].shape[0])
        
    #     # We need legal move masks for the offline metrics function. 
    #     # We can extract them by mapping env.init or evaluating the current state, 
    #     # but since pgx observations don't store the raw mask explicitly, we can generate 
    #     # a dummy array or filter our batch. For offline mapping against human targets,
    #     # we can pass an all-True mask or a structural mask if available.
    #     # Let's create an all-True fallback mask if the dataset doesn't have it.
    #     val_obs = np.asarray(val_dataset["observations"][:val_subset_size])
    #     val_actions = np.asarray(val_dataset["actions"][:val_subset_size])
        
    #     # Shape: (val_subset_size, 4672) matching pgx.chess action space
    #     val_masks = np.ones((val_subset_size, env.num_actions), dtype=bool) 
        
    #     # Reshape data structures cleanly to shard across your PMAP devices
    #     val_obs = val_obs.reshape(num_devices, val_subset_size // num_devices, *val_obs.shape[1:])
    #     val_actions = val_actions.reshape(num_devices, val_subset_size // num_devices)
    #     val_masks = val_masks.reshape(num_devices, val_subset_size // num_devices, env.num_actions)
        
    #     print(f"Loaded {val_subset_size} offline validation positions successfully.")
    # else:
    #     print("\n[Warning] sl_dataset.pkl not found. Offline metric tracking will be skipped.")

    # Load a champion
    champion_model_cpu = jax.device_get(model)
    if config.champion is not None and os.path.exists(config.champion):
        print("Found champion checkpoint! Loading weights...")
        with open(config.champion, "rb") as f:
            champion_data = pickle.load(f)
            
        # Check if the champion file is a dictionary or raw parameters
        if isinstance(champion_data, dict) and "model" in champion_data:
            print("Extracting raw model parameters from champion dictionary snapshot.")
            champion_model_cpu = champion_data["model"]
        else:
            print("Loading raw parameters from champion file.")
            champion_model_cpu = champion_data

    # Shard model elements for multi-GPU pmap operations
    model = jax.tree_util.tree_map(
        lambda x: jax.device_put(jnp.stack([x] * num_devices), sharding), model
    )
    opt_state = jax.tree_util.tree_map(
        lambda x: jax.device_put(jnp.stack([x] * num_devices), sharding), opt_state
    )

    data_loader = make_data_loader(config, num_devices)
    next(data_loader)
    loader_ready = True

    if buffer_state is None and replay_buffer_path and os.path.exists(replay_buffer_path):
        with open(replay_buffer_path, "rb") as f:
            buffer_state = pickle.load(f)

    if buffer_state is not None:
        print("\n=== Found saved experience history. Restoring Replay Buffer Workspace ===")
        data_loader.send(buffer_state)
        next(data_loader)
        print("Replay Buffer successfully primed with historical experience arrays.")

    print("\n=== Initializing Replay Buffer Warmup ===")
    warmup_batches = None
    buffer_metrics = {}
    warmup_round = 0
    
    while warmup_batches is None:
        st = time.time()
        
        # Generate raw un-incremented selfplay data
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data: SelfplayOutput = selfplay(model, keys)
        samples = compute_loss_input(data)

        # Track frames experienced during warmup
        steps_per_game = jax.device_get(samples.mask.sum(axis=2))
        avg_game_length = float(steps_per_game.mean())
        frames += int(steps_per_game.sum())
        rollout_log = rollout_metrics(data, samples)
        
        # Send data to see if buffer clears its internal warmup size thresholds
        if not loader_ready:
            next(data_loader)
        warmup_batches, buffer_metrics = data_loader.send(samples)
        loader_ready = False
        warmup_round += 1
        
        et = time.time()
        hours += (et - st) / 3600
        
        print(
            f"RL Warmup {warmup_round} Complete | Frames: {frames} | "
            f"Avg Game Length: {avg_game_length:.2f} | "
            f"Replay: {buffer_metrics.get('replay_buffer/size', 0)}"
        )
        
        # Log purely environmental data during the warmup stage
        wandb.log({
            "iteration": 0,  # Explicitly locked at 0
            "hours": hours,
            "frames": frames,
            "selfplay/avg_game_length": avg_game_length,
            "speed/fps": 0,
            **rollout_log,
            **buffer_metrics
        })

    print(f"=== Warmup Complete! Replay buffer primed with {frames} frames. Commencing Active RL Loop ===\n")

    if replay_buffer_path:
        next(data_loader)
        save_replay_buffer(replay_buffer_path, data_loader.send("SAVE"))

    minibatches = warmup_batches
    best_random_score = -np.inf
    random_regression_wait = 0
    best_random_iteration = 0

    while True:
        if iteration % config.eval_interval == 0:
            current_model_cpu = jax.tree_util.tree_map(lambda x: x[0], model)
            rng_key, baseline_key, random_key = jax.random.split(rng_key, 3)
            baseline_stats, _ = run_baseline_evaluation(
                baseline_key,
                jax.device_get(current_model_cpu),
                champion_model_cpu,
                config.eval_games,
            )
            random_stats, rng_key = run_random_evaluation(
                random_key, jax.device_get(current_model_cpu), config.eval_games
            )
            if random_stats["score"] > best_random_score + config.rl_regression_min_delta:
                best_random_score = random_stats["score"]
                best_random_iteration = iteration
                random_regression_wait = 0
            else:
                random_regression_wait += 1

            log = {
                "eval/vs_baseline/avg_R": baseline_stats["win_rate"] - baseline_stats["loss_rate"],
                "eval/vs_baseline/win_rate": baseline_stats["win_rate"],
                "eval/vs_baseline/draw_rate": baseline_stats["draw_rate"],
                "eval/vs_baseline/lose_rate": baseline_stats["loss_rate"],
                "eval/vs_baseline/score": baseline_stats["score"],
                "eval/vs_baseline/games": baseline_stats["games"],
                "eval/vs_baseline/white/score": baseline_stats["by_color"]["white"]["score"],
                "eval/vs_baseline/black/score": baseline_stats["by_color"]["black"]["score"],
                "eval/vs_random/avg_R": random_stats["score"],
                "eval/vs_random/win_rate": random_stats["win_rate"],
                "eval/vs_random/draw_rate": random_stats["draw_rate"],
                "eval/vs_random/lose_rate": random_stats["loss_rate"],
                "eval/vs_random/games": random_stats["games"],
                "eval/vs_random/best_score": best_random_score,
                "eval/vs_random/regression_wait": random_regression_wait,
                "iteration": iteration, "frames": frames, "hours": hours
            }

            # if has_validation_data:
            #     # Extract a clean, single-device parameter slice from your multi-device model weights
            #     model_params, model_state = model
                
            #     # Define a pure local forward function that does not close over sharded parameters
            #     def batch_forward(obs):
            #         (logits, _), _ = forward.apply(model_params, model_state, state.observation, is_eval=True)
            #         return logits

            #     # Reconstruct your validation data arrays without the leading device axis
            #     flat_val_obs = val_obs.reshape(-1, *val_obs.shape[2:])
            #     flat_val_masks = val_masks.reshape(-1, *val_masks.shape[2:])
            #     flat_val_actions = val_actions.reshape(-1)

            #     # FIX: Explicitly pass batch_forward as the first positional argument!
            #     val_outputs = evaluate_offline_metrics(
            #         batch_forward, 
            #         flat_val_obs, 
            #         flat_val_masks, 
            #         flat_val_actions
            #     )

            #     # Pull the results back to the logging dictionary
            #     log["eval/offline/top1_accuracy"] = float(jax.device_get(val_outputs["val_top_1_accuracy"]))
            #     log["eval/offline/top5_accuracy"] = float(jax.device_get(val_outputs["val_top_5_accuracy"]))
            #     log["eval/offline/policy_entropy"] = float(jax.device_get(val_outputs["val_policy_entropy"]))
            #     log["eval/offline/legal_move_mass"] = float(jax.device_get(val_outputs["val_legal_move_mass"]))
            
            wandb.log(log)
            print(
                f"RL Evaluation {iteration} | "
                f"Baseline W/D/L {baseline_stats['wins']}/{baseline_stats['draws']}/{baseline_stats['losses']} "
                f"Score: {baseline_stats['score']:.2%} | "
                f"Random W/D/L {random_stats['wins']}/{random_stats['draws']}/{random_stats['losses']} "
                f"Score: {random_stats['score']:.2%} "
                f"(White: {random_stats['by_color']['white']['score']:.2%}, "
                f"Black: {random_stats['by_color']['black']['score']:.2%})"
            )

            next(data_loader)
            current_buffer_snapshot = data_loader.send("SAVE")
            if replay_buffer_path:
                save_replay_buffer(replay_buffer_path, current_buffer_snapshot)

            # Store iteration metrics via our new unified save handler
            model_0, opt_state_0 = jax.tree_util.tree_map(lambda x: x[0], (model, opt_state))
            chpt_0 = os.path.join(ckpt_dir, f"{iteration:06d}.ckpt")
            save_checkpoint(ckpt_dir, f"{iteration:06d}.ckpt", {
                "config": config,
                "rng_key": rng_key,
                "model": jax.device_get(model_0),
                "opt_state": jax.device_get(opt_state_0),
                "iteration": iteration,
                "frames": frames,
                "hours": hours,
                "replay_buffer_path": replay_buffer_path,
                "pgx.__version__": pgx.__version__,
                "env_id": env.id,
                "env_version": env.version,
            })

            if iteration == best_random_iteration:
                save_checkpoint(ckpt_dir, "best_random.ckpt", {
                    "config": config,
                    "rng_key": rng_key,
                    "model": jax.device_get(model_0),
                    "opt_state": jax.device_get(opt_state_0),
                    "iteration": iteration,
                    "frames": frames,
                    "hours": hours,
                    "replay_buffer_path": replay_buffer_path,
                    "pgx.__version__": pgx.__version__,
                    "env_id": env.id,
                    "env_version": env.version,
                })

            if baseline_stats["win_rate"] > 0.55:
                print(f"Model {chpt_0} dethroned the champion!")
                champion_model_cpu = jax.device_get(model_0)
                champion_path = os.path.join(ckpt_dir, "champion")
                with open(champion_path, "w", encoding="utf-8") as f:
                    f.write(chpt_0)

            if random_regression_wait >= config.rl_regression_patience:
                print(
                    f"Random-opponent score failed to improve for "
                    f"{config.rl_regression_patience} evaluations; "
                    f"best score was {best_random_score:.2%} at iteration {best_random_iteration}."
                )
                break

        if iteration >= config.max_num_iters:
            break

        loop_start_time = time.time()
        training_start_time = loop_start_time
        policy_losses, value_losses, val_magnitudes = [], [], []
        sharded_minibatches = [jax.device_put(b) for b in minibatches]

        for minibatch in sharded_minibatches:
            model, opt_state, policy_loss, value_loss, v_mag = train(model, opt_state, minibatch)
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(value_loss.mean().item())
            val_magnitudes.append(v_mag.mean().item())

        training_elapsed_time = time.time() - training_start_time

        iteration += 1

        # Re-Generate Fresh Samples for the next iteration cycle
        selfplay_start_time = time.time()
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data: SelfplayOutput = selfplay(model, keys)
        samples = compute_loss_input(data)
        samples = jax.device_get(samples)

        gpu_avg_max_visits = jnp.mean(data.max_visits[samples.mask])
        gpu_avg_legal_pct = jnp.mean(data.legal_pct[samples.mask])

        avg_max_visits = float(jax.device_get(gpu_avg_max_visits))
        avg_legal_pct = float(jax.device_get(gpu_avg_legal_pct))

        steps_per_game = samples.mask.sum(axis=2)
        avg_game_length = float(steps_per_game.mean())
        selfplay_frames = int(steps_per_game.sum())
        frames += selfplay_frames
        rollout_terminated = int(np.asarray(jax.device_get(data.terminated)).sum())
        rollout_truncated = int(np.asarray(jax.device_get(data.truncated)).sum())
        rollout_steps = int(np.asarray(jax.device_get(data.terminated)).size)
        rollout_log = rollout_metrics(data, samples)

        next(data_loader)
        minibatches, buffer_metrics = data_loader.send(samples)
        
        # If the latest game ended so fast that no new full batch fits,
        # forcefully gather more data until the buffer can yield a batch
        while minibatches is None:
            rng_key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, num_devices)
            data: SelfplayOutput = selfplay(model, keys)
            samples = compute_loss_input(data)
            steps_per_game = jax.device_get(samples.mask.sum(axis=2))
            selfplay_frames += int(steps_per_game.sum())
            frames += int(steps_per_game.sum())
            rollout_terminated += int(np.asarray(jax.device_get(data.terminated)).sum())
            rollout_truncated += int(np.asarray(jax.device_get(data.truncated)).sum())
            rollout_steps += int(np.asarray(jax.device_get(data.terminated)).size)
            next(data_loader)
            minibatches, buffer_metrics = data_loader.send(samples)

        loop_elapsed_time = time.time() - loop_start_time
        selfplay_elapsed_time = time.time() - selfplay_start_time
        hours += loop_elapsed_time / 3600
        rollout_log["selfplay/termination_rate"] = rollout_terminated / rollout_steps
        rollout_log["selfplay/truncation_rate"] = rollout_truncated / rollout_steps

        policy_loss = sum(policy_losses) / len(policy_losses)
        value_loss = sum(value_losses) / len(value_losses)
        value_magnitude = sum(val_magnitudes) / len(val_magnitudes)
        updates_completed = len(sharded_minibatches)
        selfplay_fps = int(selfplay_frames / (selfplay_elapsed_time + 1e-8))
        train_fps = int((updates_completed * config.training_batch_size) / (training_elapsed_time + 1e-8))
        loop_fps = int(selfplay_frames / (loop_elapsed_time + 1e-8))
        log = {
            "iteration": iteration,
            "hours": hours,
            "frames": frames,
            "train/policy_loss": policy_loss,
            "train/value_loss": value_loss,
            "train/value_prediction_magnitude": value_magnitude,
            "train/updates": updates_completed,
            "selfplay/avg_game_length": avg_game_length,
            "selfplay/search_confidence_max_visits": avg_max_visits,
            "selfplay/legal_moves_percentage": avg_legal_pct,
            "speed/fps": selfplay_fps,
            "speed/selfplay_fps": selfplay_fps,
            "speed/train_fps": train_fps,
            "speed/loop_fps": loop_fps,
            **rollout_log,
            **buffer_metrics,
        }
        wandb.log(log)
        print(
            f"RL Iteration {iteration} Complete | Policy Loss: {policy_loss:.4f} | "
            f"Value Loss: {value_loss:.4f} | Value Magnitude: {value_magnitude:.4f} | "
            f"Avg Game Length: {avg_game_length:.2f} | Self-play FPS: {selfplay_fps} | "
            f"Replay: {buffer_metrics.get('replay_buffer/size', 0)}/"
            f"{config.replay_buffer_capacity} | Hours: {hours:.2f}"
        )


def make_data_loader(config, num_devices):
    """Keep one preallocated host-side ring buffer and sample it for updates."""
    max_capacity = getattr(config, "replay_buffer_capacity", 100000)
    rng = np.random.default_rng(config.seed)
    batch_per_device = config.training_batch_size // num_devices
    buffer_obs = buffer_policy = buffer_value = None
    buffer_sequence = None
    buffer_size = 0
    write_index = 0
    total_inserted = 0

    def allocate(sample_obs, sample_policy, sample_value):
        return (
            np.empty((max_capacity, *sample_obs.shape[1:]), dtype=sample_obs.dtype),
            np.empty((max_capacity, *sample_policy.shape[1:]), dtype=np.float16),
            np.empty((max_capacity, *sample_value.shape[1:]), dtype=sample_value.dtype),
        )

    def snapshot():
        if buffer_size == 0:
            return {
                "buffer_obs": np.empty((0,), dtype=np.float16),
                "buffer_policy": np.empty((0,), dtype=np.float16),
                "buffer_value": np.empty((0,), dtype=np.float32),
            }
        if buffer_size < max_capacity:
            order = np.arange(buffer_size)
        else:
            order = np.concatenate((np.arange(write_index, max_capacity), np.arange(write_index)))
        return {
            "buffer_obs": buffer_obs[order],
            "buffer_policy": buffer_policy[order],
            "buffer_value": buffer_value[order],
        }

    while True:
        command_or_samples = yield
        if isinstance(command_or_samples, str) and command_or_samples == "SAVE":
            yield snapshot()
            continue

        if isinstance(command_or_samples, dict) and "buffer_obs" in command_or_samples:
            restored_obs = np.asarray(command_or_samples["buffer_obs"])
            restored_policy = np.asarray(command_or_samples["buffer_policy"])
            restored_value = np.asarray(command_or_samples["buffer_value"])
            buffer_size = min(restored_obs.shape[0], max_capacity)
            if buffer_size:
                buffer_obs, buffer_policy, buffer_value = allocate(
                    restored_obs[:1], restored_policy[:1], restored_value[:1]
                )
                buffer_sequence = np.arange(max_capacity, dtype=np.int64)
                buffer_obs[:buffer_size] = restored_obs[-buffer_size:]
                buffer_policy[:buffer_size] = restored_policy[-buffer_size:]
                buffer_value[:buffer_size] = restored_value[-buffer_size:]
                total_inserted = buffer_size
                write_index = buffer_size % max_capacity
            print(f"[Buffer Restore] Loaded {buffer_size} historical frames into RAM.")
            yield None
            continue

        samples = command_or_samples
        samples = jax.device_get(samples)
        valid_obs = samples.obs[samples.mask]
        valid_policy = samples.policy_tgt[samples.mask]
        valid_value = samples.value_tgt[samples.mask]

        if valid_obs.shape[0] and buffer_obs is None:
            buffer_obs, buffer_policy, buffer_value = allocate(valid_obs, valid_policy, valid_value)
            buffer_sequence = np.empty(max_capacity, dtype=np.int64)

        if valid_obs.shape[0]:
            if valid_obs.shape[0] >= max_capacity:
                valid_obs = valid_obs[-max_capacity:]
                valid_policy = valid_policy[-max_capacity:]
                valid_value = valid_value[-max_capacity:]
            count = valid_obs.shape[0]
            positions = (write_index + np.arange(count)) % max_capacity
            buffer_obs[positions] = valid_obs
            buffer_policy[positions] = valid_policy
            buffer_value[positions] = valid_value
            buffer_sequence[positions] = total_inserted + np.arange(count)
            write_index = (write_index + count) % max_capacity
            buffer_size = min(max_capacity, buffer_size + count)
            total_inserted += count

        buffer_metrics = {
            "replay_buffer/size": buffer_size,
            "replay_buffer/saturation_pct": buffer_size / max_capacity,
        }

        num_fresh_frames = valid_obs.shape[0]
        num_updates = num_fresh_frames // config.training_batch_size
        total_required_elements = num_updates * config.training_batch_size
        if num_updates == 0 or buffer_size < max(total_required_elements, config.training_batch_size * 2):
            yield None, buffer_metrics
            continue

        sample_indices = rng.choice(buffer_size, size=total_required_elements, replace=False)
        if buffer_size == max_capacity:
            sample_indices = (write_index + sample_indices) % max_capacity
        sample_ages = total_inserted - 1 - buffer_sequence[sample_indices]
        train_obs = buffer_obs[sample_indices]
        train_policy = buffer_policy[sample_indices]
        train_value = buffer_value[sample_indices]

        minibatches_obs = train_obs.reshape(num_updates, num_devices, batch_per_device, *train_obs.shape[1:])
        minibatches_policy = train_policy.reshape(num_updates, num_devices, batch_per_device, *train_policy.shape[1:])
        minibatches_value = train_value.reshape(num_updates, num_devices, batch_per_device, *train_value.shape[1:])
        batches_list = [
            Sample(obs=minibatches_obs[i], policy_tgt=minibatches_policy[i], value_tgt=minibatches_value[i],
                   mask=np.ones((num_devices, batch_per_device), dtype=bool))
            for i in range(num_updates)
        ]
        buffer_metrics["replay_buffer/sample_age_mean"] = float(np.mean(sample_ages))
        buffer_metrics["replay_buffer/sample_age_max"] = int(np.max(sample_ages))
        yield batches_list, buffer_metrics


def save_replay_buffer(path, state):
    """Overwrite one replay-buffer sidecar instead of embedding it in checkpoints."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, path)


def load_checkpoint(config, load_path, optimizer):
    """Loads and unpacks structural training configurations from disk.
    
    Handles comprehensive run dictionaries and raw fallback weights.
    Returns:
        tuple: (model, opt_state, iteration, frames, hours, rng_key)
    """
    print(f"Opening checkpoint source at {load_path}...")
    with open(load_path, "rb") as f:
        checkpoint_data = pickle.load(f)

    # Initialize standard default loop baselines
    iteration = 0
    frames = 0
    hours = 0.0
    rng_key = jax.random.PRNGKey(config.seed)
    buffer_state = None

    # Case A: Rich Dictionary Configuration
    if isinstance(checkpoint_data, dict) and "model" in checkpoint_data:
        print("Comprehensive snapshot structure detected. Extracting track metrics...")
        model = checkpoint_data["model"]
        
        if "opt_state" in checkpoint_data:
            opt_state = checkpoint_data["opt_state"]
        else:
            # Reconstruct empty optimizer weights if missing from dictionary layout
            opt_state = optimizer.init(params=model[0])
            
        iteration = checkpoint_data.get("iteration", 0)
        frames = checkpoint_data.get("frames", 0)
        hours = checkpoint_data.get("hours", 0.0)
        buffer_state = checkpoint_data.get("replay_buffer_state", None)
        
        if "rng_key" in checkpoint_data:
            rng_key = checkpoint_data["rng_key"]

    # Case B: Raw Parameter Fallback (e.g. sl_weights.pkl)
    else:
        print("Raw parameter snapshot detected. Re-initializing optimization trackers...")
        model = checkpoint_data
        opt_state = optimizer.init(params=model[0])

    return model, opt_state, iteration, frames, hours, rng_key, buffer_state


def save_checkpoint(ckpt_dir, filename, dic):
    """Safely exports runtime snapshot dictionaries to disk with explicit tracking metadata."""
    ckpt_path = os.path.join(ckpt_dir, filename)
    with open(ckpt_path, "wb") as f:
        pickle.dump(dic, f)


if __name__ == "__main__":
    # Initialize connection logging
    wandb.init(project="pgx-chess-muzero", config=config.model_dump())

    # Build unique run directory
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y%m%d%H%M%S")
    ckpt_dir = os.path.join("checkpoints", f"{config.env_id}_{now}")
    os.makedirs(ckpt_dir, exist_ok=True)
    replay_buffer_path = config.replay_buffer_path
    if replay_buffer_path is None:
        source_dir = os.path.dirname(config.load_ckpt) if config.load_ckpt else ckpt_dir
        replay_buffer_path = os.path.abspath(os.path.join(source_dir, "replay_buffer.pkl"))

    # Initialize standard network structures
    init_key = jax.random.PRNGKey(config.seed)
    dummy_state = jax.vmap(env.init)(jax.random.split(init_key, 2))
    model = forward.init(init_key, dummy_state.observation)
    opt_state = optimizer.init(params=model[0])

    # Instantiate default loop state trackers
    iteration, frames, hours = 0, 0, 0.0
    rng_key = jax.random.PRNGKey(config.seed)
    buffer_state = None

    # Automatically resume from intermediate states if path parameters are provided
    if config.load_ckpt is not None and os.path.exists(config.load_ckpt):
        model, opt_state, iteration, frames, hours, rng_key, buffer_state = load_checkpoint(
            config=config, load_path=config.load_ckpt, optimizer=optimizer
        )
        print(f"Resuming pipeline from iteration tracker index: {iteration}")

    if config.training_mode == "pipeline":
        evaluation_key = rng_key
        report_random_evaluation(
            evaluation_key, model, config.supervised_eval_games, "supervised_eval/before"
        )

    # ROUTING LOGIC: Hand over code processing to the isolated function nodes
    if config.training_mode in ("sl", "pipeline"):
        model, opt_state = run_supervised_training(
            config, model, opt_state, num_devices, sharding, ckpt_dir
        )

    if config.training_mode == "pipeline":
        random_stats, rng_key = report_random_evaluation(
            evaluation_key, model, config.supervised_eval_games, "supervised_eval/after"
        )
        if config.require_random_win and random_stats["score"] < config.supervised_min_random_score:
            print(
                "Random-opponent gate failed; RL is paused. "
                f"Required score: {config.supervised_min_random_score:.2%}."
            )
            raise SystemExit(0)
        opt_state = optimizer.init(params=model[0])

    if config.training_mode in ("rl", "pipeline"):
        run_rl_training(
            config, model, opt_state, num_devices, sharding, ckpt_dir,
            iteration, frames, hours, rng_key, buffer_state, replay_buffer_path
        )
