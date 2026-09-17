# MuZero example

A simple (Gumbel) AlphaZero [[Silver+18](https://www.science.org/doi/10.1126/science.aar6404), [Danihelka+22](https://openreview.net/forum?id=bERaNdoegnO)] example using [Mctx](https://github.com/deepmind/mctx) library. See [Pgx paper](https://openreview.net/forum?id=UvX8QfhfUx) for more details.

![](assets/pgx-az-training.png)

> [!NOTE]
> This implementation of AlphaZero demonstrates sufficient learning performance in environments including 9x9 Go, but it has some slight differences in learning details compared to the original AlphaZero and Gumbel AlphaZero. An implementation that addresses these differences and focuses on enhanced efficiency is currently under development and is expected to be released shortly.

## Usage

Note that you need to install `jax` and `jaxlib` in addition to the packages written in `requirements.txt` according to your execution environment.

```sh
$ pip install -U pip && pip install -r requirements.txt
$ python3 data.py
$ python3 train.py env_id=chess seed=0
```

The data-preparation step scans three monthly elite databases and retains a
stratified random sample of up to 250,000 positions: 25% opening, 50%
middlegame, and 25% endgame, with a per-opening cap. It stores observations as
compressed `float16` data and removes the extracted PGNs after preprocessing.
Ten percent of complete games are reserved for `data/sl_validation.npz`; no
positions from those games enter training. Supervised training reports held-out
loss and top-1/top-5 accuracy after every epoch.
If an older compiled dataset contains fewer positions, `data.py` rebuilds it.
While scanning, it overwrites `data/sl_dataset.npz` every 1,000,000 positions
and after each month. Each save reports retained games, month coverage, action
coverage, and action entropy, so an interrupted run still leaves a usable dataset.
The default chess run uses a smaller network, eight self-play games per batch,
64 search simulations, and a 50,000-position host replay buffer for a 6 GB GPU.
`training_mode=pipeline` first trains on `data/sl_dataset.npz`, plays 512 balanced
games against a uniform random player, and starts RL only when the score reaches
`supervised_min_random_score`. Set `require_random_win=false` to bypass that gate.
The pipeline reports the same random-opponent evaluation before and after
supervised training for a direct comparison.
Evaluation uses deterministic MCTS with the configured simulation count; only the
random opponent samples moves.
RL evaluations now use deterministic MCTS with the configured simulation count,
including uniform opponent priors during random-opponent search. They save the
best random-opponent model as `best_random.ckpt`, report score confidence bounds,
and stop after `rl_regression_patience` evaluations without improvement.
Regular evaluations use `eval_games` with an `eval_max_steps` cap. Promotion
iterations run only the larger `promotion_eval_games` evaluation, rather than
running both evaluation sizes. Regression stopping also writes
`final.ckpt` from the best checkpoint.
Supervised training runs up to `supervised_epochs`, stopping when held-out loss
fails to improve for `supervised_validation_patience` epochs and restoring the
best validation-loss model. Training entropy and accuracy are logged only.
The supervised policy uses 5% label smoothing, AdamW weight decay, and game
result targets for the value head. Validation metrics are broken down by phase
and color, while random-opponent evaluations report White and Black scores.

Checkpoints contain model and optimizer state only. The replay buffer is stored
once as `replay_buffer.pkl` beside the checkpoints and overwritten in place;
override it with `replay_buffer_path=...` when resuming a run.
RL uses a 50,000-position ring buffer, samples it without copying the whole
buffer, and bootstraps value targets for rollouts that reach the step limit.
W&B also records separate self-play, training, and whole-loop FPS, termination
and truncation rates, value-target statistics, update counts, and replay-sample age.
RL batches are prefetched to devices, and `replay_update_ratio` controls how many
updates are made per newly collected batch.

Useful resource controls:

```sh
$ python3 train.py training_mode=pipeline selfplay_batch_size=4 num_simulations=32
$ python3 train.py training_mode=pipeline eval_games=128 supervised_eval_games=128
```

## Reference

- [[Silver+18](https://www.science.org/doi/10.1126/science.aar6404)] "A general reinforcement learning algorithm that masters
chess, shogi, and go through self-play"
- [[Danihelka+22](https://openreview.net/forum?id=bERaNdoegnO)] "Policy improvement by planning with Gumbel"


## Change history

- **[#1107](https://github.com/sotetsuk/pgx/pull/1107)** Extract `compute_loss_input` ([wandb report](https://api.wandb.ai/links/sotetsuk/979hmps8)).
- **[#1106](https://github.com/sotetsuk/pgx/pull/1106)** Use `optax.softmax_cross_entropy` ([wandb report](https://api.wandb.ai/links/sotetsuk/8w0or84k)).
- **[#1088](https://github.com/sotetsuk/pgx/pull/1088)** Adjust to API v2 ([wandb report](https://api.wandb.ai/links/sotetsuk/0g44pjsg)).
- **[#1055](https://github.com/sotetsuk/pgx/pull/1055)** Use default Gumbel AlphaZero hyperparameters ([wandb report](https://api.wandb.ai/links/sotetsuk/o8752t54)).
- **[#1026](https://github.com/sotetsuk/pgx/pull/1026)** Initial version. Supposed to reproduce the [Pgx paper](https://openreview.net/forum?id=UvX8QfhfUx) results ([wandb report](https://api.wandb.ai/links/sotetsuk/5q30e5n9)).
