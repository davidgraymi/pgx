import os
# Force JAX to use the CPU for data preprocessing to avoid VRAM allocation errors
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import urllib.request
import zipfile
import pickle
import chess
import chess.pgn
import numpy as np
import pgx
import jax
from pgx._src.games.chess import TO_PLANE

# Lichess Elite Database: monthly dumps of games between 2400+ (2500+ from
# 2021-12 onward) vs 2200+ (2300+) rated players. https://database.nikonoel.fr/
ELITE_DB_URL = "https://database.nikonoel.fr/lichess_elite_{month}.zip"

# pgx's chess action space is the AlphaZero-style 64x73 encoding (source
# square x move-plane), not a UCI string, and its board is always represented
# from the *current mover's* point of view (flipped every ply). These two
# constants and the two helpers below convert a python-chess move into that
# encoding. See pgx/_src/games/chess.py for the reference encoding.
QUEENSIDE_CASTLE_ACTION = 2364
KINGSIDE_CASTLE_ACTION = 2367
UNDERPROMOTION_PIECE_PLANE = {chess.ROOK: 0, chess.BISHOP: 1, chess.KNIGHT: 2}
UNDERPROMOTION_DIRECTION = {1: 0, 9: 1, -7: 2}  # pov_to - pov_from -> plane offset


def _square_to_pov(square: int, white_to_move: bool) -> int:
    file, rank = chess.square_file(square), chess.square_rank(square)
    pov = file * 8 + rank
    if not white_to_move:
        pov = (pov // 8) * 8 + (7 - pov % 8)  # mirror rank: board is flipped on Black's turn
    return pov


def move_to_action(board: chess.Board, move: chess.Move) -> int:
    """Convert a legal python-chess move (given the board it's played on) into
    pgx's action index for `pgx.make("chess")`."""
    if board.is_kingside_castling(move):
        return KINGSIDE_CASTLE_ACTION
    if board.is_queenside_castling(move):
        return QUEENSIDE_CASTLE_ACTION

    white_to_move = board.turn == chess.WHITE
    pov_from = _square_to_pov(move.from_square, white_to_move)
    pov_to = _square_to_pov(move.to_square, white_to_move)

    if move.promotion is not None and move.promotion != chess.QUEEN:
        piece_plane = UNDERPROMOTION_PIECE_PLANE[move.promotion]
        direction = UNDERPROMOTION_DIRECTION[pov_to - pov_from]
        return pov_from * 73 + piece_plane * 3 + direction

    plane = int(TO_PLANE[pov_from, pov_to])
    return pov_from * 73 + plane


def download_month(month: str, data_dir: str) -> str:
    """Download and extract one month of the Lichess Elite Database. Returns
    the path to the extracted .pgn file."""
    pgn_path = os.path.join(data_dir, f"lichess_elite_{month}.pgn")
    if os.path.exists(pgn_path):
        return pgn_path

    zip_path = os.path.join(data_dir, f"lichess_elite_{month}.zip")
    if not os.path.exists(zip_path):
        url = ELITE_DB_URL.format(month=month)
        print(f"Downloading {url} ...")
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, zip_path)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".pgn")]
        if not names:
            raise RuntimeError(f"No .pgn file found inside {zip_path}")
        with zf.open(names[0]) as src, open(pgn_path, "wb") as dst:
            dst.write(src.read())
    os.remove(zip_path)
    return pgn_path


def iter_training_games(pgn_path: str, min_elo: int):
    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                return
            headers = game.headers
            try:
                white_elo = int(headers.get("WhiteElo", 0))
                black_elo = int(headers.get("BlackElo", 0))
            except ValueError:
                continue
            if white_elo < min_elo or black_elo < min_elo:
                continue
            if headers.get("Termination", "").lower() == "abandoned":
                continue
            yield game


class StratifiedReservoir:
    phases = ("opening", "middlegame", "endgame")

    def __init__(self, max_positions: int, rng: np.random.Generator):
        opening_capacity = max(1, max_positions // 4)
        middlegame_capacity = max(1, max_positions // 2)
        endgame_capacity = max(1, max_positions - opening_capacity - middlegame_capacity)
        self.phase_capacities = {
            "opening": opening_capacity,
            "middlegame": middlegame_capacity,
            "endgame": endgame_capacity,
        }
        self.opening_buckets = 50
        self.rng = rng
        self.buckets = {}
        self.seen = {}
        self.phase_counts = {phase: 0 for phase in self.phases}

    def _remove_random(self, phase, excluded_key):
        candidates = [
            key
            for key, bucket in self.buckets.items()
            if key[0] == phase and key != excluded_key and bucket["obs"]
        ]
        if not candidates:
            candidates = [
                key for key, bucket in self.buckets.items() if key[0] == phase and bucket["obs"]
            ]
        largest_size = max(len(self.buckets[key]["obs"]) for key in candidates)
        candidates = [key for key in candidates if len(self.buckets[key]["obs"]) == largest_size]
        key = candidates[int(self.rng.integers(len(candidates)))]
        bucket = self.buckets[key]
        index = int(self.rng.integers(len(bucket["obs"])))
        for values in bucket.values():
            values.pop(index)
        self.phase_counts[phase] -= 1

    def add(self, observation, action, month, game_id, phase, opening, value, player):
        key = (phase, opening)
        bucket = self.buckets.setdefault(key, {"obs": [], "actions": [], "months": [], "games": [], "phases": [], "values": [], "players": []})
        seen = self.seen.get(key, 0) + 1
        self.seen[key] = seen
        capacity = max(1, self.phase_capacities[phase] // self.opening_buckets)
        if len(bucket["obs"]) >= capacity:
            index = self.rng.integers(seen)
            if index >= capacity:
                return
            values = (observation, action, month, game_id, phase, value, player)
            bucket["obs"][index], bucket["actions"][index], bucket["months"][index], bucket["games"][index], bucket["phases"][index], bucket["values"][index], bucket["players"][index] = values
            return

        values = (observation, action, month, game_id, phase, value, player)
        for name, item in zip(("obs", "actions", "months", "games", "phases", "values", "players"), values):
            bucket[name].append(item)
        self.phase_counts[phase] += 1
        if self.phase_counts[phase] > self.phase_capacities[phase]:
            self._remove_random(phase, key)

    def flatten(self):
        observations, actions, months, games, phases, openings, values, players = [], [], [], [], [], [], [], []
        for phase in self.phases:
            for (bucket_phase, opening), bucket in sorted(self.buckets.items()):
                if bucket_phase != phase:
                    continue
                observations.extend(bucket["obs"])
                actions.extend(bucket["actions"])
                months.extend(bucket["months"])
                games.extend(bucket["games"])
                phases.extend(bucket["phases"])
                values.extend(bucket["values"])
                players.extend(bucket["players"])
                openings.extend([opening] * len(bucket["obs"]))
        return observations, actions, months, games, phases, openings, values, players

    def __len__(self):
        return sum(len(bucket["obs"]) for bucket in self.buckets.values())


def save_dataset_checkpoint(
    output_path: str,
    observations,
    actions,
    sample_months,
    sample_game_ids,
    sample_phases,
    sample_openings,
    sample_values,
    sample_players,
    positions_seen: int,
    games_seen: int,
    reason: str,
    split: str,
):
    observations = np.stack(observations, axis=0).astype(np.float16)
    actions = np.asarray(actions, dtype=np.uint16)
    action_counts = np.bincount(actions, minlength=4672)
    action_probs = action_counts[action_counts > 0] / actions.size
    action_entropy = -np.sum(action_probs * np.log2(action_probs))
    action_entropy_pct = action_entropy / np.log2(4672) * 100.0
    unique_games = len(set(sample_game_ids))
    unique_months = len(set(sample_months))
    unique_openings = len(set(sample_openings))
    unique_actions = np.count_nonzero(action_counts)
    phase_counts = {phase: sample_phases.count(phase) for phase in ("opening", "middlegame", "endgame")}

    metadata = {
        "format_version": 6,
        "split": split,
        "positions_seen": positions_seen,
        "games_seen": games_seen,
        "sample_months": np.asarray(sample_months),
        "sample_game_ids": np.asarray(sample_game_ids, dtype=np.int64),
        "sample_phases": np.asarray(sample_phases),
        "sample_openings": np.asarray(sample_openings),
        "values": np.asarray(sample_values, dtype=np.float32),
        "players": np.asarray(sample_players, dtype=np.int8),
    }
    temporary_path = output_path + ".tmp"
    if output_path.endswith(".npz"):
        with open(temporary_path, "wb") as f:
            np.savez_compressed(
                f,
                observations=observations,
                actions=actions,
                **metadata,
            )
    else:
        with open(temporary_path, "wb") as f:
            pickle.dump({"observations": observations, "actions": actions, **metadata}, f)
    os.replace(temporary_path, output_path)

    print(
        f"[Dataset Save: {split} / {reason}] {output_path} | "
        f"{len(actions)} retained / {positions_seen} seen | "
        f"{unique_games} games, {unique_openings} openings across {unique_months} months | "
        f"phases O/M/E {phase_counts['opening']}/{phase_counts['middlegame']}/{phase_counts['endgame']} | "
        f"{unique_actions}/4672 actions | "
        f"action entropy {action_entropy_pct:.1f}%"
    )


def _dataset_is_current(path: str, min_positions: int) -> bool:
    if not os.path.exists(path):
        return False
    try:
        if path.endswith(".npz"):
            with np.load(path) as dataset:
                positions = dataset["observations"].shape[0]
                version = int(dataset["format_version"]) if "format_version" in dataset.files else 0
        else:
            with open(path, "rb") as f:
                dataset = pickle.load(f)
            positions = dataset["observations"].shape[0]
            version = dataset.get("format_version", 0)
        return positions >= min_positions and version >= 6
    except (OSError, KeyError, ValueError, pickle.PickleError):
        return False


def download_and_preprocess(
    months=("2024-10", "2024-11", "2024-12"),
    data_dir: str = "data",
    output_path: str = "data/sl_dataset.npz",
    max_positions: int = 250000,
    min_elo: int = 2200,
    cleanup_source: bool = True,
    sample_seed: int = 0,
    save_interval_positions: int = 1_000_000,
    validation_fraction: float = 0.1,
    validation_output_path: str = "data/sl_validation.npz",
    validation_max_positions: int | None = None,
):
    if save_interval_positions <= 0:
        raise ValueError("save_interval_positions must be positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if validation_max_positions is None:
        validation_max_positions = max_positions // 5
    if validation_max_positions <= 0:
        raise ValueError("validation_max_positions must be positive")
    if os.path.abspath(output_path) == os.path.abspath(validation_output_path):
        raise ValueError("Training and validation outputs must be different files")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(validation_output_path) or ".", exist_ok=True)

    if _dataset_is_current(output_path, max_positions) and _dataset_is_current(
        validation_output_path, validation_max_positions
    ):
        print("Training and validation datasets already compiled.")
        return

    env = pgx.make("chess")
    env_init = jax.jit(env.init)
    env_step = jax.jit(env.step)
    key = jax.random.PRNGKey(0)
    sample_rng = np.random.default_rng(sample_seed)
    split_rng = np.random.default_rng(sample_seed + 1)

    games_seen = games_used = positions_seen = 0
    train_positions_seen = validation_positions_seen = 0
    last_saved_positions = 0
    train_reservoir = StratifiedReservoir(max_positions, sample_rng)
    validation_reservoir = StratifiedReservoir(validation_max_positions, np.random.default_rng(sample_seed + 2))

    def save_reservoir(reservoir, path, positions, reason, split):
        if len(reservoir) == 0:
            return
        save_dataset_checkpoint(
            path,
            *reservoir.flatten(),
            positions,
            games_seen,
            reason,
            split,
        )

    def save_datasets(reason):
        save_reservoir(train_reservoir, output_path, train_positions_seen, reason, "train")
        save_reservoir(
            validation_reservoir,
            validation_output_path,
            validation_positions_seen,
            reason,
            "validation",
        )

    for month in months:
        pgn_path = download_month(month, data_dir)
        print(f"Parsing {pgn_path} ...")

        for game in iter_training_games(pgn_path, min_elo):
            games_seen += 1
            board = game.board()
            state = env_init(key)
            added = 0
            validation_game = split_rng.random() < validation_fraction

            opening = str(game.headers.get("ECO", "")).strip()
            if not opening or opening in {"?", "-"}:
                opening = f"unknown_{games_seen % train_reservoir.opening_buckets:02d}"
            result = game.headers.get("Result", "*")
            white_value = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)

            for ply, move in enumerate(game.mainline_moves()):
                try:
                    action = move_to_action(board, move)
                except KeyError:
                    break  # unexpected move shape; stop trusting the rest of this game
                if not (0 <= action < 4672) or not bool(state.legal_action_mask[action]):
                    break  # conversion/board desync; stop trusting the rest of this game

                observation = np.array(state.observation)
                positions_seen += 1
                if validation_game:
                    validation_positions_seen += 1
                    target_reservoir = validation_reservoir
                else:
                    train_positions_seen += 1
                    target_reservoir = train_reservoir
                non_pawn_pieces = sum(
                    len(board.pieces(piece, color))
                    for piece in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
                    for color in (chess.WHITE, chess.BLACK)
                )
                if ply < 20:
                    phase = "opening"
                elif non_pawn_pieces <= 6:
                    phase = "endgame"
                else:
                    phase = "middlegame"
                player = 0 if board.turn == chess.WHITE else 1
                value = white_value if player == 0 else -white_value
                target_reservoir.add(
                    observation, action, month, games_seen, phase, opening, value, player
                )
                added += 1

                if positions_seen - last_saved_positions >= save_interval_positions:
                    save_datasets(f"{positions_seen} positions seen")
                    last_saved_positions = positions_seen

                state = env_step(state, action)
                board.push(move)
                if bool(state.terminated):
                    break

            games_used += added > 0
            if games_seen % 200 == 0:
                print(
                    f"  ...{games_seen} games scanned, {games_used} used, "
                    f"{positions_seen} positions seen, {len(train_reservoir)} train / "
                    f"{len(validation_reservoir)} validation retained"
                )

        if positions_seen > last_saved_positions and len(train_reservoir):
            save_datasets(f"completed {month}")
            last_saved_positions = positions_seen

    if not len(train_reservoir) or not len(validation_reservoir):
        print("No positions found matching filter criteria.")
        return

    save_datasets("final")

    if cleanup_source:
        for month in months:
            source_path = os.path.join(data_dir, f"lichess_elite_{month}.pgn")
            if os.path.exists(source_path):
                os.remove(source_path)

    print(
        f"Complete! {len(train_reservoir)} train and {len(validation_reservoir)} validation "
        f"positions from {games_used}/{games_seen} games saved."
    )


if __name__ == "__main__":
    download_and_preprocess()
