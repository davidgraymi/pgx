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


def download_and_preprocess(
    months=("2024-10",),
    data_dir: str = "data",
    output_path: str = "checkpoints/sl_dataset.pkl",
    max_positions: int = 100000,
    min_elo: int = 2200,
):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print("Dataset already compiled.")
        return

    env = pgx.make("chess")
    env_init = jax.jit(env.init)
    env_step = jax.jit(env.step)
    key = jax.random.PRNGKey(0)

    obs_list, act_list = [], []
    games_seen = games_used = 0

    for month in months:
        if len(obs_list) >= max_positions:
            break

        pgn_path = download_month(month, data_dir)
        print(f"Parsing {pgn_path} ...")

        for game in iter_training_games(pgn_path, min_elo):
            if len(obs_list) >= max_positions:
                break

            games_seen += 1
            board = game.board()
            state = env_init(key)
            added = 0

            for move in game.mainline_moves():
                if len(obs_list) >= max_positions:
                    break

                try:
                    action = move_to_action(board, move)
                except KeyError:
                    break  # unexpected move shape; stop trusting the rest of this game
                if not (0 <= action < 4672) or not bool(state.legal_action_mask[action]):
                    break  # conversion/board desync; stop trusting the rest of this game

                obs_list.append(np.array(state.observation))
                act_list.append(action)
                added += 1

                state = env_step(state, action)
                board.push(move)
                if bool(state.terminated):
                    break

            games_used += added > 0
            if games_seen % 200 == 0:
                print(f"  ...{games_seen} games scanned, {games_used} used, {len(obs_list)} positions collected")

    if not obs_list:
        print("No positions found matching filter criteria.")
        return

    dataset = {
        "observations": np.stack(obs_list, axis=0).astype(np.float32),
        "actions": np.array(act_list, dtype=np.int32),
    }
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(
        f"Complete! {len(act_list)} positions from {games_used}/{games_seen} games saved to "
        f"{output_path} (shape: {dataset['observations'].shape})"
    )


if __name__ == "__main__":
    download_and_preprocess()
