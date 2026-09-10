import os
# Force JAX to use the CPU for this script to prevent GPU OOM warnings
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import urllib.request
import pickle
import pandas as pd
import chess
import numpy as np
import pgx
import jax

def download_and_preprocess_xlsx(
    url: str = "https://mendeley.com",
    xlsx_path: str = "data/chess_dataset.xlsx",
    output_path: str = "checkpoints/sl_dataset.pkl",
    max_positions: int = 100000,
    min_elo: int = 2000
):
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    if not os.path.exists(xlsx_path) and not os.path.exists(output_path):
        print(" Downloading dataset file...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, xlsx_path)

    if os.path.exists(output_path):
        print("Dataset already compiled.")
        return

    print(f"Loading spreadsheet database: {xlsx_path}")
    target_cols = ['WhiteElo', 'BlackElo', 'Termination', 'Moves']
    df = pd.read_excel(xlsx_path, usecols=target_cols)
    
    df['WhiteElo'] = pd.to_numeric(df['WhiteElo'], errors='coerce')
    df['BlackElo'] = pd.to_numeric(df['BlackElo'], errors='coerce')
    
    mask = (df['WhiteElo'] >= min_elo) & (df['BlackElo'] >= min_elo)
    if 'Termination' in df.columns:
        mask &= (df['Termination'].str.lower() != 'abandoned')
        
    df_filtered = df[mask].dropna(subset=['Moves'])
    print(f" Ready to process {len(df_filtered)} expert games.")

    env = pgx.make("chess")
    obs_list, act_list = [], []
    total_positions = 0
    
    print(" Extracting board positions...")
    for idx, row in df_filtered.iterrows():
        if total_positions >= max_positions:
            break
            
        moves_list = str(row['Moves']).split()
        py_board = chess.Board()
        state = env.init(jax.random.PRNGKey(0))
        
        for move_str in moves_list:
            if total_positions >= max_positions:
                break
                
            # FIX: Skip move numbers (e.g., "1.", "2...", "12") instead of breaking
            if '.' in move_str or move_str.isdigit() or move_str == '*':
                continue
                
            try:
                try:
                    move = py_board.parse_san(move_str)
                except ValueError:
                    move = py_board.parse_uci(move_str)
                
                action_idx = env.action_names.index(move.uci())
            except Exception:
                # If a specific move fails to parse, skip the remainder of THIS game
                break
                
            if not state.legal_action_mask[action_idx]:
                break
                
            obs_list.append(np.array(state.observation))
            act_list.append(action_idx)
            total_positions += 1
            
            py_board.push(move)
            state = env.step(state, jax.shape_as_value(action_idx))
            if state.terminated:
                break
                
        if total_positions % 10000 == 0 and total_positions > 0:
            print(f" Progress: Gathered {total_positions}/{max_positions} positions...")

    if total_positions == 0:
        print("No positions found matching filter criteria.")
        return

    dataset = {
        "observations": np.stack(obs_list, axis=0).astype(np.float32),
        "actions": np.array(act_list, dtype=np.int32)
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
        
    print(f" Complete! File saved to {output_path} (Shape: {dataset['observations'].shape})")

if __name__ == "__main__":
    download_and_preprocess_xlsx()
