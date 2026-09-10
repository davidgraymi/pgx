import os
# Force JAX to use the CPU for data preprocessing to avoid VRAM allocation errors
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
    min_elo: int = 1600  # Lowered to 1600 to cleanly catch games like your sample row
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
    
    # Filter: At least one player is an advanced 1600+ user, and eliminate toxic/abandoned records
    mask = (df['WhiteElo'] >= min_elo) | (df['BlackElo'] >= min_elo)
    if 'Termination' in df.columns:
        mask &= (df['Termination'].str.lower().str.contains("abandoned") == False)
        
    df_filtered = df[mask].dropna(subset=['Moves'])
    print(f" Ready to process {len(df_filtered)} qualified high-level games.")

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
                
            # Skip invalid characters or headers if they leak into string
            if move_str in ['*', '1-0', '0-1', '1/2-1/2'] or len(move_str) < 4:
                continue
                
            try:
                # Optimized directly for your dataset's pure UCI format (e.g., 'd2d4')
                move = py_board.parse_uci(move_str)
                action_idx = env.action_names.index(move.uci())
            except Exception:
                # If a move fails to parse, skip the remainder of this specific game
                break
                
            # Skip if the move is flagged as illegal by Pgx internal logic
            if not state.legal_action_mask[action_idx]:
                break
                
            # Keep the features and target move index
            obs_list.append(np.array(state.observation))
            act_list.append(action_idx)
            total_positions += 1
            
            py_board.push(move)
            state = env.step(state, jax.shape_as_value(action_idx))
            if state.terminated:
                break
                
        if total_positions % 1 == 0 and total_positions > 0:
            print(f" Progress: Gathered {total_positions}/{max_positions} positions...")
        else:
            print("skip")

    if total_positions == 0:
        print("No positions found matching filter criteria.")
        return

    # Pack into array representations
    dataset = {
        "observations": np.stack(obs_list, axis=0).astype(np.float32),
        "actions": np.array(act_list, dtype=np.int32)
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
        
    print(f" Complete! File saved to {output_path} (Shape: {dataset['observations'].shape})")

if __name__ == "__main__":
    download_and_preprocess_xlsx()
