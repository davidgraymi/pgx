import os
import urllib.request
import pickle
import pandas as pd
import chess
import numpy as np
import pgx
import jax

def download_and_preprocess_xlsx(
    url: str = "https://data.mendeley.com/public-files/datasets/m2wxzmhv6s/files/cf304182-cdb2-4004-8e6a-ea7335019539/file_downloaded",
    xlsx_path: str = "data/chess_dataset.xlsx",
    output_path: str = "data/sl_dataset.pkl",
    max_positions: int = 100000,
    min_elo: int = 2000
):
    """Downloads the spreadsheet and compiles board tensors from high-Elo master games."""
    os.makedirs("data", exist_ok=True)
    
    # -------------------------------------------------------------------------
    # STEP 1: DATASET DOWNLOAD
    # -------------------------------------------------------------------------
    if not os.path.exists(xlsx_path) and not os.path.exists(output_path):
        print(f"Downloading Excel dataset from Mendeley: {url}")
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, xlsx_path)
            print("Download complete.")
        except Exception as e:
            print(f"Network error: {e}. Please download file manually to '{xlsx_path}'")
            return

    # -------------------------------------------------------------------------
    # STEP 2: QUALITY FILTERING AND TENSOR EXTRACTION
    # -------------------------------------------------------------------------
    if os.path.exists(output_path):
        print(f"Processed target dataset already exists at {output_path}. Skipping.")
        return

    print(f"Loading spreadsheet database: {xlsx_path}")
    target_cols = ['WhiteElo', 'BlackElo', 'Termination', 'Moves']
    df = pd.read_excel(xlsx_path, usecols=target_cols)
    
    # Clean and cast rating data columns safely to numeric types
    df['WhiteElo'] = pd.to_numeric(df['WhiteElo'], errors='coerce')
    df['BlackElo'] = pd.to_numeric(df['BlackElo'], errors='coerce')
    
    # Filter for high-quality games: Both players 2000+ Elo AND normal game finishes (no abandons)
    quality_mask = (df['WhiteElo'] >= min_elo) & (df['BlackElo'] >= min_elo)
    if 'Termination' in df.columns:
        quality_mask &= (df['Termination'].str.lower() != 'abandoned')

    df_filtered = df[quality_mask].dropna(subset=['Moves'])
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
                
            try:
                try:
                    move = py_board.parse_san(move_str)
                except ValueError:
                    move = py_board.parse_uci(move_str)
                
                action_idx = env.action_names.index(move.uci())
            except Exception:
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
                
        # Status update every 10,000 positions
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
