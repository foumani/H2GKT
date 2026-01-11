import pandas as pd
import numpy as np
import argparse
import os
import tqdm


def create_splits(input_path, output_path, dataset_name='assist2009', fractions=(0.6, 0.2, 0.1, 0.1)):
    """
    Reads a CSV, sorts it chronologically per user, and assigns a split label.

    The labels correspond to the exact indices used in HybridDataManager:
    - 'train_gnn': 0% -> 60%
    - 'train_rnn': 60% -> 80%
    - 'val':       80% -> 90%
    - 'test':      90% -> 100%
    """
    print(f"Reading data from {input_path}...")
    df = pd.read_csv(input_path, encoding='ISO-8859-1', low_memory=False)

    if dataset_name == "assist2012":
        print("[Info] Applying ASSIST2012 specific preprocessing...")

        # Rename 'skill' -> 'skill_name' if needed
        if 'skill' in df.columns and 'skill_name' not in df.columns:
            print("Renaming 'skill' column to 'skill_name'...")
            df.rename(columns={'skill': 'skill_name'}, inplace=True)

        # Ensure Numeric Types
        # We use 'errors=coerce' to turn bad strings into NaN, then drop or fill if necessary (usually casting works fine)
        # Note: We cast to float first to handle potential "1.0" strings, then to Int.
        cols_to_cast = ['user_id', 'problem_id', 'correct']
        if 'skill_id' in df.columns:
            cols_to_cast.append('skill_id')

        for col in cols_to_cast:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int)

        # Generate 'order_id' from 'start_time' if missing
        if 'order_id' not in df.columns:
            if 'start_time' in df.columns:
                print("Generating 'order_id' from 'start_time'...")
                df['order_id'] = pd.to_datetime(df['start_time'], errors='coerce',
                                                format='mixed').astype(np.int64)
            else:
                print("Warning: No 'start_time' found. Using DataFrame index as 'order_id'.")
                df['order_id'] = df.index

    # Drop ros without skill_name
    if 'skill_name' in df.columns:
        before = len(df)
        df.dropna(subset=['skill_name'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        if len(df) < before:
            print(
                f"Dropped {before - len(df)} rows where 'skill_name' was NaN.")
    # Fallback: If no skill_name, ensure we at least have valid skill_ids
    elif 'skill_id' in df.columns:
        df.dropna(subset=['skill_id'], inplace=True)
        df.reset_index(drop=True, inplace=True)

    # 1. Sort Data Chronologically (Crucial!)
    print("Sorting data by user_id and order_id...")
    df = df.sort_values(by=['user_id', 'order_id']).reset_index(drop=True)

    # Initialize split column with default value 'test' (captures the last 10%)
    split_labels = np.array(['test'] * len(df), dtype=object)

    # 2. Calculate Split Fractions
    gnn_frac, rnn_frac, val_frac, test_frac = fractions

    print("Calculating splits per user...")
    grouped = df.groupby('user_id')

    # Lists to collect indices for bulk update
    gnn_indices = []
    rnn_indices = []
    val_indices = []

    for user_id, group in tqdm.tqdm(grouped):
        indices = group.index.values  # Global indices
        n = len(group)

        # Exact Integer Arithmetic matching HybridDataManager
        n_gnn = int(n * gnn_frac)
        n_rnn = int(n * rnn_frac)
        n_val = int(n * val_frac)

        # Calculate Cutoff Points
        # 1. GNN Slice (0 -> 60%)
        if n_gnn > 0:
            gnn_indices.append(indices[:n_gnn])

        # 2. RNN Slice (60% -> 80%)
        # Starts where GNN ended, length is n_rnn
        rnn_start = n_gnn
        rnn_end = n_gnn + n_rnn
        if n_rnn > 0:
            rnn_indices.append(indices[rnn_start:rnn_end])

        # 3. Validation Slice (80% -> 90%)
        # Starts where RNN ended
        val_start = rnn_end
        val_end = rnn_end + n_val
        if n_val > 0:
            val_indices.append(indices[val_start:val_end])

        # Remainder (90% -> 100%) is left as 'test' by default

    # 3. Apply Labels
    print("Applying split labels...")
    if gnn_indices:
        split_labels[np.concatenate(gnn_indices)] = 'train_gnn'

    if rnn_indices:
        split_labels[np.concatenate(rnn_indices)] = 'train_rnn'

    if val_indices:
        split_labels[np.concatenate(val_indices)] = 'val'

    # Assign column
    df['split'] = split_labels

    # 4. Save
    print(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False)

    # Stats
    print("\nFinal Split Statistics:")
    print(df['split'].value_counts())
    print(f"Total Rows: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split data into train_gnn, train_rnn, val, test.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input CSV file.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save output CSV file.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Name of dataset (e.g. assist2009, assist2012)")

    args = parser.parse_args()

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    create_splits(args.input, args.output, args.dataset)