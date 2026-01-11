import pandas as pd
import numpy as np
import argparse
import os
import tqdm
import json
import sys

# -------------------------------------------------------------------------
# 1. SETUP & IMPORTS
# -------------------------------------------------------------------------

# Add the parent directory (root) to sys.path to find create_splits.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from create_splits import create_splits
    print("Successfully imported 'create_splits' from root.")
except ImportError:
    print("Error: Could not import 'create_splits.py'. Make sure it is in the root folder.")
    sys.exit(1)

# -------------------------------------------------------------------------
# 2. AKT FORMATTING LOGIC
# -------------------------------------------------------------------------

def write_interaction(file_handle, student_id, interactions):
    """
    Writes a single student's history in the 4-line AKT format.
    """
    # 1. Extract Data
    problem_seq = interactions['problem_id'].astype(str).tolist()
    skill_seq = interactions['skill_id'].astype(str).tolist()

    # Handle float correctness (e.g. 1.0 -> 1)
    correct_seq = interactions['correct'].astype(float).astype(int).astype(str).tolist()

    seq_len = len(interactions)
    if seq_len == 0: return

    # 2. Construct the 4 Lines
    line1 = f"{seq_len},{student_id}\n"
    line2 = ",".join(problem_seq) + "\n"
    line3 = ",".join(skill_seq) + "\n"
    line4 = ",".join(correct_seq) + "\n"

    # 3. Write
    file_handle.write(line1)
    file_handle.write(line2)
    file_handle.write(line3)
    file_handle.write(line4)

def process_for_akt(split_df, output_dir, dataset_name):
    """
    Converts the dataframe with 'split' column into AKT train/val/test csvs.
    """
    print(f"Formatting for AKT (Transductive)...")

    # --- SAFETY SORT (Prevents Time Travel) ---
    if 'order_id' in split_df.columns:
        split_df = split_df.sort_values(by=['user_id', 'order_id'])
    else:
        split_df = split_df.sort_values(by=['user_id'])
    # ------------------------------------------

    # ---------------------------------------------------------
    # Standardize Column Names (just in case split_df missed it)
    # ---------------------------------------------------------

    # 1. Create Integer Mappings
    # FIX: We start mapping at 1, not 0.
    # This reserves '0' for the model's padding/masking mechanism.
    print("Creating ID Mappings (starting at 1 for sequence data)...")

    unique_students = sorted(split_df['user_id'].unique())
    unique_problems = sorted(split_df['problem_id'].unique())
    unique_skills = sorted(split_df['skill_id'].unique())

    # User IDs usually don't need padding logic (since they aren't a sequence),
    # but Problem and Skill IDs definitely do.
    student_map = {id: i for i, id in enumerate(unique_students)} # 0-based is fine for users
    problem_map = {id: i + 1 for i, id in enumerate(unique_problems)} # 1-based (0 is Pad)
    skill_map = {id: i + 1 for i, id in enumerate(unique_skills)}     # 1-based (0 is Pad)

    # Apply Mappings
    split_df['user_id_mapped'] = split_df['user_id'].map(student_map)
    split_df['problem_id_mapped'] = split_df['problem_id'].map(problem_map)
    split_df['skill_id_mapped'] = split_df['skill_id'].map(skill_map)

    # Save Mappings
    os.makedirs(output_dir, exist_ok=True)

    def save_json(data, filename):
        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump({str(k): v for k, v in data.items()}, f)

    save_json(student_map, f"{dataset_name}_student_map.json")
    save_json(problem_map, f"{dataset_name}_problem_map.json")
    save_json(skill_map, f"{dataset_name}_skill_map.json")

    # 2. Prepare Output Files
    train_path = os.path.join(output_dir, f"{dataset_name}_akt_train.csv")
    valid_path = os.path.join(output_dir, f"{dataset_name}_akt_valid.csv")
    test_path = os.path.join(output_dir, f"{dataset_name}_akt_test.csv")

    f_train = open(train_path, 'w')
    f_valid = open(valid_path, 'w')
    f_test = open(test_path, 'w')

    print("Writing AKT files...")

    grouped = split_df.groupby('user_id_mapped')

    for student_id, group in tqdm.tqdm(grouped):

        # Calculate Lengths based on 'split' column tags
        # Train (AKT) = train_gnn (60%) + train_rnn (20%) -> Total 80%
        mask_gnn = group['split'] == 'train_gnn'
        mask_rnn = group['split'] == 'train_rnn'
        mask_val = group['split'] == 'val'

        train_len = mask_gnn.sum() + mask_rnn.sum()
        val_len = train_len + mask_val.sum()

        # Prepare Data Slice
        data_to_write = pd.DataFrame({
            'problem_id': group['problem_id_mapped'],
            'skill_id': group['skill_id_mapped'],
            'correct': group['correct']
        })

        # WRITE SLICES
        # 1. Train File (0 -> 80%)
        if train_len > 0:
            write_interaction(f_train, student_id, data_to_write.iloc[:train_len])

        # 2. Valid File (0 -> 90%)
        if val_len > 0:
            write_interaction(f_valid, student_id, data_to_write.iloc[:val_len])

        # 3. Test File (0 -> 100%)
        write_interaction(f_test, student_id, data_to_write)

    f_train.close()
    f_valid.close()
    f_test.close()

    print(f"Done! Files saved to {output_dir}")

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for AKT from raw CSV.")
    parser.add_argument("input_file", type=str, help="Path to the raw dataset CSV file.")

    args = parser.parse_args()

    raw_path = args.input_file

    if not os.path.exists(raw_path):
        print(f"Error: File not found: {raw_path}")
        sys.exit(1)

    # Determine Base Directory and Dataset Name
    base_dir = os.path.dirname(os.path.abspath(raw_path))
    filename = os.path.basename(raw_path)

    # Heuristic for dataset name
    if "2009" in filename or "2010" in filename:
        dataset_name = "assist2009"
    elif "2012" in filename or "2013" in filename:
        dataset_name = "assist2012"
    else:
        dataset_name = os.path.splitext(filename)[0]

    # Path for the split file
    split_file_path = os.path.join(base_dir, f"{dataset_name}_split.csv")

    # Step 1: Logic to reuse create_splits
    if os.path.exists(split_file_path):
        print(f"Found existing split file: {split_file_path}")
        print("Loading...")
        df = pd.read_csv(split_file_path, encoding='ISO-8859-1', low_memory=False)
    else:
        # Call the imported function from root
        print(f"Split file not found. Generating using create_splits...")
        create_splits(raw_path, split_file_path)
        # Load the newly created file to ensure consistency
        df = pd.read_csv(split_file_path, encoding='ISO-8859-1', low_memory=False)

    # Step 2: Format for AKT
    process_for_akt(df, base_dir, dataset_name)