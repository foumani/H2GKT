import copy
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from create_splits import create_splits
# Custom Modules
from hybrid_data_manager import HybridDataManager
from hybrid_model import LinkPredGNN, HybridRNN, EndToEndHybrid, HybridTransformer
from utils import compute_metrics, create_dataloader
import run

def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Arguments
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Hybrid GNN-RNN Knowledge Tracing")
    parser.add_argument("--data-dir", type=str, default="data", help="Base data directory")
    parser.add_argument("--dataset", type=str, default="assist2009", choices=["assist2009", "assist2012"], help="Dataset name")
    parser.add_argument("--no-class", action="store_true", help="Ablation: Remove Class nodes")
    parser.add_argument("--no-teacher", action="store_true", help="Ablation: Remove Teacher nodes")
    parser.add_argument("--no-skill", action="store_true", help="Ablation: Remove Skill nodes")
    parser.add_argument("--no-rnn", action="store_true", help="Ablation: GNN-Only Mode (Link Prediction)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic training (slower)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-model", action="store_true", help="Save the best model to disk")
    parser.add_argument("--load-model", type=str, default=None, help="Path to a .pth file to load before training/testing")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory where models and logs are saved")
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # 2. Setup Device & Seeds
    # --------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    print(f"Dataset: {args.dataset} | No Class: {args.no_class} | No Teacher: {args.no_teacher} | No Skill: {args.no_skill} | Mode: {'GNN-Only' if args.no_rnn else 'Hybrid'}")

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.deterministic:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print(f"Deterministic Mode: ON (Seed: {args.seed})")
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"Deterministic Mode: OFF (Optimizing for speed)")

    # --------------------------------------------------------------------------
    # 3. Data Loading & Manager
    # --------------------------------------------------------------------------
    if args.dataset == "assist2009":
        data_folder = os.path.join(args.data_dir, "assist2009")
        raw_path = os.path.join(data_folder, "ASSISTments_2009_2010_skill_builder_data.csv")
        split_path = os.path.join(data_folder, "assist2009_split.csv")
        if not os.path.exists(split_path):
            print(f"[Info] Split file not found. Generating {split_path}...")
            create_splits(raw_path, split_path, "assist2009")
        print(f"[Info] Loading data from {split_path}...")
        df = pd.read_csv(split_path, encoding='ISO-8859-1', low_memory=False)

    elif args.dataset == "assist2012":
        data_folder = os.path.join(args.data_dir, "assist2012")
        raw_path = os.path.join(data_folder, "2012-2013-data-with-predictions-4-final.csv")
        split_path = os.path.join(data_folder, "assist2012_split.csv")
        if not os.path.exists(split_path):
            print(f"[Info] Split file not found. Generating {split_path}...")
            create_splits(raw_path, split_path, "assist2012")
        print(f"[Info] Loading data from {split_path}...")
        df = pd.read_csv(split_path, low_memory=False)


    manager = HybridDataManager(
        df,
        fractions=(0.6, 0.2, 0.1, 0.1), # GNN_Train, RNN_Train, Val, Test
        seq_len=200,
        batch_size=args.batch_size,
        gnn_only=args.no_rnn,
        no_class=args.no_class,
        no_teacher=args.no_teacher,
        no_skill=args.no_skill
    )
    # Move all internal tensors to GPU
    manager.to(device)

    # Create DataLoaders manually (Since Manager no longer holds them)
    if args.no_rnn:
        train_loader, val_loader, test_loader = None, None, None
    else:
        train_loader = create_dataloader(manager.train_seq, batch_size=args.batch_size, shuffle=True)
        val_loader   = create_dataloader(manager.val_seq, batch_size=args.batch_size, shuffle=False)
        test_loader  = create_dataloader(manager.test_seq, batch_size=args.batch_size, shuffle=False)

    # --------------------------------------------------------------------------
    # 4. Model Initialization
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("INITIALIZING MODELS")
    print("=" * 50)

    # Construct the save path based on dataset and seed
    if args.save_model:
        os.makedirs(args.log_dir, exist_ok=True)
        save_path = os.path.join(args.log_dir, f"H2_GKT_{args.dataset}_seed{args.seed}_best_model.pth")
        print(f"Best model will be saved to: {save_path}")

    # GNN is always required (It provides embeddings or predictions)
    gnn_model = LinkPredGNN(
        manager.graph,
        hidden=64,
        edge_hidden=6,
        num_layers=2,
        dropout=0.4,
        norm=True
    ).to(device)

    if args.no_rnn:
        # --- ABLATION: GNN Only ---
        model = gnn_model
        # GNN usually benefits from a higher LR when trained alone
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    else:
        # --- FULL HYBRID ---
        rnn_model = HybridRNN(
            gnn_hidden_dim=gnn_model.hidden,
            rnn_hidden_dim=64,
            num_layers=2,
            dropout=0.5,
            no_class=args.no_class,
            no_teacher=args.no_teacher,
            no_skill=args.no_skill
        ).to(device)

        # Combine GNN + RNN
        model = EndToEndHybrid(gnn_model, rnn_model)

        opt = torch.optim.Adam([
            {'params': rnn_model.parameters(), 'lr': args.lr, 'weight_decay': 1e-5},
            {'params': gnn_model.parameters(), 'lr': args.lr, 'weight_decay': 5e-4}
        ])

    if args.no_rnn:
        # "Disable" it by making it wait forever. GNN only mode does not work well with the scheduler.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=999999)
    else:
        # In the normal mode we allow it.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()
    # --- LOAD PRE-TRAINED WEIGHTS IF REQUESTED ---
    if args.load_model:
        print(f"Loading model from {args.load_model}...")
        if os.path.exists(args.load_model):
            checkpoint = torch.load(args.load_model, map_location=device)
            model.load_state_dict(checkpoint)
            print("Model loaded successfully.")
        else:
            print(f"Error: Model file {args.load_model} not found!")
            exit(1)

    # --------------------------------------------------------------------------
    # 5. Training Loop
    # --------------------------------------------------------------------------
    best_auc = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        # A. Train
        avg_tr_loss, tr_probs, tr_targets = run.train(
            model, opt, train_loader, criterion,
            gnn_only=args.no_rnn, manager=manager
        )

        # B. Validate
        # Note: We pass target_split='val' to tell the GNN which edges to predict
        avg_val_loss, val_probs, val_targets = run.test(
            model, val_loader, criterion,
            gnn_only=args.no_rnn, manager=manager, target_split='val'
        )

        # C. Test Monitor (Optional, but good for tracking)
        _, test_probs, test_targets = run.test(
            model, test_loader, criteria=None,
            gnn_only=args.no_rnn, manager=manager, target_split='test'
        )

        # D. Metrics
        tr_acc, tr_prec, tr_rec, tr_f1, tr_auc = compute_metrics(tr_targets, tr_probs)
        val_acc, val_prec, val_rec, val_f1, val_auc = compute_metrics(val_targets, val_probs)
        test_acc, test_prec, test_rec, test_f1, test_auc = compute_metrics(test_targets, test_probs)

        print(f'Epoch {epoch:03d} {"-" * 100}')
        print(f"[Train] Loss: {avg_tr_loss:.4f} | ACC: {tr_acc:.4f} | PREC: {tr_prec:.4f} | REC: {tr_rec:.4f} | F1: {tr_f1:.4f} | AUC: {tr_auc}")
        print(f"[Val  ] Loss: {avg_val_loss:.4f} | ACC: {val_acc:.4f} | PREC: {val_prec:.4f} | REC: {val_rec:.4f} | F1: {val_f1:.4f} | AUC: {val_auc}")
        print(f"[Test ] Loss: ------ | ACC: {test_acc:.4f} | PREC: {test_prec:.4f} | REC: {test_rec:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f}")

        # E. Save Best
        scheduler.step(val_auc)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())

            # Save to disk if flag is active
            if args.save_model:
                torch.save(best_state, save_path)
                print(f"Saved new best model to {save_path}")

    # --------------------------------------------------------------------------
    # 6. Final Testing
    # --------------------------------------------------------------------------
    print("\n" + "="*50 + "\nFINAL TESTING (Best Epoch)\n" + "="*50)
    model.load_state_dict(best_state)

    _, test_probs, test_targets = run.test(
        model, test_loader, criteria=None,
        gnn_only=args.no_rnn, manager=manager, target_split='test'
    )

    test_acc, test_prec, test_rec, test_f1, test_auc = compute_metrics(test_targets, test_probs)

    print(f"Best Val AUC: {best_auc:.4f}")
    print(f"[Test ] ACC: {test_acc:.4f} | PREC: {test_prec:.4f} | REC: {test_rec:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f}")

if __name__ == "__main__":
    main()