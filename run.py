import numpy as np
import torch

def train(model, opt, loader, criteria, gnn_only=False, manager=None):
    """
    Performs one training epoch.

    Args:
        gnn_only (bool): If True, trains using 'Masked Auto-Encoder' style on the graph structure.
                         If False, trains the Hybrid RNN on student sequences.
        manager: The DataManager containing the full graph and split indices.
    """
    model.train()

    # Define Edge Keys
    rel_key = ('student', 'answers', 'problem')
    rev_rel_key = ('problem', 'rev_answers', 'student')

    if gnn_only:
        # --- GNN-ONLY MODE (Edge Splitting / K-Fold Simulation) ---
        opt.zero_grad()

        # 1. Define Training Indices
        # Matches 'N_tr' and 'edge_index_spu' from old code
        train_idx = torch.cat([manager.gnn_train_idx, manager.rnn_train_idx])
        num_train = train_idx.size(0)

        # 2. Create Split (Simulating K=2 KFold)
        # We shuffle and split 50/50:
        # held = indices for PREDICTION (Targets)
        # struct_mask = indices for STRUCTURE (Input)
        perm = torch.randperm(num_train, device=train_idx.device)
        split = num_train // 2

        held = perm[:split]        # Edges we hide and predict
        struct_mask = perm[split:] # Edges we keep in the graph

        # 3. Build Training Graph (Structure Only)
        train_graph = manager.graph.clone()
        edge_store = manager.graph[rel_key]

        # Get absolute indices for structure
        abs_struct_idx = train_idx[struct_mask]

        # Extract Structure Edges
        struct_edge_index = edge_store.edge_index[:, abs_struct_idx].clone()
        struct_edge_attr = edge_store.edge_attr[abs_struct_idx].clone()

        # Assign to Graph (Forward)
        train_graph[rel_key].edge_index = struct_edge_index
        train_graph[rel_key].edge_attr = struct_edge_attr

        # Assign to Graph (Reverse - Symmetric)
        rev_struct_idx = torch.stack([struct_edge_index[1], struct_edge_index[0]], dim=0)
        train_graph[rev_rel_key].edge_index = rev_struct_idx
        train_graph[rev_rel_key].edge_attr = struct_edge_attr

        # 4. Forward Pass
        z_dict = model.encode(train_graph)

        # 5. Decode / Predict Targets
        abs_held_idx = train_idx[held]

        target_edge_index = edge_store.edge_index[:, abs_held_idx]
        batch_y = edge_store.edge_attr[abs_held_idx].view(-1)

        # Decode
        logits = model.decode(z_dict, target_edge_index)

        # 6. Loss & Optimization
        loss = criteria(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        return loss.item(), torch.sigmoid(logits).detach().cpu().numpy(), batch_y.cpu().numpy()
    else:
        # --- HYBRID MODE (Sequence Training) ---
        total_loss = 0
        preds = []
        targets = []

        # 1. Build Knowledge Graph
        # In Hybrid mode, the GNN provides a static "World View" based on the GNN_Train split.
        graph = manager.graph.clone()
        context_idx = manager.gnn_train_idx

        # Slice Forward Edges
        graph[rel_key].edge_index = manager.graph[rel_key].edge_index[:, context_idx].clone()
        graph[rel_key].edge_attr = manager.graph[rel_key].edge_attr[context_idx].clone()

        # Slice Reverse Edges
        graph[rev_rel_key].edge_index = manager.graph[rev_rel_key].edge_index[:, context_idx].clone()
        graph[rev_rel_key].edge_attr = manager.graph[rev_rel_key].edge_attr[context_idx].clone()

        # !!! UPDATED LOOP !!!
        # loader now yields a dictionary directly
        for batch_data in loader:
            opt.zero_grad()

            # Forward pass using the static graph context + dynamic sequences
            logits = model(graph, batch_data)

            # Apply Mask (Ignore padding)
            mask = batch_data['mask'].view(-1)
            flat_logits = logits.view(-1)[mask]
            flat_targets = batch_data['correct_seq'].view(-1)[mask]

            loss = criteria(flat_logits, flat_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += loss.item()
            preds.append(torch.sigmoid(flat_logits).detach().cpu().numpy())
            targets.append(flat_targets.cpu().numpy())

        avg_train_loss = total_loss / len(loader)
        return avg_train_loss, np.concatenate(preds), np.concatenate(targets)


def test(model, loader, criteria=None, gnn_only=False, manager=None, target_split='val'):
    """
    Evaluates the model.
    """
    model.eval()

    # Define Edge Keys
    rel_key = ('student', 'answers', 'problem')
    rev_rel_key = ('problem', 'rev_answers', 'student')
    if gnn_only:
        # --- GNN-ONLY MODE (Link Prediction) ---

        # 1. Determine Context and Targets
        if target_split == 'val':
            context_idx = torch.cat([manager.gnn_train_idx, manager.rnn_train_idx])
            target_idx = manager.val_idx
        elif target_split == 'test':
            context_idx = torch.cat(
                [manager.gnn_train_idx, manager.rnn_train_idx, manager.val_idx])
            target_idx = manager.test_idx
        else:
            raise ValueError(f"Unknown target_split: {target_split}")

        # 2. Build Context Graph
        context_graph = manager.graph.clone()
        edge_store = manager.graph[rel_key]

        context_edge_index = edge_store.edge_index[:, context_idx].clone()
        context_edge_attr = edge_store.edge_attr[context_idx].clone()

        context_graph[rel_key].edge_index = context_edge_index
        context_graph[rel_key].edge_attr = context_edge_attr

        rev_idx = torch.stack([context_edge_index[1], context_edge_index[0]], dim=0)
        context_graph[rev_rel_key].edge_index = rev_idx
        context_graph[rev_rel_key].edge_attr = context_edge_attr

        # 3. Identify Targets
        target_edge_index = edge_store.edge_index[:, target_idx]
        target_labels = edge_store.edge_attr[target_idx].view(-1)

        with torch.no_grad():
            # 4. Forward
            z_dict = model.encode(context_graph)

            # 5. Decode
            logits = model.decode(z_dict, target_edge_index)

            # 6. Metrics
            loss = 0.0
            if criteria:
                loss = criteria(logits, target_labels).item()

            return loss, torch.sigmoid(logits).cpu().numpy(), target_labels.cpu().numpy()
    else:
        # --- HYBRID MODE (Sequence Evaluation) ---
        total_loss = 0
        preds = []
        targets = []

        # 1. Build Knowledge Graph
        graph = manager.graph.clone()
        context_idx = manager.gnn_train_idx

        # Slice Forward
        graph[rel_key].edge_index = manager.graph[rel_key].edge_index[:, context_idx].clone()
        graph[rel_key].edge_attr = manager.graph[rel_key].edge_attr[context_idx].clone()

        # Slice Reverse
        graph[rev_rel_key].edge_index = manager.graph[rev_rel_key].edge_index[:, context_idx].clone()
        graph[rev_rel_key].edge_attr = manager.graph[rev_rel_key].edge_attr[context_idx].clone()

        with torch.no_grad():
            # !!! UPDATED LOOP !!!
            for batch_data in loader:
                # batch_data is now directly a dictionary.
                logits = model(graph, batch_data)

                # Use 'eval_mask' to select only the prediction targets
                mask = batch_data['eval_mask'].view(-1)

                if mask.sum() == 0: continue

                flat_logits = logits.view(-1)[mask]
                flat_targets = batch_data['correct_seq'].view(-1)[mask]

                if criteria is not None:
                    val_loss = criteria(flat_logits, flat_targets)
                    total_loss += val_loss.item()

                preds.append(torch.sigmoid(flat_logits).cpu().numpy())
                targets.append(flat_targets.cpu().numpy())

        avg_val_loss = total_loss / len(loader) if len(loader) > 0 else 0.0

        if len(preds) == 0:
            return avg_val_loss, np.array([]), np.array([])

        return avg_val_loss, np.concatenate(preds), np.concatenate(targets)