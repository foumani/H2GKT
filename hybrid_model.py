import copy
import torch
from torch import nn
from torch_geometric.nn import HeteroConv, GraphNorm, GINEConv, GATConv, SAGEConv
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LinkPredGNN(nn.Module):
    def __init__(self, data, hidden=32, edge_hidden=4, num_layers=2, dropout=0.0, norm=True):
        super().__init__()
        self.norm = norm
        self.edge_hidden = edge_hidden
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden = hidden

        self.embs = nn.ModuleDict()
        self.edge_enc = nn.ModuleDict()
        self.rels_with_attr = set()

        # 1. Node Embeddings
        for nt in data.node_types:
            if hasattr(data[nt], "x") and data[nt].x is not None:
                self.embs[nt] = nn.Linear(data[nt].x.shape[1], hidden)
            else:
                self.embs[nt] = nn.Embedding(data[nt].num_nodes, hidden)

        # 2. Edge Feature Encoders
        for et in data.edge_types:
            if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
                self.rels_with_attr.add(et)
                edge_dim = data[et].edge_attr.shape[1]
                self.edge_enc[str(et)] = nn.Sequential(
                    nn.Linear(edge_dim, edge_hidden),
                    nn.ReLU(),
                    nn.LayerNorm(edge_hidden)
                )

        # 3. Convolutions
        convs = []
        for _ in range(num_layers):
            conv_dict = {}
            for et in data.edge_types:
                if et in self.rels_with_attr:
                    # Dynamic Interaction
                    mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                    conv_dict[et] = GINEConv(mlp, edge_dim=edge_hidden)
                else:
                    # Static Structure
                    conv_dict[et] = GATConv((-1, -1), hidden, heads=4, concat=False, add_self_loops=False)
            convs.append(HeteroConv(conv_dict, aggr="sum"))
        self.convs = nn.ModuleList(convs)

        # 4. Norms
        if norm:
            self.norms = nn.ModuleList([
                nn.ModuleDict({nt: GraphNorm(hidden) for nt in data.node_types})
                for _ in range(num_layers)
            ])

        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        # 5. GNN-only case
        self.pred_head = nn.Sequential(
            nn.Linear(2*hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def encode(self, data):
        x_dict = {}
        # Init
        for nt in data.node_types:
            if hasattr(data[nt], "x") and data[nt].x is not None:
                x_dict[nt] = self.embs[nt](data[nt].x)
            else:
                x_dict[nt] = self.embs[nt].weight

        # Edge Attr
        edge_attr_dict = {}
        for et in data.edge_types:
            if et in self.rels_with_attr:
                 edge_attr_dict[et] = self.edge_enc[str(et)](data[et].edge_attr.float())

        # Propagate
        for l, conv in enumerate(self.convs):
            x_dict = conv(x_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict)
            if self.norm:
                x_dict = {nt: self.norms[l][nt](x) for nt, x in x_dict.items()}
            x_dict = {k: self.drop(self.act(v)) for k, v in x_dict.items()}
        return x_dict

    def decode(self, x_dict, edge_index):
        """
        Predicts link probability for specific edges.
        Args:
            x_dict: Dictionary of node embeddings {type: tensor}
            edge_index: [2, Num_Edges] tensor of (student_idx, problem_idx) pairs to predict
        """
        # Get source (student) and dest (problem) embeddings
        # Row 0 of edge_index is Student, Row 1 is Problem
        student_emb = x_dict['student'][edge_index[0]]
        problem_emb = x_dict['problem'][edge_index[1]]

        # Concatenate and Predict
        cat_emb = torch.cat([student_emb, problem_emb], dim=-1)
        logits = self.pred_head(cat_emb)
        return logits.squeeze(-1)  # Shape: [Num_Edges]

    def forward(self, data, target_edge_index=None):
        """
        Args:
            data: The HeteroData graph (Context)
            target_edge_index: (Optional) Specific edges to predict.
                               If None, just returns node embeddings.
        """
        # 1. Encode the graph (Context)
        x_dict = self.encode(data)

        # 2. Predict (if targets provided)
        if target_edge_index is not None:
            return self.decode(x_dict, target_edge_index)

        return x_dict


class HybridRNN(nn.Module):
    def __init__(self, gnn_hidden_dim, rnn_hidden_dim=64, num_layers=1, dropout=0.2, no_class=False, no_teacher=False, no_skill=False):
        super().__init__()
        self.no_class = no_class
        self.no_teacher = no_teacher
        self.no_skill = no_skill

        # Calculate input dim dynamically
        # Base: Problem(1)
        input_cnt = 1
        if not self.no_skill: input_cnt += 1
        if not self.no_class: input_cnt += 1
        if not self.no_teacher: input_cnt += 1

        # (Count * gnn_dim) + 1 (prev_correct)
        input_feature_dim = (gnn_hidden_dim * input_cnt) + 1

        self.input_proj = nn.Linear(input_feature_dim, rnn_hidden_dim)

        self.rnn = nn.LSTM(
            input_size=rnn_hidden_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden_dim, rnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(rnn_hidden_dim // 2, 1)
        )

        # Student Init Projection
        if gnn_hidden_dim != rnn_hidden_dim:
            self.student_init_proj = nn.Linear(gnn_hidden_dim, rnn_hidden_dim)
        else:
            self.student_init_proj = nn.Identity()

    def forward(self, batch_data, static_embs):
        # Unpack
        student_ids = batch_data['student_id']
        p_seq = batch_data['problem_seq']
        r_seq = batch_data['correct_seq']
        mask = batch_data['mask']

        # Lookups - Base is Problem
        emb_p = static_embs['problem'][p_seq]
        feats = [emb_p]

        # Conditional Lookups
        if not self.no_skill:
            s_seq = batch_data['skill_seq']
            emb_s = static_embs['skill'][s_seq]
            feats.append(emb_s)

        if not self.no_class:
            c_seq = batch_data['class_seq']
            emb_c = static_embs['class'][c_seq]
            feats.append(emb_c)

        if not self.no_teacher:
            t_seq = batch_data['teacher_seq']
            emb_t = static_embs['teacher'][t_seq]
            feats.append(emb_t)

        # Previous Correctness
        prev_r = torch.roll(r_seq, shifts=1, dims=1)
        prev_r[:, 0] = 0.0
        prev_r = prev_r.unsqueeze(-1)
        feats.append(prev_r)

        # Concat & Project
        x = torch.cat(feats, dim=-1)
        x = self.input_proj(x)

        # RNN Init
        student_static = static_embs['student'][student_ids]
        h0 = self.student_init_proj(student_static).unsqueeze(0)
        c0 = torch.zeros_like(h0)

        if self.rnn.num_layers > 1:
            h0 = h0.repeat(self.rnn.num_layers, 1, 1)
            c0 = c0.repeat(self.rnn.num_layers, 1, 1)

        # Pack
        lengths = mask.sum(dim=1).cpu()
        # Enforce length > 0 to prevent crash if batch is all padding (unlikely but safe)
        lengths = torch.clamp(lengths, min=1)

        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input, (h0, c0))
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=p_seq.shape[1])

        return self.fc(output).squeeze(-1)

    def emb(self, batch_data, static_embs):
        """
        Returns the raw RNN hidden states (embeddings) instead of predictions.
        Shape: [Batch, Seq_Len, RNN_Hidden_Dim]
        """
        # 1. Unpack Base Data
        student_ids = batch_data['student_id']
        p_seq = batch_data['problem_seq']
        r_seq = batch_data['correct_seq']
        mask = batch_data['mask']

        # 2. Lookups - Base is Problem
        emb_p = static_embs['problem'][p_seq]
        feats = [emb_p]

        # Conditional Lookups (Ablations)
        if not self.no_skill:
            s_seq = batch_data['skill_seq']
            emb_s = static_embs['skill'][s_seq]
            feats.append(emb_s)

        if not self.no_class:
            c_seq = batch_data['class_seq']
            emb_c = static_embs['class'][c_seq]
            feats.append(emb_c)

        if not self.no_teacher:
            t_seq = batch_data['teacher_seq']
            emb_t = static_embs['teacher'][t_seq]
            feats.append(emb_t)

        # Previous Correctness
        prev_r = torch.roll(r_seq, shifts=1, dims=1)
        prev_r[:, 0] = 0.0
        prev_r = prev_r.unsqueeze(-1)
        feats.append(prev_r)

        # 3. Project & RNN
        x = torch.cat(feats, dim=-1)
        x = self.input_proj(x)

        # RNN Init (Context-Aware)
        student_static = static_embs['student'][student_ids]
        h0 = self.student_init_proj(student_static).unsqueeze(0)
        c0 = torch.zeros_like(h0)

        if self.rnn.num_layers > 1:
            h0 = h0.repeat(self.rnn.num_layers, 1, 1)
            c0 = c0.repeat(self.rnn.num_layers, 1, 1)

        # Pack
        lengths = mask.sum(dim=1).cpu()
        lengths = torch.clamp(lengths, min=1)

        packed_input = pack_padded_sequence(x, lengths, batch_first=True,
                                            enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input, (h0, c0))
        output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                        total_length=p_seq.shape[1])

        # RETURN EMBEDDINGS DIRECTLY
        return output


class HybridTransformer(nn.Module):
    def __init__(self, gnn_hidden_dim, d_model=128, nhead=4, num_layers=2, dropout=0.3,
                 use_metadata=True):
        super().__init__()
        self.use_metadata = use_metadata
        self.d_model = d_model

        # Feature Projection
        input_feature_dim = (gnn_hidden_dim * 4) + 1 if use_metadata else (
                                                                                      gnn_hidden_dim * 2) + 1
        self.input_proj = nn.Linear(input_feature_dim, d_model)

        # Positional Encoding (Simple Learnable)
        self.pos_emb = nn.Embedding(500, d_model)  # Max Seq Len 500

        # Transformer Encoder (SAINT style)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, 1)

    def generate_causal_mask(self, sz):
        # Creates a triangular matrix to hide future tokens
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, batch_data, static_embs):
        # Unpack
        student_ids = batch_data['student_id']
        p_seq = batch_data['problem_seq']
        s_seq = batch_data['skill_seq']
        r_seq = batch_data['correct_seq']
        mask = batch_data['mask']

        # Lookups
        emb_p = static_embs['problem'][p_seq]
        emb_s = static_embs['skill'][s_seq]

        # Metadata Lookups
        feats = [emb_p, emb_s]
        if self.use_metadata:
            c_seq = batch_data['class_seq']
            t_seq = batch_data['teacher_seq']
            emb_c = static_embs['class'][c_seq]
            emb_t = static_embs['teacher'][t_seq]
            feats.extend([emb_c, emb_t])

        # Previous Correctness
        prev_r = torch.roll(r_seq, shifts=1, dims=1)
        prev_r[:, 0] = 0.0
        prev_r = prev_r.unsqueeze(-1)
        feats.append(prev_r)

        # Concat
        x = torch.cat(feats, dim=-1)

        # Project Features
        x = self.input_proj(x)  # [Batch, Seq, D_Model]

        # Add Positional Embeddings
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(positions)

        # Create Masks
        # 1. Padding Mask (True where padding exists)
        # Note: PyTorch Transformer expects True for BLOCKED items (Padding), False for REAL items
        key_padding_mask = ~mask  # Invert your mask (False=Real, True=Pad)

        # 2. Causal Mask (Square matrix)
        causal_mask = self.generate_causal_mask(seq_len).to(x.device)

        # Forward
        output = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask
        )

        return self.fc(output).squeeze(-1)

class EndToEndHybrid(nn.Module):
    def __init__(self, gnn_model, rnn_model):
        super().__init__()
        self.gnn = gnn_model
        self.rnn = rnn_model

    def forward(self, graph, batch_data):
        z_dict = self.gnn.encode(graph)
        return self.rnn(batch_data, z_dict)

    def get_embeddings(self, graph, batch_data):
        z_dict = self.gnn.encode(graph)
        return self.rnn.emb(batch_data, z_dict)