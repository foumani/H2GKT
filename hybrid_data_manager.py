import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import torch

from utils import create_dataloader

class HybridDataManager:
    def __init__(self, raw_df, fractions=(0.6, 0.2, 0.1, 0.1), seq_len=100, batch_size=128, gnn_only=False, no_class=False, no_teacher=False, no_skill=False):
        """
        Manages graph construction, data splitting, and sequence generation.
        """
        self.no_class = no_class
        self.no_teacher = no_teacher
        self.no_skill = no_skill
        self.seq_len = seq_len
        self.batch_size = batch_size

        # 1. Preprocess Data
        # Sort chronologically to ensure valid time-based splitting
        self.raw_df = raw_df.sort_values(by=['user_id', 'order_id']).reset_index(drop=True)

        self._handle_missing_metadata()

        # 2. Initialize Mappings
        self._init_mappings()

        # 3. Build Static Graph
        # Constructs the full graph structure (nodes + all interaction edges)
        self.graph = self._build_full_graph()

        # 4. Generate Split Indices
        # Divides interactions into: GNN Training (Structure), RNN Training (History), Validation, and Test
        self.gnn_train_idx, self.rnn_train_idx, self.val_idx, self.test_idx = self._generate_split_indices(fractions)

        # 5. Build Sequences
        if not gnn_only:
            # Generate padded sequences for RNN input
            self.train_seq = self._build_train_seq(seq_len)
            self.val_seq = self._build_val_seq(seq_len)
            self.test_seq = self._build_test_seq(seq_len)

    def _handle_missing_metadata(self):
        # Fill missing class/teacher columns with default 0 ID
        for col in ['student_class_id', 'teacher_id']:
            if col not in self.raw_df.columns:
                print(f"Warning: '{col}' missing. Filling with 0.")
                self.raw_df[col] = 0

    def _init_mappings(self):
        # Maps raw IDs (e.g., Problem ID 5021) to contiguous 0-indexed Integers
        self.unique_students = self.raw_df["user_id"].unique()
        self.n_students = len(self.unique_students)
        self.student_map = {int(uid): i+1 for i, uid in enumerate(self.unique_students)}

        self.unique_problems = self.raw_df["problem_id"].unique()
        self.n_problems = len(self.unique_problems)
        self.problem_map = {int(pid): i+1 for i, pid in enumerate(self.unique_problems)}

       # Skill
        if not self.no_skill:
            self.unique_skills = self.raw_df["skill_id"].unique()
            self.n_skills = len(self.unique_skills)
            self.skill_map = {int(sid): i+1 for i, sid in enumerate(self.unique_skills)}
        else:
            self.n_skills = 0
            self.skill_map = {}

        # Class
        if not self.no_class:
            self.unique_classes = self.raw_df["student_class_id"].unique()
            self.n_classes = len(self.unique_classes)
            self.class_map = {int(cid): i+1 for i, cid in enumerate(self.unique_classes)}
        else:
            self.n_classes = 0
            self.class_map = {}

        # Teacher
        if not self.no_teacher:
            self.unique_teachers = self.raw_df["teacher_id"].unique()
            self.n_teachers = len(self.unique_teachers)
            self.teacher_map = {int(tid): i+1 for i, tid in enumerate(self.unique_teachers)}
        else:
            self.n_teachers = 0
            self.teacher_map = {}

        print(f"Stats: {self.n_students} Students, {self.n_problems} Problems, {self.n_skills} Skills, {self.n_classes} Classes, {self.n_teachers} Teachers")

    # def _build_full_graph(self):
    #     # Creates the HeteroData object containing all static relationships and dynamic interactions
    #     data = HeteroData()
    #
    #     # A. Set Node Counts
    #     data["student"].num_nodes = self.n_students+1
    #     data["problem"].num_nodes = self.n_problems+1
    #     data["skill"].num_nodes = self.n_skills+1
    #
    #     if self.use_metadata:
    #         data["class"].num_nodes = self.n_classes+1
    #         data["teacher"].num_nodes = self.n_teachers+1
    #
    #     # B. Helper for Static Edges
    #     def add_static_edge(src_col, dst_col, src_map, dst_map, edge_name, rev_name, src_node, dst_node):
    #         pairs = self.raw_df[[src_col, dst_col]].dropna().drop_duplicates()
    #         src = [src_map[int(x)] for x in pairs[src_col]]
    #         dst = [dst_map[int(x)] for x in pairs[dst_col]]
    #         idx = torch.tensor([src, dst], dtype=torch.long)
    #         data[src_node, edge_name, dst_node].edge_index = idx
    #         data[dst_node, rev_name, src_node].edge_index = torch.stack([idx[1], idx[0]], dim=0)
    #
    #     # Static Edges
    #     add_static_edge('problem_id', 'skill_id', self.problem_map, self.skill_map,
    #                     'has_skill', 'rev_has_skill', 'problem', 'skill')
    #
    #     if self.use_metadata:
    #         add_static_edge('user_id', 'student_class_id', self.student_map, self.class_map,
    #                         'member_of', 'rev_member_of', 'student', 'class')
    #         add_static_edge('student_class_id', 'teacher_id', self.class_map, self.teacher_map,
    #                         'taught_by', 'rev_taught_by', 'class', 'teacher')
    #
    #     # C. Add Dynamic Interaction Edges (Full History)
    #     src = [self.student_map[int(x)] for x in self.raw_df['user_id']]
    #     dst = [self.problem_map[int(x)] for x in self.raw_df['problem_id']]
    #
    #     edge_index = torch.tensor([src, dst], dtype=torch.long)
    #     edge_attr = torch.tensor(self.raw_df['correct'].values, dtype=torch.float).view(-1, 1)
    #
    #     data['student', 'answers', 'problem'].edge_index = edge_index
    #     data['student', 'answers', 'problem'].edge_attr = edge_attr
    #     data['problem', 'rev_answers', 'student'].edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
    #     data['problem', 'rev_answers', 'student'].edge_attr = edge_attr
    #
    #     return data

    def _build_full_graph(self):
        data = HeteroData()

        # A. Set Node Counts
        data["student"].num_nodes = self.n_students+1
        data["problem"].num_nodes = self.n_problems+1

        if not self.no_skill: data["skill"].num_nodes = self.n_skills+1
        if not self.no_class: data["class"].num_nodes = self.n_classes+1
        if not self.no_teacher: data["teacher"].num_nodes = self.n_teachers+1

        # B. Helper
        def add_static_edge(src_col, dst_col, src_map, dst_map, edge_name, rev_name, src_node, dst_node):
            pairs = self.raw_df[[src_col, dst_col]].dropna().drop_duplicates()
            src = [src_map[int(x)] for x in pairs[src_col]]
            dst = [dst_map[int(x)] for x in pairs[dst_col]]
            idx = torch.tensor([src, dst], dtype=torch.long)
            data[src_node, edge_name, dst_node].edge_index = idx
            data[dst_node, rev_name, src_node].edge_index = torch.stack([idx[1], idx[0]], dim=0)

        # C. Static Edges
        if not self.no_skill:
            add_static_edge('problem_id', 'skill_id', self.problem_map, self.skill_map, 'has_skill', 'rev_has_skill', 'problem', 'skill')

        if not self.no_class:
            add_static_edge('user_id', 'student_class_id', self.student_map, self.class_map, 'member_of', 'rev_member_of', 'student', 'class')

        # Teacher edge requires both Teacher AND Class (since it links Class->Teacher)
        if not self.no_teacher and not self.no_class:
            add_static_edge('student_class_id', 'teacher_id', self.class_map, self.teacher_map, 'taught_by', 'rev_taught_by', 'class', 'teacher')

        # D. Dynamic Edges
        src = [self.student_map[int(x)] for x in self.raw_df['user_id']]
        dst = [self.problem_map[int(x)] for x in self.raw_df['problem_id']]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(self.raw_df['correct'].values, dtype=torch.float).view(-1, 1)

        data['student', 'answers', 'problem'].edge_index = edge_index
        data['student', 'answers', 'problem'].edge_attr = edge_attr
        data['problem', 'rev_answers', 'student'].edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        data['problem', 'rev_answers', 'student'].edge_attr = edge_attr

        return data

    def _generate_split_indices(self, fractions):
        # Use the pre-computed 'split' column from match_dataset.py
        if 'split' in self.raw_df.columns:
            print("Using pre-computed 'split' column for data partitioning.")

            # Since raw_df index is reset to 0..N in __init__, we can just grab indices
            idx_gnn = self.raw_df.index[self.raw_df['split'] == 'train_gnn'].values
            idx_rnn = self.raw_df.index[self.raw_df['split'] == 'train_rnn'].values
            idx_val = self.raw_df.index[self.raw_df['split'] == 'val'].values
            idx_test = self.raw_df.index[self.raw_df['split'] == 'test'].values

            return (torch.tensor(idx_gnn, dtype=torch.long),
                    torch.tensor(idx_rnn, dtype=torch.long),
                    torch.tensor(idx_val, dtype=torch.long),
                    torch.tensor(idx_test, dtype=torch.long))

    def _select_rnn_data(self, indices, seq_len=100):
        # Selects rows based on indices for sequence processing
        rnn_df_slice = self.raw_df.iloc[indices.cpu().numpy()].copy()
        return self._extract_sequences(rnn_df_slice, seq_len=seq_len)

    def _build_train_seq(self, seq_len=100):
        # Training Sequences (for RNN learning)
        rnn_data = self._select_rnn_data(self.rnn_train_idx, seq_len)
        if rnn_data:
            print(f"RNN Training View: {len(rnn_data['student_id'])} students.")
        return rnn_data

    def _build_val_seq(self, seq_len=100):
        # Validation Sequences (for Model Tuning)
        val_seq_data = self._select_rnn_data(self.val_idx, seq_len=seq_len)
        if val_seq_data:
            print(f"Validation RNN View: {len(val_seq_data['student_id'])} students.")
        return val_seq_data

    def _build_test_seq(self, seq_len=100):
        # Test Sequences (Includes full context: RNN_Train + Validation + Test)
        # Masks earlier parts so loss is only calculated on the Test portion.
        df_context1 = self.raw_df.iloc[self.rnn_train_idx.cpu().numpy()].copy()
        df_context1['is_target'] = False

        df_context2 = self.raw_df.iloc[self.val_idx.cpu().numpy()].copy()
        df_context2['is_target'] = False

        df_target = self.raw_df.iloc[self.test_idx.cpu().numpy()].copy()
        df_target['is_target'] = True

        df_full = pd.concat([df_context1, df_context2, df_target]).sort_values(['user_id', 'order_id'])

        test_seq_data = self._extract_sequences(df_full, seq_len=seq_len)
        if test_seq_data:
            print(f"Test RNN View (Full Context): {len(test_seq_data['student_id'])} students.")
        return test_seq_data

    def _extract_sequences(self, df, seq_len=100):
        if df.empty: return None
        if 'is_target' not in df.columns: df['is_target'] = True

        grouped = df.groupby('user_id')

        # Init out dict ONLY with base keys + correct
        keys = ['student_id', 'problem_seq', 'correct_seq', 'mask', 'eval_mask']
        if not self.no_skill: keys.append('skill_seq')
        if not self.no_class: keys.append('class_seq')
        if not self.no_teacher: keys.append('teacher_seq')
        out = {k: [] for k in keys}

        for uid, group in grouped:
            if int(uid) not in self.student_map: continue

            # Base features
            p = [self.problem_map[int(x)] for x in group['problem_id'].values]
            corr = group['correct'].values.tolist()
            is_target_list = group['is_target'].values.tolist()

            # Optional features
            s = [self.skill_map[int(x)] for x in
                 group['skill_id'].values] if not self.no_skill else None
            c = [self.class_map[int(x)] for x in
                 group['student_class_id'].values] if not self.no_class else None
            t = [self.teacher_map[int(x)] for x in
                 group['teacher_id'].values] if not self.no_teacher else None

            # Pad/Truncate
            cur_len = len(p)
            if cur_len > seq_len:
                p, corr = p[-seq_len:], corr[-seq_len:]
                mask = [True] * seq_len
                eval_mask = is_target_list[-seq_len:]
                if s: s = s[-seq_len:]
                if c: c = c[-seq_len:]
                if t: t = t[-seq_len:]
            else:
                pad_len = seq_len - cur_len
                pad = [0] * pad_len
                p += pad;
                corr += pad
                mask = [True] * cur_len + [False] * pad_len
                eval_mask = is_target_list + [False] * pad_len
                if s: s += pad
                if c: c += pad
                if t: t += pad

            out['student_id'].append(self.student_map[int(uid)])
            out['problem_seq'].append(p)
            out['correct_seq'].append(corr)
            out['mask'].append(mask)
            out['eval_mask'].append(eval_mask)
            if s: out['skill_seq'].append(s)
            if c: out['class_seq'].append(c)
            if t: out['teacher_seq'].append(t)

        # Convert to Tensor
        res = {
            'student_id': torch.tensor(out['student_id'], dtype=torch.long),
            'problem_seq': torch.tensor(out['problem_seq'], dtype=torch.long),
            'correct_seq': torch.tensor(out['correct_seq'], dtype=torch.float),
            'mask': torch.tensor(out['mask'], dtype=torch.bool),
            'eval_mask': torch.tensor(out['eval_mask'], dtype=torch.bool)
        }
        if not self.no_skill: res['skill_seq'] = torch.tensor(out['skill_seq'],
                                                              dtype=torch.long)
        if not self.no_class: res['class_seq'] = torch.tensor(out['class_seq'],
                                                              dtype=torch.long)
        if not self.no_teacher: res['teacher_seq'] = torch.tensor(out['teacher_seq'],
                                                                  dtype=torch.long)

        return res

    def to(self, device):
        # Moves indices, graph, and sequence tensors to the specified device
        self.gnn_train_idx = self.gnn_train_idx.to(device)
        self.rnn_train_idx = self.rnn_train_idx.to(device)
        self.val_idx = self.val_idx.to(device)
        self.test_idx = self.test_idx.to(device)
        self.graph = self.graph.to(device)

        # Move RNN sequence dictionaries
        def move_seq_dict(seq_dict):
            if seq_dict is None: return None
            return {k: v.to(device) for k, v in seq_dict.items()}

        if hasattr(self, 'train_seq'): self.train_seq = move_seq_dict(self.train_seq)
        if hasattr(self, 'val_seq'):   self.val_seq = move_seq_dict(self.val_seq)
        if hasattr(self, 'test_seq'):  self.test_seq = move_seq_dict(self.test_seq)
        return self
