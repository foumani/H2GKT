import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import build_dense_graph

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com


class KTDataset(Dataset):
    def __init__(self, features, questions, answers, masks):
        super(KTDataset, self).__init__()
        self.features = features
        self.questions = questions
        self.answers = answers
        self.masks = masks

    def __getitem__(self, index):
        return self.features[index], self.questions[index], self.answers[index], self.masks[index]

    def __len__(self):
        return len(self.features)


# In processing.py
def pad_collate(batch):
    (features, questions, answers, masks) = zip(*batch)

    MAX_LEN = 200  # <--- Drastic speedup.

    # Slice the ends (most recent interactions are what matter)
    features = [torch.LongTensor(feat[-MAX_LEN:]) for feat in features]
    questions = [torch.LongTensor(qt[-MAX_LEN:]) for qt in questions]
    answers = [torch.LongTensor(ans[-MAX_LEN:]) for ans in answers]

    masks = [torch.tensor(msk[-MAX_LEN:], dtype=torch.bool) for msk in masks]
    # ----------------------

    feature_pad = pad_sequence(features, batch_first=True, padding_value=0)
    question_pad = pad_sequence(questions, batch_first=True, padding_value=0)
    answer_pad = pad_sequence(answers, batch_first=True, padding_value=0)
    mask_pad = pad_sequence(masks, batch_first=True, padding_value=False)

    return feature_pad, question_pad, answer_pad, mask_pad

# changed their load_dataset to ours
# def load_dataset(file_path, batch_size, graph_type, dkt_graph_path=None, train_ratio=0.7, val_ratio=0.2, shuffle=True, model_type='GKT', use_binary=True, res_len=2, use_cuda=True):
#     r"""
#     Parameters:
#         file_path: input file path of knowledge tracing data
#         batch_size: the size of a student batch
#         graph_type: the type of the concept graph
#         shuffle: whether to shuffle the dataset or not
#         use_cuda: whether to use GPU to accelerate training speed
#     Return:
#         concept_num: the number of all concepts(or questions)
#         graph: the static graph is graph type is in ['Dense', 'Transition', 'DKT'], otherwise graph is None
#         train_data_loader: data loader of the training dataset
#         valid_data_loader: data loader of the validation dataset
#         test_data_loader: data loader of the test dataset
#     NOTE: stole some code from https://github.com/lccasagrande/Deep-Knowledge-Tracing/blob/master/deepkt/data_util.py
#     """
#     df = pd.read_csv(file_path)
#     if "skill_id" not in df.columns:
#         raise KeyError(f"The column 'skill_id' was not found on {file_path}")
#     if "correct" not in df.columns:
#         raise KeyError(f"The column 'correct' was not found on {file_path}")
#     if "user_id" not in df.columns:
#         raise KeyError(f"The column 'user_id' was not found on {file_path}")
#
#     # if not (df['correct'].isin([0, 1])).all():
#     #     raise KeyError(f"The values of the column 'correct' must be 0 or 1.")
#
#     # Step 1.1 - Remove questions without skill
#     df.dropna(subset=['skill_id'], inplace=True)
#
#     # Step 1.2 - Remove users with a single answer
#     df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()
#
#     # Step 2 - Enumerate skill id
#     df['skill'], _ = pd.factorize(df['skill_id'], sort=True)  # we can also use problem_id to represent exercises
#
#     # Step 3 - Cross skill id with answer to form a synthetic feature
#     # use_binary: (0,1); !use_binary: (1,2,3,4,5,6,7,8,9,10,11,12). Either way, the correct result index is guaranteed to be 1
#     if use_binary:
#         df['skill_with_answer'] = df['skill'] * 2 + df['correct']
#     else:
#         df['skill_with_answer'] = df['skill'] * res_len + df['correct'] - 1
#
#
#     # Step 4 - Convert to a sequence per user id and shift features 1 timestep
#     feature_list = []
#     question_list = []
#     answer_list = []
#     seq_len_list = []
#
#     def get_data(series):
#         feature_list.append(series['skill_with_answer'].tolist())
#         question_list.append(series['skill'].tolist())
#         answer_list.append(series['correct'].eq(1).astype('int').tolist())
#         seq_len_list.append(series['correct'].shape[0])
#
#     df.groupby('user_id').apply(get_data)
#     max_seq_len = np.max(seq_len_list)
#     print('max seq_len: ', max_seq_len)
#     student_num = len(seq_len_list)
#     print('student num: ', student_num)
#     feature_dim = int(df['skill_with_answer'].max() + 1)
#     print('feature_dim: ', feature_dim)
#     question_dim = int(df['skill'].max() + 1)
#     print('question_dim: ', question_dim)
#     concept_num = question_dim
#
#     # print('feature_dim:', feature_dim, 'res_len*question_dim:', res_len*question_dim)
#     # assert feature_dim == res_len * question_dim
#
#     kt_dataset = KTDataset(feature_list, question_list, answer_list)
#     train_size = int(train_ratio * student_num)
#     val_size = int(val_ratio * student_num)
#     test_size = student_num - train_size - val_size
#     train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(kt_dataset, [train_size, val_size, test_size])
#     print('train_size: ', train_size, 'val_size: ', val_size, 'test_size: ', test_size)
#
#     train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
#     valid_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
#     test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
#
#     graph = None
#     if model_type == 'GKT':
#         if graph_type == 'Dense':
#             graph = build_dense_graph(concept_num)
#         elif graph_type == 'Transition':
#             graph = build_transition_graph(question_list, seq_len_list, train_dataset.indices, student_num, concept_num)
#         elif graph_type == 'DKT':
#             graph = build_dkt_graph(dkt_graph_path, concept_num)
#         if use_cuda and graph_type in ['Dense', 'Transition', 'DKT']:
#             graph = graph.cuda()
#     return concept_num, graph, train_data_loader, valid_data_loader, test_data_loader

# def load_dataset(file_path, batch_size, graph_type, dkt_graph_path=None,
#                  train_ratio=None, val_ratio=None, shuffle=True,  # Ignoring ratio args
#                  model_type='GKT', use_binary=True, res_len=2, use_cuda=True):
#     print(f"Loading and splitting data from {file_path}...")
#     df = pd.read_csv(file_path)
#
#     # Check for split column
#     if "split" not in df.columns:
#         raise KeyError(f"The column 'split' was not found on {file_path}. Please run create_splits.py first.")
#
#     # Standard Preprocessing (Same as original GKT)
#     df.dropna(subset=['skill_id'], inplace=True)
#     df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()
#
#     # Global Skill Mapping
#     df['skill'], _ = pd.factorize(df['skill_id'], sort=True)
#
#     if use_binary:
#         df['skill_with_answer'] = df['skill'] * 2 + df['correct']
#     else:
#         df['skill_with_answer'] = df['skill'] * res_len + df['correct'] - 1
#
#     # Initialize Lists for our 3 Datasets
#     # Each list holds: features, questions, answers, masks
#     train_feat, train_q, train_ans, train_mask = [], [], [], []
#     val_feat, val_q, val_ans, val_mask = [], [], [], []
#     test_feat, test_q, test_ans, test_mask = [], [], [], []
#
#     seq_len_list = []  # For GKT graph building
#     question_list = []  # For GKT graph building
#
#     # Group by User and Slice
#     grouped = df.groupby('user_id')
#
#     for uid, group in grouped:
#         # Get full sequence
#         f = group['skill_with_answer'].values.tolist()
#         q = group['skill'].values.tolist()
#         a = group['correct'].values.tolist()
#         splits = group['split'].values.tolist()
#
#         seq_len_list.append(len(f))
#         question_list.append(q)
#
#         # --- 1. Construct TRAIN Set (0 - 80%) ---
#         # Includes 'train_gnn' and 'train_rnn'
#         train_indices = [i for i, x in enumerate(splits) if 'train' in x]
#
#         if train_indices:
#             stop = train_indices[-1] + 1
#             train_feat.append(f[:stop])
#             train_q.append(q[:stop])
#             train_ans.append(a[:stop])
#             # Mask: True for everything (Grade the whole history during training)
#             train_mask.append([True] * stop)
#
#         # --- 2. Construct VALIDATION Set (0 - 90%) ---
#         # Includes Train (Context) + Val (Target)
#         val_indices = [i for i, x in enumerate(splits) if x == 'val']
#
#         if val_indices:
#             stop = val_indices[-1] + 1
#             val_feat.append(f[:stop])
#             val_q.append(q[:stop])
#             val_ans.append(a[:stop])
#
#             # Mask: False for Context, True for Val
#             msk = [False] * stop
#             for i in val_indices: msk[i] = True
#             val_mask.append(msk)
#
#         # --- 3. Construct TEST Set (0 - 100%) ---
#         # Includes Train + Val (Context) + Test (Target)
#         test_indices = [i for i, x in enumerate(splits) if x == 'test']
#
#         if test_indices:
#             stop = test_indices[-1] + 1  # Likely end of list
#             test_feat.append(f[:stop])
#             test_q.append(q[:stop])
#             test_ans.append(a[:stop])
#
#             # Mask: False for Context, True for Test
#             msk = [False] * stop
#             for i in test_indices: msk[i] = True
#             test_mask.append(msk)
#
#     # Convert to Datasets
#     train_dataset = KTDataset(train_feat, train_q, train_ans, train_mask)
#     val_dataset = KTDataset(val_feat, val_q, val_ans, val_mask)
#     test_dataset = KTDataset(test_feat, test_q, test_ans, test_mask)
#
#     print(
#         f"Split sizes | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
#
#     # Helper to create loaders
#     # Note: We usually keep shuffle=True for Train, False for Val/Test
#     train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
#                                    collate_fn=pad_collate)
#     valid_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
#                                    collate_fn=pad_collate)
#     test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
#                                   collate_fn=pad_collate)
#
#     concept_num = int(df['skill'].max() + 1)
#
#     # Build Graph (Original Logic Preserved)
#     graph = None
#     if model_type == 'GKT':
#         if graph_type == 'Dense':
#             graph = build_dense_graph(concept_num)
#         elif graph_type == 'Transition':
#             # Note: We must pass indices carefully if using Transition graph,
#             # but for Dense/DKT (most common) this is fine.
#             # Passing range(len) as indices for now since we built lists manually
#             graph = build_transition_graph(question_list, seq_len_list,
#                                            range(len(question_list)), len(question_list),
#                                            concept_num)
#         elif graph_type == 'DKT':
#             graph = build_dkt_graph(dkt_graph_path, concept_num)
#         if use_cuda and graph is not None:
#             graph = graph.cuda()
#
#     return concept_num, graph, train_data_loader, valid_data_loader, test_data_loader

def load_dataset(file_path, batch_size, graph_type, dkt_graph_path=None,
                 train_ratio=None, val_ratio=None, shuffle=True,
                 model_type='GKT', use_binary=True, res_len=2, use_cuda=True):

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)

    if "split" not in df.columns:
        raise KeyError(f"Column 'split' not found in {file_path}. Run create_splits.py first.")

    # 1. Cleaning
    df.dropna(subset=['skill_id'], inplace=True)
    df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()

    # 2. Create Mappings (Internal, no JSON)
    # We add +1 to ensure Skill IDs start at 1 (0 is reserved for padding)
    df['skill'], _ = pd.factorize(df['skill_id'], sort=True)
    df['skill'] = df['skill'] + 1

    # 3. Create Features
    # Since skills start at 1, the minimum feature value will be > 0.
    if use_binary:
        # e.g., Skill 1 (Correct 0) -> 2, Skill 1 (Correct 1) -> 3
        df['skill_with_answer'] = df['skill'] * 2 + df['correct']
    else:
        df['skill_with_answer'] = df['skill'] * res_len + df['correct'] - 1

    # 4. Separate Data by Split
    train_feat, train_q, train_ans, train_mask = [], [], [], []
    val_feat, val_q, val_ans, val_mask = [], [], [], []
    test_feat, test_q, test_ans, test_mask = [], [], [], []

    grouped = df.groupby('user_id')

    # For Graph construction
    full_q_list = []
    full_len_list = []

    for uid, group in grouped:
        f = group['skill_with_answer'].values.tolist()
        q = group['skill'].values.tolist()
        a = group['correct'].values.tolist()
        splits = group['split'].values

        full_q_list.append(q)
        full_len_list.append(len(q))

        # --- Train Set (train_gnn + train_rnn) ---
        is_train = np.isin(splits, ['train_gnn', 'train_rnn'])
        if is_train.any():
            mask = np.zeros(len(f), dtype=bool)
            mask[is_train] = True
            train_feat.append(f); train_q.append(q); train_ans.append(a); train_mask.append(mask)

        # --- Validation Set ---
        is_val = (splits == 'val')
        if is_val.any():
            mask = np.zeros(len(f), dtype=bool)
            mask[is_val] = True
            val_feat.append(f); val_q.append(q); val_ans.append(a); val_mask.append(mask)

        # --- Test Set ---
        is_test = (splits == 'test')
        if is_test.any():
            mask = np.zeros(len(f), dtype=bool)
            mask[is_test] = True
            test_feat.append(f); test_q.append(q); test_ans.append(a); test_mask.append(mask)

    # 5. Build Datasets
    train_dataset = KTDataset(train_feat, train_q, train_ans, train_mask)
    val_dataset   = KTDataset(val_feat, val_q, val_ans, val_mask)
    test_dataset  = KTDataset(test_feat, test_q, test_ans, test_mask)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 6. Build Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

    # 7. Build Graph
    concept_num = int(df['skill'].max() + 1)
    graph = None

    if model_type == 'GKT':
        if graph_type == 'Dense':
            graph = build_dense_graph(concept_num)
        elif graph_type == 'Transition':
            # Use full history to build transition graph
            graph = build_transition_graph(full_q_list, full_len_list, range(len(full_q_list)), len(full_q_list), concept_num)
        elif graph_type == 'DKT':
            graph = build_dkt_graph(dkt_graph_path, concept_num)

        if use_cuda and graph is not None:
            graph = graph.cuda()

    return concept_num, graph, train_loader, valid_loader, test_loader


def build_transition_graph(question_list, seq_len_list, indices, student_num, concept_num):
    graph = np.zeros((concept_num, concept_num))
    student_dict = dict(zip(indices, np.arange(student_num)))
    for i in range(student_num):
        if i not in student_dict:
            continue
        questions = question_list[i]
        seq_len = seq_len_list[i]
        for j in range(seq_len - 1):
            pre = questions[j]
            next = questions[j + 1]
            graph[pre, next] += 1
    np.fill_diagonal(graph, 0)
    # row normalization
    rowsum = np.array(graph.sum(1))
    def inv(x):
        if x == 0:
            return x
        return 1. / x
    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)
    # covert to tensor
    graph = torch.from_numpy(graph).float()
    return graph


def build_dkt_graph(file_path, concept_num):
    graph = np.loadtxt(file_path)
    assert graph.shape[0] == concept_num and graph.shape[1] == concept_num
    graph = torch.from_numpy(graph).float()
    return graph