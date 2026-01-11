import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader, Dataset


def compute_metrics(y_true, y_prob, t=0.5):
    """Computes generic classification metrics."""
    y_pred = (y_prob >= t).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0
    return acc, prec, rec, f1, auc


class DictTensorDataset(Dataset):
    """
    Wraps a dictionary of tensors.
    __getitem__ returns a dictionary of values for that index.
    DataLoader will automatically collate these into a dictionary of batched tensors.
    """

    def __init__(self, data_dict):
        self.data_dict = data_dict
        # Get keys safely (assuming all tensors have same length)
        self.keys = list(data_dict.keys())
        self.length = len(data_dict[self.keys[0]])

    def __getitem__(self, index):
        return {k: self.data_dict[k][index] for k in self.keys}

    def __len__(self):
        return self.length


def create_dataloader(seq_data, batch_size=32, shuffle=True):
    """Wraps the dictionary of tensors into a PyTorch DataLoader."""
    if seq_data is None:
        return None

    # Use the custom DictTensorDataset instead of TensorDataset
    dataset = DictTensorDataset(seq_data)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# def create_dataloader(seq_data, batch_size=32, shuffle=True):
#     """Wraps the dictionary of tensors into a PyTorch DataLoader."""
#     if seq_data is None:
#         return None
#
#     # Ensure strict order matches the unpacking in run.py
#     dataset = TensorDataset(
#         seq_data['student_id'],
#         seq_data['problem_seq'],
#         seq_data['skill_seq'],
#         seq_data['class_seq'],
#         seq_data['teacher_seq'],
#         seq_data['correct_seq'],
#         seq_data['mask'],
#         seq_data['eval_mask']
#     )
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)