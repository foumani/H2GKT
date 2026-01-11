import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from utils import nll_gaussian, kl_categorical, kl_categorical_uniform, accuracy

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com


import torch
import torch.nn as nn
# !!! ENSURE THESE ARE IMPORTED !!!
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np


class KTLoss(nn.Module):

    def __init__(self):
        super(KTLoss, self).__init__()

    def forward(self, pred_answers, real_answers, training=False):
        # 1. Masking (Though inputs from your train.py are likely already clean)
        answer_mask = torch.ne(real_answers, -1)

        # 2. Clone predictions to avoid modifying the original tensor used for Log later
        pred_one = pred_answers.clone()
        pred_zero = 1.0 - pred_answers.clone()

        auc, acc, prec, rec, f1 = -1, -1, -1, -1, -1

        if not training:
            try:
                # Move to CPU/Numpy
                y_true = real_answers[answer_mask].cpu().detach().numpy()
                y_pred_prob = pred_one[answer_mask].cpu().detach().numpy()

                # Check for NaNs (Common cause of failure)
                if np.isnan(y_pred_prob).any():
                    print("[Metric Error] Predictions contain NaNs!")
                    raise ValueError("NaNs in predictions")

                # Check for "Only One Class" (Common in small batches)
                if len(np.unique(y_true)) < 2:
                    # This is normal for small batches, but shouldn't happen for ALL batches
                    # We skip metrics for this specific batch
                    pass
                else:
                    # Calculate Metrics
                    y_pred_bin = (y_pred_prob >= 0.5).astype(int)

                    auc = roc_auc_score(y_true, y_pred_prob)
                    acc = (y_true == y_pred_bin).mean()
                    prec = precision_score(y_true, y_pred_bin, zero_division=0)
                    rec = recall_score(y_true, y_pred_bin, zero_division=0)
                    f1 = f1_score(y_true, y_pred_bin, zero_division=0)

            except Exception as e:
                # !!! THIS PRINT WILL REVEAL THE MYSTERY !!!
                print(f"[Metric Error]: {e}")
                auc, acc, prec, rec, f1 = -1, -1, -1, -1, -1

        # 3. NLL Loss Calculation
        # Add epsilon to prevent log(0) -> NaN
        pred_one[answer_mask] = torch.log(pred_one[answer_mask] + 1e-6)
        pred_zero[answer_mask] = torch.log(pred_zero[answer_mask] + 1e-6)

        pred_answers_log = torch.cat(
            (pred_zero.unsqueeze(dim=1), pred_one.unsqueeze(dim=1)), dim=1)

        nll_loss = nn.NLLLoss(ignore_index=-1)
        loss = nll_loss(pred_answers_log, real_answers.long())

        return loss, auc, acc, prec, rec, f1


class VAELoss(nn.Module):

    def __init__(self, concept_num, edge_type_num=2, prior=False, var=5e-5):
        super(VAELoss, self).__init__()
        self.concept_num = concept_num
        self.edge_type_num = edge_type_num
        self.prior = prior
        self.var = var

    def forward(self, ec_list, rec_list, z_prob_list, log_prior=None):
        time_stamp_num = len(ec_list)
        loss = 0
        for time_idx in range(time_stamp_num):
            output = rec_list[time_idx]
            target = ec_list[time_idx]
            prob = z_prob_list[time_idx]
            loss_nll = nll_gaussian(output, target, self.var)
            if self.prior:
                assert log_prior is not None
                loss_kl = kl_categorical(prob, log_prior, self.concept_num)
            else:
                loss_kl = kl_categorical_uniform(prob, self.concept_num, self.edge_type_num)
            if time_idx == 0:
                loss = loss_nll + loss_kl
            else:
                loss = loss + loss_nll + loss_kl
        return loss / time_stamp_num



