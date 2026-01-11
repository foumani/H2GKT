# # Code reused from https://github.com/jennyzhang0215/DKVMN.git
# import numpy as np
# import torch
# import math
# from sklearn import metrics
# from utils import model_isPid_type
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# transpose_data_model = {'akt'}
#
#
# def binaryEntropy(target, pred, mod="avg"):
#     loss = target * np.log(np.maximum(1e-10, pred)) + \
#         (1.0 - target) * np.log(np.maximum(1e-10, 1.0-pred))
#     if mod == 'avg':
#         return np.average(loss)*(-1.0)
#     elif mod == 'sum':
#         return - loss.sum()
#     else:
#         assert False
#
#
# def compute_auc(all_target, all_pred):
#     #fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
#     return metrics.roc_auc_score(all_target, all_pred)
#
#
# def compute_accuracy(all_target, all_pred):
#     all_pred[all_pred > 0.5] = 1.0
#     all_pred[all_pred <= 0.5] = 0.0
#     return metrics.accuracy_score(all_target, all_pred)
#
#
# def train(net, params,  optimizer,  q_data, qa_data, pid_data,  label):
#     net.train()
#     pid_flag, model_type = model_isPid_type(params.model)
#     N = int(math.ceil(len(q_data) / params.batch_size))
#     # for increased speed we move data to gpu hence these changes.
#     # q_data = q_data.T  # Shape: (200,3633)
#     # qa_data = qa_data.T  # Shape: (200,3633)
#     q_data = torch.from_numpy(q_data).long().to(device).t()  # Shape: [Seq, N]
#     qa_data = torch.from_numpy(qa_data).long().to(device).t()  # Shape: [Seq, N]
#     # Shuffle the data
#     # shuffled_ind = np.arange(q_data.shape[1])
#     # np.random.shuffle(shuffled_ind)
#     shuffled_ind = torch.randperm(q_data.shape[1], device=device)
#     # q_data = q_data[:, shuffled_ind]
#     # qa_data = qa_data[:, shuffled_ind]
#     q_data = q_data[:, shuffled_ind]
#     qa_data = qa_data[:, shuffled_ind]
#
#     if pid_flag:
#         # pid_data = pid_data.T
#         # pid_data = pid_data[:, shuffled_ind]
#         pid_data = torch.from_numpy(pid_data).long().to(device).t()
#         pid_data = pid_data[:, shuffled_ind]
#
#     pred_list = []
#     target_list = []
#
#     element_count = 0
#     true_el = 0
#     for idx in range(N):
#         optimizer.zero_grad()
#
#         # q_one_seq = q_data[:, idx*params.batch_size:(idx+1)*params.batch_size]
#         # if pid_flag:
#         #     pid_one_seq = pid_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
#         #
#         # qa_one_seq = qa_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
#         start_idx = idx * params.batch_size
#         end_idx = min((idx + 1) * params.batch_size, q_data.shape[1])
#
#         q_one_seq = q_data[:, start_idx:end_idx]
#         qa_one_seq = qa_data[:, start_idx:end_idx]
#         if pid_flag:
#             pid_one_seq = pid_data[:, start_idx:end_idx]
#
#         if model_type in transpose_data_model:
#             # input_q = np.transpose(q_one_seq[:, :])  # Shape (bs, seqlen)
#             # input_qa = np.transpose(qa_one_seq[:, :])  # Shape (bs, seqlen)
#             # target = np.transpose(qa_one_seq[:, :])
#             # if pid_flag:
#             #     # Shape (seqlen, batch_size)
#             #     input_pid = np.transpose(pid_one_seq[:, :])
#             input_q = q_one_seq.t()
#             input_qa = qa_one_seq.t()
#             target = qa_one_seq.t()
#             if pid_flag:
#                 input_pid = pid_one_seq.t()
#         else:
#             # input_q = (q_one_seq[:, :])  # Shape (seqlen, batch_size)
#             # input_qa = (qa_one_seq[:, :])  # Shape (seqlen, batch_size)
#             # target = (qa_one_seq[:, :])
#             # if pid_flag:
#             #     input_pid = (pid_one_seq[:, :])  # Shape (seqlen, batch_size)
#             input_q = q_one_seq
#             input_qa = qa_one_seq
#             target = qa_one_seq
#             if pid_flag:
#                 input_pid = pid_one_seq
#         target = (target - 1) / params.n_question
#         # target_1 = np.floor(target)
#         target_1 = torch.floor(target)
#         # el = np.sum(target_1 >= -.9)
#         el = (target_1 >= -0.9).sum().item()
#         element_count += el
#
#         # input_q = torch.from_numpy(input_q).long().to(device)
#         # input_qa = torch.from_numpy(input_qa).long().to(device)
#         # target = torch.from_numpy(target_1).float().to(device)
#         input_q = input_q.long()
#         input_qa = input_qa.long()
#         target_1 = target_1.float()
#         target = target_1.float()
#         if pid_flag:
#             # input_pid = torch.from_numpy(input_pid).long().to(device)
#             input_pid = input_pid.long()
#
#         if pid_flag:
#             loss, pred, true_ct = net(input_q, input_qa, target, input_pid)
#         else:
#             loss, pred, true_ct = net(input_q, input_qa, target)
#         # pred = pred.detach().cpu().numpy()  # (seqlen * batch_size, 1)
#         loss.backward()
#         true_el += true_ct.item()
#
#         if params.maxgradnorm > 0.:
#             torch.nn.utils.clip_grad_norm_(
#                 net.parameters(), max_norm=params.maxgradnorm)
#
#         optimizer.step()
#
#         # correct: 1.0; wrong 0.0; padding -1.0
#         # target = target_1.reshape((-1,))
#         # target = target_1.cpu().numpy().reshape((-1,))
#         #
#         #
#         # nopadding_index = np.flatnonzero(target >= -0.9)
#         # nopadding_index = nopadding_index.tolist()
#         # pred_nopadding = pred[nopadding_index]
#         # target_nopadding = target[nopadding_index]
#         #
#         # pred_list.append(pred_nopadding)
#         # target_list.append(target_nopadding)
#
#         target_flat = target_1.view(-1)
#         pred_flat = pred.view(-1)
#
#         # Create boolean mask (Fast GPU operation)
#         mask = target_flat >= -0.9
#
#         # Apply mask on GPU (Select only valid elements)
#         target_nopadding = target_flat[mask]
#         pred_nopadding = pred_flat[mask]
#
#         # 3. Move ONLY valid data to CPU
#         pred_list.append(pred_nopadding.detach().cpu().numpy())
#         target_list.append(target_nopadding.detach().cpu().numpy())
#
#     all_pred = np.concatenate(pred_list, axis=0)
#     all_target = np.concatenate(target_list, axis=0)
#
#     loss = binaryEntropy(all_target, all_pred)
#     auc = compute_auc(all_target, all_pred)
#     accuracy = compute_accuracy(all_target, all_pred)
#
#     # Calculate additional metrics
#     # Round predictions to 0 or 1 for classification metrics
#     binary_pred = (all_pred >= 0.5).astype(int)
#     precision = metrics.precision_score(all_target, binary_pred, zero_division=0)
#     recall = metrics.recall_score(all_target, binary_pred, zero_division=0)
#     f1 = metrics.f1_score(all_target, binary_pred, zero_division=0)
#
#     return loss, accuracy, auc, precision, recall, f1
#
#
# def test(net, params, optimizer, q_data, qa_data, pid_data, label, older_qa_data=None):
#     # dataArray: [ array([[],[],..])] Shape: (3633, 200)
#     pid_flag, model_type = model_isPid_type(params.model)
#     net.eval()
#     N = int(math.ceil(float(len(q_data)) / float(params.batch_size)))
#     # q_data = q_data.T  # Shape: (200,3633)
#     # qa_data = qa_data.T  # Shape: (200,3633)
#     q_data = torch.from_numpy(q_data).long().to(device).t()  # Shape: (SeqLen, N)
#     qa_data = torch.from_numpy(qa_data).long().to(device).t()  # Shape: (SeqLen, N)
#     if pid_flag:
#         # pid_data = pid_data.T
#         pid_data = torch.from_numpy(pid_data).long().to(device).t()
#     if older_qa_data is not None:
#         older_qa_data = older_qa_data.T  # Transpose to match q_data orientation (Seq, N)
#     seq_num = q_data.shape[1]
#     pred_list = []
#     target_list = []
#
#     count = 0
#     true_el = 0
#     element_count = 0
#     for idx in range(N):
#         # q_one_seq = q_data[:, idx*params.batch_size:(idx+1)*params.batch_size]
#         start_idx = idx * params.batch_size
#         end_idx = min((idx + 1) * params.batch_size, seq_num)  # GPU safe min
#         q_one_seq = q_data[:, start_idx:end_idx]
#
#         if pid_flag:
#             # pid_one_seq = pid_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
#             pid_one_seq = pid_data[:, start_idx:end_idx]
#         # input_q = q_one_seq[:, :]  # Shape (seqlen, batch_size)
#         # qa_one_seq = qa_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
#         qa_one_seq = qa_data[:, start_idx:end_idx]
#         # input_qa = qa_one_seq[:, :]  # Shape (seqlen, batch_size)
#
#         # print 'seq_num', seq_num
#         if model_type in transpose_data_model:
#             # # Shape (seqlen, batch_size)
#             # input_q = np.transpose(q_one_seq[:, :])
#             # # Shape (seqlen, batch_size)
#             # input_qa = np.transpose(qa_one_seq[:, :])
#             # target = np.transpose(qa_one_seq[:, :])
#             # if pid_flag:
#             #     input_pid = np.transpose(pid_one_seq[:, :])
#             input_q = q_one_seq.t()
#             input_qa = qa_one_seq.t()
#             target = qa_one_seq.t()
#             if pid_flag:
#                 input_pid = pid_one_seq.t()
#         else:
#             # input_q = (q_one_seq[:, :])  # Shape (seqlen, batch_size)
#             # input_qa = (qa_one_seq[:, :])  # Shape (seqlen, batch_size)
#             # target = (qa_one_seq[:, :])
#             # if pid_flag:
#             #     input_pid = (pid_one_seq[:, :])
#             input_q = q_one_seq
#             input_qa = qa_one_seq
#             target = qa_one_seq
#             if pid_flag:
#                 input_pid = pid_one_seq
#         target = (target - 1) / params.n_question
#         # target_1 = np.floor(target)
#         target_1 = torch.floor(target)
#         #target = np.random.randint(0,2, size = (target.shape[0],target.shape[1]))
#
#         # input_q = torch.from_numpy(input_q).long().to(device)
#         # input_qa = torch.from_numpy(input_qa).long().to(device)
#         # target = torch.from_numpy(target_1).float().to(device)
#         # if pid_flag:
#         #     input_pid = torch.from_numpy(input_pid).long().to(device)
#         input_q = input_q.long()
#         input_qa = input_qa.long()
#         target_1 = target_1.float()  # target used for loss/pred
#         target = target_1.float()
#         if pid_flag:
#             input_pid = input_pid.long()
#
#         with torch.no_grad():
#             if pid_flag:
#                 loss, pred, ct = net(input_q, input_qa, target, input_pid)
#             else:
#                 loss, pred, ct = net(input_q, input_qa, target)
#         pred = pred.cpu().numpy()  # (seqlen * batch_size, 1)
#         # true_el += ct.cpu().numpy()
#         #target = target.cpu().numpy()
#         true_el += ct.item()
#         if (idx + 1) * params.batch_size > seq_num:
#             real_batch_size = seq_num - idx * params.batch_size
#             count += real_batch_size
#         else:
#             count += params.batch_size
#
#         # correct: 1.0; wrong 0.0; padding -1.0
#         # target = target_1.reshape((-1,))
#         target = target_1.cpu().numpy().reshape((-1,))
#         nopadding_index = np.flatnonzero(target >= -0.9)
#         # --- NEW MASKING LOGIC ---
#         if older_qa_data is not None:
#             # 1. Get corresponding training batch
#             train_batch = older_qa_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
#             if model_type in transpose_data_model:
#                 train_batch = np.transpose(train_batch)
#
#                 # 2. Calculate length of training history (non-zero entries)
#             train_lens = np.sum(train_batch != 0, axis=1)
#
#             # 3. Create boolean mask (True = Validation Part, False = Training Part)
#             bs, sl = train_batch.shape
#             seq_idx = np.arange(sl)
#             # True if current index >= training length
#             future_mask = seq_idx[None, :] >= train_lens[:, None]
#             flat_future_mask = future_mask.reshape(-1)
#
#             # 4. Intersect with existing padding mask
#             # We only keep indices that are (Not Padding) AND (In Future)
#             valid_set = set(np.flatnonzero(flat_future_mask))
#             nopadding_index = [x for x in nopadding_index if x in valid_set]
#         else:
#             nopadding_index = nopadding_index.tolist()
#         pred_nopadding = pred[nopadding_index]
#         target_nopadding = target[nopadding_index]
#
#         element_count += pred_nopadding.shape[0]
#         # print avg_loss
#         pred_list.append(pred_nopadding)
#         target_list.append(target_nopadding)
#
#     assert count == seq_num, "Seq not matching"
#
#     all_pred = np.concatenate(pred_list, axis=0)
#     all_target = np.concatenate(target_list, axis=0)
#     loss = binaryEntropy(all_target, all_pred)
#     auc = compute_auc(all_target, all_pred)
#     accuracy = compute_accuracy(all_target, all_pred)
#
#     # Calculate additional metrics
#     # Round predictions to 0 or 1 for classification metrics
#     binary_pred = (all_pred >= 0.5).astype(int)
#     precision = metrics.precision_score(all_target, binary_pred, zero_division=0)
#     recall = metrics.recall_score(all_target, binary_pred, zero_division=0)
#     f1 = metrics.f1_score(all_target, binary_pred, zero_division=0)
#
#     return loss, accuracy, auc, precision, recall, f1


# Code reused from https://github.com/jennyzhang0215/DKVMN.git
import os
import time

import numpy as np
import torch
import math
from sklearn import metrics
from utils import model_isPid_type

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transpose_data_model = {'akt'}


def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log(np.maximum(1e-10, pred)) + (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    if mod == 'avg':
        return np.average(loss) * (-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False


def compute_auc(all_target, all_pred):
    # fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def train(net, params, optimizer, q_data, qa_data, pid_data, label):
    t = time.time()
    pid_flag, model_type = model_isPid_type(params.model)
    N = int(math.ceil(len(q_data) / params.batch_size))

    # --- GPU OPTIMIZATION START ---
    # Move entire dataset to GPU and Transpose immediately
    # Replaces: q_data = q_data.T
    q_data = torch.from_numpy(q_data).long().to(device).t()  # Shape: [Seq, N]
    qa_data = torch.from_numpy(qa_data).long().to(device).t()  # Shape: [Seq, N]

    # Shuffle on GPU (Fast)
    shuffled_ind = torch.randperm(q_data.shape[1], device=device)
    q_data = q_data[:, shuffled_ind]
    qa_data = qa_data[:, shuffled_ind]

    if pid_flag:
        pid_data = torch.from_numpy(pid_data).long().to(device).t()
        pid_data = pid_data[:, shuffled_ind]
    # --- GPU OPTIMIZATION END ---
    pred_list = []
    target_list = []

    element_count = 0
    true_el = 0
    for idx in range(N):
        optimizer.zero_grad()
        # --- GPU SLICING START ---
        start_idx = idx * params.batch_size
        end_idx = min((idx + 1) * params.batch_size, q_data.shape[1])

        q_one_seq = q_data[:, start_idx:end_idx]
        qa_one_seq = qa_data[:, start_idx:end_idx]
        if pid_flag:
            pid_one_seq = pid_data[:, start_idx:end_idx]
        # --- GPU SLICING END ---
        if model_type in transpose_data_model:
            input_q = q_one_seq.t()
            input_qa = qa_one_seq.t()
            target = qa_one_seq.t()
            if pid_flag:
                input_pid = pid_one_seq.t()
        else:
            input_q = q_one_seq
            input_qa = qa_one_seq
            target = qa_one_seq
            if pid_flag:
                input_pid = pid_one_seq
        target = (target - 1) / params.n_question
        target_1 = torch.floor(target)
        el = (target_1 >= -0.9).sum().item()
        element_count += el

        input_q = input_q.long()
        input_qa = input_qa.long()

        # --- FIX: Overwrite target with floored values ---
        target = target_1.float()
        if pid_flag:
            input_pid = input_pid.long()

        if pid_flag:
            # Pass correct target (floored)
            loss, pred, true_ct = net(input_q, input_qa, target, input_pid)
        else:
            loss, pred, true_ct = net(input_q, input_qa, target)

        # Keep pred on GPU for masking
        # pred = pred.detach().cpu().numpy() <-- REMOVED

        loss.backward()
        true_el += true_ct.item()

        if params.maxgradnorm > 0.:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=params.maxgradnorm)

        optimizer.step()
        # --- GPU MASKING OPTIMIZATION ---
        # Perform masking on GPU to avoid heavy CPU-NumPy sync
        target_flat = target_1.reshape(-1)  # FIX: Use reshape
        pred_flat = pred.reshape(-1)  # FIX: Use reshape

        mask = target_flat >= -0.9

        # Filter and THEN move to CPU
        target_nopadding = target_flat[mask].detach().cpu().numpy()
        pred_nopadding = pred_flat[mask].detach().cpu().numpy()
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)
        print(f"Time to run the batch: {time.time() - t} s")

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    # Calculate additional metrics
    binary_pred = (all_pred >= 0.5).astype(int)
    precision = metrics.precision_score(all_target, binary_pred, zero_division=0)
    recall = metrics.recall_score(all_target, binary_pred, zero_division=0)
    f1 = metrics.f1_score(all_target, binary_pred, zero_division=0)

    return loss, accuracy, auc, precision, recall, f1


def test(net, params, optimizer, q_data, qa_data, pid_data, label, older_qa_data=None):
    # dataArray: [ array([[],[],..])] Shape: (3633, 200)
    pid_flag, model_type = model_isPid_type(params.model)
    net.eval()
    N = int(math.ceil(float(len(q_data)) / float(params.batch_size)))

    # --- GPU OPTIMIZATION START ---
    q_data = torch.from_numpy(q_data).long().to(device).t()  # Shape: (SeqLen, N)
    qa_data = torch.from_numpy(qa_data).long().to(device).t()  # Shape: (SeqLen, N)
    if pid_flag:
        pid_data = torch.from_numpy(pid_data).long().to(device).t()

    # Keep older_qa_data as numpy for complex CPU masking logic if needed
    if older_qa_data is not None:
        older_qa_data = older_qa_data.T
        # --- GPU OPTIMIZATION END ---

    seq_num = q_data.shape[1]
    pred_list = []
    target_list = []

    count = 0
    true_el = 0
    element_count = 0
    for idx in range(N):
        # --- GPU SLICING START ---
        start_idx = idx * params.batch_size
        end_idx = min((idx + 1) * params.batch_size, seq_num)

        q_one_seq = q_data[:, start_idx:end_idx]
        qa_one_seq = qa_data[:, start_idx:end_idx]

        if pid_flag:
            pid_one_seq = pid_data[:, start_idx:end_idx]
        # --- GPU SLICING END ---

        # Inputs are already sliced
        input_q = q_one_seq
        input_qa = qa_one_seq

        if model_type in transpose_data_model:
            input_q = q_one_seq.t()
            input_qa = qa_one_seq.t()
            target = qa_one_seq.t()
            if pid_flag:
                input_pid = pid_one_seq.t()
        else:
            input_q = q_one_seq
            input_qa = qa_one_seq
            target = qa_one_seq
            if pid_flag:
                input_pid = pid_one_seq

        target = (target - 1) / params.n_question
        target_1 = torch.floor(target)

        input_q = input_q.long()
        input_qa = input_qa.long()

        # --- FIX: Overwrite target with floored values ---
        target = target_1.float()

        if pid_flag:
            input_pid = input_pid.long()

        with torch.no_grad():
            if pid_flag:
                loss, pred, ct = net(input_q, input_qa, target, input_pid)
            else:
                loss, pred, ct = net(input_q, input_qa, target)

        # Don't move pred to CPU yet
        true_el += ct.item()

        if (idx + 1) * params.batch_size > seq_num:
            real_batch_size = seq_num - idx * params.batch_size
            count += real_batch_size
        else:
            count += params.batch_size

        # --- MASKING LOGIC ---
        if older_qa_data is None:
            # FAST PATH: GPU Masking
            # target_flat = target_1.view(-1)
            # pred_flat = pred.view(-1)
            target_flat = target_1.reshape(-1)  # FIX: Use reshape
            pred_flat = pred.reshape(-1)  # FIX: Use reshape
            mask = target_flat >= -0.9

            pred_nopadding = pred_flat[mask].detach().cpu().numpy()
            target_nopadding = target_flat[mask].detach().cpu().numpy()

        else:
            # SLOW PATH: Legacy CPU Masking (for older_qa_data logic)
            pred_cpu = pred.cpu().numpy()
            target_cpu = target_1.cpu().numpy().reshape((-1,))
            nopadding_index = np.flatnonzero(target_cpu >= -0.9)

            # 1. Get corresponding training batch (numpy slicing)
            train_batch = older_qa_data[:, start_idx:end_idx]
            if model_type in transpose_data_model:
                train_batch = np.transpose(train_batch)

            # 2. Calculate length of training history
            train_lens = np.sum(train_batch != 0, axis=1)

            # 3. Create boolean mask
            bs, sl = train_batch.shape
            seq_idx = np.arange(sl)
            future_mask = seq_idx[None, :] >= train_lens[:, None]
            flat_future_mask = future_mask.reshape(-1)

            # 4. Intersect
            valid_set = set(np.flatnonzero(flat_future_mask))
            nopadding_index = [x for x in nopadding_index if x in valid_set]

            pred_nopadding = pred_cpu[nopadding_index]
            target_nopadding = target_cpu[nopadding_index]

        element_count += pred_nopadding.shape[0]
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    assert count == seq_num, "Seq not matching"

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    # Calculate additional metrics
    binary_pred = (all_pred >= 0.5).astype(int)
    precision = metrics.precision_score(all_target, binary_pred, zero_division=0)
    recall = metrics.recall_score(all_target, binary_pred, zero_division=0)
    f1 = metrics.f1_score(all_target, binary_pred, zero_division=0)

    return loss, accuracy, auc, precision, recall, f1