import os
import numpy as np
import time
import random
import argparse
import pickle
import gc
import datetime
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from create_splits import create_splits
from models import GKT, MultiHeadAttention, VAE, DKT
from metrics import KTLoss, VAELoss
from processing import load_dataset

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com
# Edited by: foumani
# Email: arashfoumani@gmail.com


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--data-dir', type=str, default='data',
                    help='Data dir for loading input data.')
parser.add_argument('--data-file', type=str, default='assistment_test15.csv',
                    help='Name of input data file.')
parser.add_argument('--save-dir', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('-graph-save-dir', type=str, default='graphs',
                    help='Dir for saving concept graphs.')
parser.add_argument('--load-dir', type=str, default='',
                    help='Where to load the trained model if finetunning. ' + 'Leave empty to train from scratch')
parser.add_argument('--dkt-graph-dir', type=str, default='dkt-graph',
                    help='Where to load the pretrained dkt graph.')
parser.add_argument('--dkt-graph', type=str, default='dkt_graph.txt',
                    help='DKT graph data file name.')
parser.add_argument('--model', type=str, default='GKT',
                    help='Model type to use, support GKT and DKT.')
parser.add_argument('--hid-dim', type=int, default=32,
                    help='Dimension of hidden knowledge states.')
parser.add_argument('--emb-dim', type=int, default=32,
                    help='Dimension of concept embedding.')
parser.add_argument('--attn-dim', type=int, default=32,
                    help='Dimension of multi-head attention layers.')
parser.add_argument('--vae-encoder-dim', type=int, default=32,
                    help='Dimension of hidden layers in vae encoder.')
parser.add_argument('--vae-decoder-dim', type=int, default=32,
                    help='Dimension of hidden layers in vae decoder.')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--graph-type', type=str, default='Dense',
                    help='The type of latent concept graph.')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--bias', type=bool, default=True,
                    help='Whether to add bias for neural network layers.')
parser.add_argument('--binary', type=bool, default=True,
                    help='Whether only use 0/1 for results.')
parser.add_argument('--result-type', type=int, default=12,
                    help='Number of results types when multiple results are used.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')
parser.add_argument('--var', type=float, default=1, help='Output variance.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--train-ratio', type=float, default=0.6,
                    help='The ratio of training samples in a dataset.')
parser.add_argument('--val-ratio', type=float, default=0.2,
                    help='The ratio of validation samples in a dataset.')
parser.add_argument('--shuffle', type=bool, default=True,
                    help='Whether to shuffle the dataset or not.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')
parser.add_argument('--test', type=bool, default=False,
                    help='Whether to test for existed model.')
parser.add_argument('--test-model-dir', type=str, default='logs/expDKT',
                    help='Existed model file dir.')
parser.add_argument('--accumulation-steps', type=int, default=4,
                    help='Number of steps to accumulate gradients before updating weights.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

res_len = 2 if args.binary else args.result_type

# Save model and meta-data. Always saves in a new sub-folder.
log = None
save_dir = args.save_dir
if args.save_dir:
    exp_counter = 0
    now = datetime.datetime.now()
    # timestamp = now.isoformat()
    timestamp = now.strftime('%Y-%m-%d %H-%M-%S')
    if args.model == 'DKT':
        model_file_name = 'DKT'
    elif args.model == 'GKT':
        model_file_name = 'GKT' + '-' + args.graph_type
    else:
        raise NotImplementedError(args.model + ' model is not implemented!')
    save_dir = '{}/exp{}/'.format(args.save_dir, model_file_name + timestamp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    meta_file = os.path.join(save_dir, 'metadata.pkl')
    model_file = os.path.join(save_dir, model_file_name + '.pt')
    optimizer_file = os.path.join(save_dir, model_file_name + '-Optimizer.pt')
    scheduler_file = os.path.join(save_dir, model_file_name + '-Scheduler.pt')
    log_file = os.path.join(save_dir, 'log.txt')
    log = open(log_file, 'w')
    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print(
        "WARNING: No save_dir provided!" + "Testing (within this script) will throw an error.")

# load dataset
# dataset_path = os.path.join(args.data_dir, args.data_file)
# dkt_graph_path = os.path.join(args.dkt_graph_dir, args.dkt_graph)
# if not os.path.exists(dkt_graph_path):
#     dkt_graph_path = None
# concept_num, graph, train_loader, valid_loader, test_loader = load_dataset(dataset_path,
#                                                                            args.batch_size,
#                                                                            args.graph_type,
#                                                                            dkt_graph_path=dkt_graph_path,
#                                                                            train_ratio=args.train_ratio,
#                                                                            val_ratio=args.val_ratio,
#                                                                            shuffle=args.shuffle,
#                                                                            model_type=args.model,
#                                                                            use_cuda=args.cuda)

# --- LOGIC TO ENSURE SPLITS EXIST ---
dataset_path = os.path.join(args.data_dir, args.data_file)
dataset_name = os.path.splitext(args.data_file)[0]
# Handle naming heuristics if needed
if "2009" in dataset_name: dataset_name = "assist2009"
elif "2012" in dataset_name: dataset_name = "assist2012"

# Define the target split file path
split_file_path = os.path.join(args.data_dir, f"{dataset_name}_split.csv")

if not os.path.exists(split_file_path):
    print(f"Split file not found at {split_file_path}. Generating from {dataset_path}...")
    create_splits(dataset_path, split_file_path)
else:
    print(f"Found existing split file: {split_file_path}")

# Point dkt_graph_path correctly
dkt_graph_path = os.path.join(args.dkt_graph_dir, args.dkt_graph)
if not os.path.exists(dkt_graph_path):
    dkt_graph_path = None

# LOAD THE SPLIT FILE, NOT THE RAW FILE
concept_num, graph, train_loader, valid_loader, test_loader = load_dataset(
    split_file_path,  # Pass the _split.csv file
    args.batch_size,
    args.graph_type,
    dkt_graph_path=dkt_graph_path,
    train_ratio=args.train_ratio,
    val_ratio=args.val_ratio,
    shuffle=args.shuffle,
    model_type=args.model,
    use_cuda=args.cuda
)

# build models
graph_model = None
if args.model == 'GKT':
    if args.graph_type == 'MHA':
        graph_model = MultiHeadAttention(args.edge_types, concept_num, args.emb_dim,
                                         args.attn_dim, dropout=args.dropout)
    elif args.graph_type == 'VAE':
        graph_model = VAE(args.emb_dim, args.vae_encoder_dim, args.edge_types,
                          args.vae_decoder_dim, args.vae_decoder_dim, concept_num,
                          edge_type_num=args.edge_types, tau=args.temp,
                          factor=args.factor, dropout=args.dropout, bias=args.bias)
        vae_loss = VAELoss(concept_num, edge_type_num=args.edge_types, prior=args.prior,
                           var=args.var)
        if args.cuda:
            vae_loss = vae_loss.cuda()
    if args.cuda and args.graph_type in ['MHA', 'VAE']:
        graph_model = graph_model.cuda()
    model = GKT(concept_num, args.hid_dim, args.emb_dim, args.edge_types, args.graph_type,
                graph=graph, graph_model=graph_model,
                dropout=args.dropout, bias=args.bias, has_cuda=args.cuda)
elif args.model == 'DKT':
    model = DKT(res_len * concept_num, args.emb_dim, concept_num, dropout=args.dropout,
                bias=args.bias)
else:
    raise NotImplementedError(args.model + ' model is not implemented!')
kt_loss = KTLoss()

# build optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)

# load model/optimizer/scheduler params
if args.load_dir:
    if args.model == 'DKT':
        model_file_name = 'DKT'
    elif args.model == 'GKT':
        model_file_name = 'GKT' + '-' + args.graph_type
    else:
        raise NotImplementedError(args.model + ' model is not implemented!')
    model_file = os.path.join(args.load_dir, model_file_name + '.pt')
    optimizer_file = os.path.join(save_dir, model_file_name + '-Optimizer.pt')
    scheduler_file = os.path.join(save_dir, model_file_name + '-Scheduler.pt')
    model.load_state_dict(torch.load(model_file))
    optimizer.load_state_dict(torch.load(optimizer_file))
    scheduler.load_state_dict(torch.load(scheduler_file))
    args.save_dir = False

# build optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)

if args.model == 'GKT' and args.prior:
    prior = np.array([0.91, 0.03, 0.03, 0.03])  # TODO: hard coded for now
    print("Using prior")
    print(prior)
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)
    if args.cuda:
        log_prior = log_prior.cuda()

if args.cuda:
    model = model.cuda()
    kt_loss = KTLoss()


def train(epoch, best_val_loss):
    t = time.time()
    loss_train = []
    kt_train = []
    vae_train = []

    # Renamed to avoid collision with VAE output
    auc_train = []
    acc_train = []
    precision_train = []
    recall_train = []
    f1_score_train = []

    if graph_model is not None:
        graph_model.train()
    model.train()
    optimizer.zero_grad()
    for batch_idx, (features, questions, answers, mask) in enumerate(train_loader):
        t1 = time.time()
        if args.cuda:
            features, questions, answers, mask = features.cuda(), questions.cuda(), answers.cuda(), mask.cuda()

        # NOTE: 'rec_list' here shadows the recall list if we aren't careful!
        # ec_list, rec_list (reconstruction), z_prob_list
        ec_list, rec_list_vae, z_prob_list = None, None, None

        if args.model == 'GKT':
            pred_res, ec_list, rec_list_vae, z_prob_list = model(features, questions)
        elif args.model == 'DKT':
            pred_res = model(features, questions)
        else:
            raise NotImplementedError(args.model + ' model is not implemented!')

        # ===========================================
        seq_len = pred_res.shape[1]
        target_answers = answers[:, -seq_len:]  # Take last 'seq_len' items
        target_mask = mask[:, -seq_len:]  # Take last 'seq_len' items

        # 2. Flatten and Apply Mask
        mask_flat = target_mask.reshape(-1)

        if mask_flat.sum() == 0:
            loss_kt = torch.tensor(0.0, requires_grad=True).to(pred_res.device)
            auc, acc, prec, rec, f1 = -1, -1, -1, -1, -1
        else:
            pred_flat = pred_res.reshape(-1)[mask_flat]
            ans_flat = target_answers.reshape(-1)[mask_flat]

            # Using your KTLoss signature which returns 6 values
            loss_kt, auc, acc, prec, rec, f1 = kt_loss(pred_flat, ans_flat, training=True)
        # ===========================================

        kt_train.append(float(loss_kt.cpu().detach().numpy()))

        # Log metrics if they were calculated (auc != -1)
        if auc != -1:
            auc_train.append(auc)
            acc_train.append(acc)
            precision_train.append(prec)
            recall_train.append(rec)
            f1_score_train.append(f1)

        if args.model == 'GKT' and args.graph_type == 'VAE':
            if args.prior:
                loss_vae = vae_loss(ec_list, rec_list_vae, z_prob_list,
                                    log_prior=log_prior)
            else:
                loss_vae = vae_loss(ec_list, rec_list_vae, z_prob_list)
                vae_train.append(float(loss_vae.cpu().detach().numpy()))
            print(
                f'batch idx: {batch_idx} loss kt: {loss_kt.item():.4f} loss vae: {loss_vae.item():.4f} auc: {auc:.4f} acc: {acc:.4f} f1: {f1:.4f}',
                end=' ')
            loss = loss_kt + loss_vae
        else:
            loss = loss_kt
            print(
                f'batch idx: {batch_idx} loss kt: {loss_kt.item():.4f} auc: {auc:.4f} acc: {acc:.4f} f1: {f1:.4f}',
                end=' ')
        loss_train.append(float(loss.cpu().detach().numpy()))

        # Accumulation
        loss = loss / args.accumulation_steps
        loss.backward()

        if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(
                train_loader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        del loss
        print('cost time: ', str(time.time() - t1))

    loss_val = []
    kt_val = []
    vae_val = []

    # Renamed lists to avoid shadowing
    auc_list, acc_list, precision_list, recall_list, f1_list = [], [], [], [], []

    if graph_model is not None:
        graph_model.eval()
    model.eval()
    with torch.no_grad():
        for batch_idx, (features, questions, answers, mask) in enumerate(valid_loader):
            if args.cuda:
                features, questions, answers, mask = features.cuda(), questions.cuda(), answers.cuda(), mask.cuda()

            # NOTE: 'rec_list_vae' used here instead of 'rec_list'
            ec_list, rec_list_vae, z_prob_list = None, None, None

            if args.model == 'GKT':
                pred_res, ec_list, rec_list_vae, z_prob_list = model(features, questions)
            elif args.model == 'DKT':
                pred_res = model(features, questions)
            else:
                raise NotImplementedError(args.model + ' model is not implemented!')

            # ===========================================
            seq_len = pred_res.shape[1]
            target_answers = answers[:, -seq_len:]
            target_mask = mask[:, -seq_len:]

            mask_flat = target_mask.reshape(-1)

            if mask_flat.sum() == 0:
                loss_kt = torch.tensor(0.0, requires_grad=True).to(pred_res.device)
                auc, acc, prec, rec, f1 = -1, -1, -1, -1, -1
            else:
                pred_flat = pred_res.reshape(-1)[mask_flat]
                ans_flat = target_answers.reshape(-1)[mask_flat]
                loss_kt, auc, acc, prec, rec, f1 = kt_loss(pred_flat, ans_flat,
                                                           training=False)
            # ===========================================

            loss_kt = float(loss_kt.cpu().detach().numpy())
            kt_val.append(loss_kt)

            if auc != -1:
                auc_list.append(auc)
                acc_list.append(acc)
                precision_list.append(prec)
                recall_list.append(rec)
                f1_list.append(f1)

            loss = loss_kt
            if args.model == 'GKT' and args.graph_type == 'VAE':
                loss_vae = vae_loss(ec_list, rec_list_vae, z_prob_list)
                loss_vae = float(loss_vae.cpu().detach().numpy())
                vae_val.append(loss_vae)
                loss = loss_kt + loss_vae
            loss_val.append(loss)
            del loss

    # Helper for printing
    def safe_mean(l):
        return np.mean(l) if len(l) > 0 else 0.0

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(safe_mean(loss_train)),
          'auc_train: {:.4f}'.format(safe_mean(auc_train)),
          'acc_train: {:.4f}'.format(safe_mean(acc_train)),
          'f1_train: {:.4f}'.format(safe_mean(f1_score_train)),
          'loss_val: {:.4f}'.format(safe_mean(loss_val)),
          'auc_val: {:.4f}'.format(safe_mean(auc_list)),
          'acc_val: {:.4f}'.format(safe_mean(acc_list)),
          'prec_val: {:.4f}'.format(safe_mean(precision_list)),
          'rec_val: {:.4f}'.format(safe_mean(recall_list)),
          'f1_val: {:.4f}'.format(safe_mean(f1_list)),
          'time: {:.4f}s'.format(time.time() - t))

    if args.save_dir and safe_mean(loss_val) < best_val_loss:
        print('Best model so far, saving...')
        torch.save(model.state_dict(), model_file)
        torch.save(optimizer.state_dict(), optimizer_file)
        torch.save(scheduler.state_dict(), scheduler_file)
        # Log writing
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.4f}'.format(safe_mean(loss_train)),
              'auc_train: {:.4f}'.format(safe_mean(auc_train)),
              'loss_val: {:.4f}'.format(safe_mean(loss_val)),
              'auc_val: {:.4f}'.format(safe_mean(auc_list)),
              'acc_val: {:.4f}'.format(safe_mean(acc_list)),
              'f1_val: {:.4f}'.format(safe_mean(f1_list)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()

    res = safe_mean(loss_val)
    gc.collect()
    if args.cuda:
        torch.cuda.empty_cache()
    return res


def test():
    loss_test = []
    kt_test = []
    vae_test = []
    auc_test, acc_test, precision_test, recall_test, f1_test = [], [], [], [], []

    if graph_model is not None:
        graph_model.eval()
    model.eval()
    model.load_state_dict(torch.load(model_file))
    with torch.no_grad():
        for batch_idx, (features, questions, answers, mask) in enumerate(test_loader):
            if args.cuda:
                features, questions, answers, mask = features.cuda(), questions.cuda(), answers.cuda(), mask.cuda()

            ec_list, rec_list_vae, z_prob_list = None, None, None

            if args.model == 'GKT':
                pred_res, ec_list, rec_list_vae, z_prob_list = model(features, questions)
            elif args.model == 'DKT':
                pred_res = model(features, questions)
            else:
                raise NotImplementedError(args.model + ' model is not implemented!')

            # ===========================================
            seq_len = pred_res.shape[1]
            target_answers = answers[:, -seq_len:]
            target_mask = mask[:, -seq_len:]

            mask_flat = target_mask.reshape(-1)

            if mask_flat.sum() == 0:
                loss_kt = torch.tensor(0.0, requires_grad=True).to(pred_res.device)
                auc, acc, prec, rec, f1 = -1, -1, -1, -1, -1
            else:
                pred_flat = pred_res.reshape(-1)[mask_flat]
                ans_flat = target_answers.reshape(-1)[mask_flat]
                loss_kt, auc, acc, prec, rec, f1 = kt_loss(pred_flat, ans_flat,
                                                           training=False)
            # ===========================================

            loss_kt = float(loss_kt.cpu().detach().numpy())
            if auc != -1:
                auc_test.append(auc)
                acc_test.append(acc)
                precision_test.append(prec)
                recall_test.append(rec)
                f1_test.append(f1)

            kt_test.append(loss_kt)
            loss = loss_kt
            if args.model == 'GKT' and args.graph_type == 'VAE':
                loss_vae = vae_loss(ec_list, rec_list_vae, z_prob_list)
                loss_vae = float(loss_vae.cpu().detach().numpy())
                vae_test.append(loss_vae)
                loss = loss_kt + loss_vae
            loss_test.append(loss)
            del loss

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')

    def safe_mean(l):
        return np.mean(l) if len(l) > 0 else 0.0

    print('loss_test: {:.4f}'.format(safe_mean(loss_test)),
          'auc_test: {:.4f}'.format(safe_mean(auc_test)),
          'acc_test: {:.4f}'.format(safe_mean(acc_test)),
          'prec_test: {:.4f}'.format(safe_mean(precision_test)),
          'rec_test: {:.4f}'.format(safe_mean(recall_test)),
          'f1_test: {:.4f}'.format(safe_mean(f1_test)))

    if args.save_dir:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        print('loss_test: {:.4f}'.format(safe_mean(loss_test)),
              'auc_test: {:.4f}'.format(safe_mean(auc_test)),
              'acc_test: {:.4f}'.format(safe_mean(acc_test)),
              'f1_test: {:.4f}'.format(safe_mean(f1_test)), file=log)
        log.flush()

    gc.collect()
    if args.cuda:
        torch.cuda.empty_cache()


if args.test is False:
    # Train model
    print('start training!')
    t_total = time.time()
    best_val_loss = np.inf
    best_epoch = 0
    for epoch in range(args.epochs):
        val_loss = train(epoch, best_val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
    print("Optimization Finished!")
    print("Best Epoch: {:04d}".format(best_epoch))
    if args.save_dir:
        print("Best Epoch: {:04d}".format(best_epoch), file=log)
        log.flush()

test()
if log is not None:
    print(save_dir)
    log.close()