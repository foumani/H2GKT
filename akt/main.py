import sys
import os
import os.path
import glob
import logging
import argparse
import numpy as np
import torch
from load_data import DATA, PID_DATA
from run import train, test
from utils import try_makedirs, load_model, get_file_name_identifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# assert torch.cuda.is_available(), "No Cuda available, AssertionError"


def train_one_dataset(params, file_name, train_q_data, train_qa_data, train_pid, valid_q_data, valid_qa_data, valid_pid):
    # ================================== model initialization ==================================

    model = load_model(params)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, betas=(0.9, 0.999), eps=1e-8)

    print("\n")

    # ================================== start training ==================================
    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}
    best_valid_auc = 0

    for idx in range(params.max_iter):

        train_loss, train_accuracy, train_auc, train_prec, train_rec, train_f1 = train(
            model, params, optimizer, train_q_data, train_qa_data, train_pid,
            label='Train')

        # Validation step
        valid_loss, valid_accuracy, valid_auc, valid_prec, valid_rec, valid_f1 = test(
            model, params, optimizer, valid_q_data, valid_qa_data, valid_pid,
            label='Valid', older_qa_data=train_qa_data)

        # formatted print
        print(f"Epoch {idx + 1} " + "-" * 100)
        print(f"[Train] Loss: {train_loss:.4f} | ACC: {train_accuracy:.4f} | PREC: {train_prec:.4f} | REC: {train_rec:.4f} | F1: {train_f1:.4f} | AUC: {train_auc}")
        print(f"[Val  ] Loss: {valid_loss:.4f} | ACC: {valid_accuracy:.4f} | PREC: {valid_prec:.4f} | REC: {valid_rec:.4f} | F1: {valid_f1:.4f} | AUC: {valid_auc}")

        try_makedirs('model')
        try_makedirs(os.path.join('model', params.model))
        try_makedirs(os.path.join('model', params.model, params.save))

        all_valid_auc[idx + 1] = valid_auc
        all_train_auc[idx + 1] = train_auc
        all_valid_loss[idx + 1] = valid_loss
        all_train_loss[idx + 1] = train_loss
        all_valid_accuracy[idx + 1] = valid_accuracy
        all_train_accuracy[idx + 1] = train_accuracy

        # output the epoch with the best validation auc
        if valid_auc > best_valid_auc:
            path = os.path.join('model', params.model, params.save,  file_name) + '_*'
            for i in glob.glob(path):
                os.remove(i)
            best_valid_auc = valid_auc
            best_epoch = idx+1
            torch.save({'epoch': idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        },
                       os.path.join('model', params.model, params.save, file_name)+'_' + str(idx+1))
        if idx-best_epoch > 40:
            break   

    try_makedirs('result')
    try_makedirs(os.path.join('result', params.model))
    try_makedirs(os.path.join('result', params.model, params.save))
    f_save_log = open(os.path.join('result', params.model, params.save, file_name), 'w')
    f_save_log.write("valid_auc:\n" + str(all_valid_auc) + "\n\n")
    f_save_log.write("train_auc:\n" + str(all_train_auc) + "\n\n")
    f_save_log.write("valid_loss:\n" + str(all_valid_loss) + "\n\n")
    f_save_log.write("train_loss:\n" + str(all_train_loss) + "\n\n")
    f_save_log.write("valid_accuracy:\n" + str(all_valid_accuracy) + "\n\n")
    f_save_log.write("train_accuracy:\n" + str(all_train_accuracy) + "\n\n")
    f_save_log.close()
    return best_epoch


def test_one_dataset(params, file_name, test_q_data, test_qa_data, test_pid, best_epoch, qa_data):
    print("\n\nStart testing ......................\n Best epoch:", best_epoch)
    model = load_model(params)

    checkpoint = torch.load(os.path.join('model', params.model, params.save, file_name) + '_'+str(best_epoch))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_accuracy, test_auc, test_prec, test_rec, test_f1 = test(
        model, params, None, test_q_data, test_qa_data, test_pid, label='Test',
        older_qa_data=qa_data)

    print("-" * 100)
    print(f"[Test ] Loss: {test_loss:.4f} | ACC: {test_accuracy:.4f} | PREC: {test_prec:.4f} | REC: {test_rec:.4f} | F1: {test_f1:.4f} | AUC: {test_auc}")

    # Now Delete all the models
    path = os.path.join('model', params.model, params.save,  file_name) + '_*'
    for i in glob.glob(path):
        os.remove(i)


if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Script to test KT')
    # Basic Parameters
    parser.add_argument('--max_iter', type=int, default=100, help='number of iterations')
    parser.add_argument('--train_set', type=str, default='1')
    parser.add_argument('--seed', type=int, default=224, help='default seed')

    # Common parameters
    parser.add_argument('--optim', type=str, default='adam',
                        help='Default Optimizer')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='the batch size')
    parser.add_argument('--lr', type=float, default=5e-6,
                        help='learning rate')
    parser.add_argument('--maxgradnorm', type=float,
                        default=-1, help='maximum gradient norm')
    parser.add_argument('--final_fc_dim', type=int, default=512,
                        help='hidden state dim for final fc layer')

    # AKT Specific Parameter
    parser.add_argument('--d_model', type=int, default=256,
                        help='Transformer d_model shape')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='Transformer d_ff shape')
    parser.add_argument('--dropout', type=float,
                        default=0.05, help='Dropout rate')
    parser.add_argument('--n_block', type=int, default=1,
                        help='number of blocks')
    parser.add_argument('--n_head', type=int, default=8,
                        help='number of heads in multihead attention')
    parser.add_argument('--kq_same', type=int, default=1)

    # AKT-R Specific Parameter
    parser.add_argument('--l2', type=float,
                        default=1e-5, help='l2 penalty for difficulty')

    # DKVMN Specific  Parameter
    parser.add_argument('--q_embed_dim', type=int, default=50,
                        help='question embedding dimensions')
    parser.add_argument('--qa_embed_dim', type=int, default=256,
                        help='answer and question embedding dimensions')
    parser.add_argument('--memory_size', type=int,
                        default=50, help='memory size')
    parser.add_argument('--init_std', type=float, default=0.1,
                        help='weight initialization std')
    # DKT Specific Parameter
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--lamda_r', type=float, default=0.1)
    parser.add_argument('--lamda_w1', type=float, default=0.1)
    parser.add_argument('--lamda_w2', type=float, default=0.1)

    # Datasets and Model
    parser.add_argument('--model', type=str, default='akt_pid',
                        help="combination of akt/sakt/dkvmn/dkt (mandatory), pid/cid (mandatory) separated by underscore '_'. For example tf_pid")
    parser.add_argument('--dataset', type=str, default="assist2009_pid")

    params = parser.parse_args()
    dataset = params.dataset

    if dataset in {"assist2009"}:
        params.n_question = 123 + 1
        params.batch_size = 128
        params.seqlen = 200
        params.data_dir = '../data/' + dataset
        params.data_name = dataset
        params.n_pid = 26688 + 1

    if dataset in {"assist2012"}:
        params.n_question = 265 + 1
        params.batch_size = 128
        params.seqlen = 200
        params.data_dir = '../data/' + dataset
        params.data_name = dataset
        params.n_pid = 179999 + 1

    # if dataset in {"assist2017_pid"}:
    #     params.batch_size = 24
    #     params.seqlen = 200
    #     params.data_dir = 'data/'+dataset
    #     params.data_name = dataset
    #     params.n_question = 102
    #     params.n_pid = 3162
    #
    # if dataset in {"assist2015"}:
    #     params.n_question = 100
    #     params.batch_size = 24
    #     params.seqlen = 200
    #     params.data_dir = 'data/'+dataset
    #     params.data_name = dataset
    #
    # if dataset in {"statics"}:
    #     params.n_question = 1223
    #     params.batch_size = 24
    #     params.seqlen = 200
    #     params.data_dir = 'data/'+dataset
    #     params.data_name = dataset

    params.save = params.data_name
    params.load = params.data_name

    # Setup
    # if "pid" not in params.data_name:
    #     dat = DATA(n_question=params.n_question,
    #                seqlen=params.seqlen, separate_char=',')
    # else:
    # Force use of PID_DATA since our files have the 4-line format
    dat = PID_DATA(n_question=params.n_question,
                       seqlen=params.seqlen, separate_char=',')
    seedNum = params.seed
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    file_name_identifier = get_file_name_identifier(params)

    ###Train- Test
    d = vars(params)
    for key in d:
        print('\t', key, '\t', d[key])
    file_name = ''
    for item_ in file_name_identifier:
        file_name = file_name+item_[0] + str(item_[1])

    # train_data_path = params.data_dir + "/" + params.data_name + "_train"+str(params.train_set)+".csv"
    # valid_data_path = params.data_dir + "/" + params.data_name + "_valid"+str(params.train_set)+".csv"


    train_data_path = params.data_dir + "/" + params.data_name + "_akt_train.csv"
    valid_data_path = params.data_dir + "/" + params.data_name + "_akt_valid.csv"

    print(f"train data path is {train_data_path}")
    print(f"valid data path is {valid_data_path}")

    train_q_data, train_qa_data, train_pid = dat.load_data(train_data_path)
    valid_q_data, valid_qa_data, valid_pid = dat.load_data(valid_data_path)

    print("\n")
    print("train_q_data.shape", train_q_data.shape)
    print("train_qa_data.shape", train_qa_data.shape)
    print("valid_q_data.shape", valid_q_data.shape)  # (1566, 200)
    print("valid_qa_data.shape", valid_qa_data.shape)  # (1566, 200)
    print("\n")
    # Train and get the best episode
    best_epoch = train_one_dataset(params, file_name, train_q_data, train_qa_data, train_pid, valid_q_data, valid_qa_data, valid_pid)
    # test_data_path = params.data_dir + "/" + params.data_name + "_test"+str(params.train_set)+".csv"
    test_data_path = params.data_dir + "/" + params.data_name + "_akt_test.csv"
    test_q_data, test_qa_data, test_index = dat.load_data(test_data_path)

    # In main.py, right before calling test_one_dataset...

    print(f"\n[DEBUG] Test Data Check:")
    print(f"  Length of test_q_data: {len(test_q_data)}")
    print(f"  Length of test_qa_data: {len(test_qa_data)}")
    print(f"  Length of test_index: {len(test_index)}")

    # Check content of first item if it exists
    if len(test_qa_data) > 0:
        print(f"  Sample QA data (first seq): {test_qa_data[0]}")
    else:
        print("  [CRITICAL] Test data is empty!")
    test_one_dataset(params, file_name, test_q_data,test_qa_data, test_index, best_epoch, valid_qa_data)
