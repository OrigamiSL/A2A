import argparse
import os
import torch
import numpy as np
import time

from exp.exp_model import Exp_Model

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=True, default='RT',
                    help='model of experiment, options: [Informer, SCTrans, LogTrans, GBET]')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S]; M:multivariate predict multivariate, '
                         'S: univariateb predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or M task')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--label_len', type=int, default=24, help='input sequence length')
parser.add_argument('--pred_list', type=str, default='24,48,168,336,720', help='group of prediction sequence length')
parser.add_argument('--enc_in', type=int, default=7, help='input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=28, help='dimension of model')
parser.add_argument('--aug_num', type=int, default=2, help='nums of data augmentation, including initial sequences')
parser.add_argument('--order_num', type=int, default=1,
                    help='nums of mixed order augmentation, excluding initial sequences')
parser.add_argument('--jitter', type=float, default=0.2, help='data augmentation amplification')
parser.add_argument('--aug', type=str, default='Scaling', help='Area loss: Scaling, Jittering')
parser.add_argument('--instance_loss', action='store_true',
                    help='include instance loss during the first stage', default=False)
parser.add_argument('--kernel', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--criterion', type=str, default='Standard', choices=['Standard', 'Maxabs'],
                    help='options:[Standard, Maxabs]')
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--cost_epochs', type=int, default=10, help='Contrastive epochs')
parser.add_argument('--cost_grow_epochs', type=int, default=5, help='Contrastive growing epochs per experiment')
parser.add_argument('--train_epochs', type=int, default=15, help='train epochs')
parser.add_argument('--cost_batch_size', type=int, default=64, help='batch size of train input data '
                                                                    'during the first stage')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data during the second stage')
parser.add_argument('--attn_nums', type=int, default=3, help='the number of streams in the main Pyramid network')
parser.add_argument('--pyramid', type=int, default=3, help='the number of Pyramid networks')
parser.add_argument('--repr_dim', type=int, default=320, help='representation dim')
parser.add_argument('--activation', type=str, default='Tanh')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--test_inverse', action='store_true', help='only inverse test data', default=False)
parser.add_argument('--save_loss', action='store_false', help='whether saving results and checkpoints', default=True)
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--train', action='store_false',
                    help='whether to train'
                    , default=True)
parser.add_argument('--reproducible', action='store_true',
                    help='whether to make results reproducible'
                    , default=False)

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() else False

if args.reproducible:
    np.random.seed(4321)  # reproducible
    torch.manual_seed(4321)
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.deterministic = False

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'M': [7, 7], 'S': [1, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'M': [7, 7], 'S': [1, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'M': [7, 7], 'S': [1, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'M': [7, 7], 'S': [1, 1]},
    'ECL': {'data': 'ECL.csv', 'M': [321, 321], 'S': [1, 1]},
    'Exchange': {'data': 'Exchange.csv', 'M': [8, 8], 'S': [1, 1]},
    'Traffic': {'data': 'Traffic.csv', 'M': [862, 862], 'S': [1, 1]},
    'weather': {'data': 'weather.csv', 'M': [21, 21], 'S': [1, 1]}
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.enc_in, args.c_out = data_info[args.features]

args.target = args.target.replace('/r', '').replace('/t', '').replace('/n', '')
args.pred_list = [int(l) for l in args.pred_list.replace(' ', '').split(',')]

lr = args.learning_rate
print('Args in experiment:')
print(args)

for ii in range(args.itr):
    # setting record of experiments
    args.cost_epochs = args.cost_epochs + args.cost_grow_epochs if ii != 0 else args.cost_epochs
    setting = '{}_{}_ft{}_ll{}_list{}_{}'.format(args.model,
                                                 args.data, args.features,
                                                 args.label_len, args.pred_list, ii)
    Exp = Exp_Model
    exp = Exp(args)  # set experiments
    if args.train:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        try:
            exp.train(setting)
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, load=True, write_loss=True, save_loss=args.save_loss)

    torch.cuda.empty_cache()
    args.learning_rate = lr
