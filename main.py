# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 15:26:51 2025

@author: umroot              
"""

import argparse
import os
import itertools
import torch
from exp_main import Exp_Main
import matplotlib.pyplot as plt
import random
import numpy as np



parser = argparse.ArgumentParser(description='LightTFF & other models for Time Series Forecasting')


##### arguments to change
# forecasting scenario
parser.add_argument('--data', type=str, default='ETTh2', help='dataset type') #required=True,
parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length') #### 96 case ettm needs refinement
parser.add_argument('--pred_len', type=int, default= 192, help='prediction sequence length')

#lr and batch size
parser.add_argument('--learning_rate', type=float, default=0.04, help='optimizer learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
#SDD
#parser.add_argument('--static', type=str, default='conv', help='type of the static component extractor, 
#cross-convolutional layer or moving average')
parser.add_argument('--kernel_size', type=int, default=8, help='decomposition-kernel')
parser.add_argument('--scale', type=float, default=0.02, help='trend extractor bias')
#SDD, with cross-channel Conv
parser.add_argument('--bias_trend', type=bool, default=False, help='trend extractor bias')
#CPTF
parser.add_argument('--model_type', default='mlp', help='model type: linear/mlp') 
parser.add_argument('--d_model', type=int, default=256, help='dimension of model') #tune




###### fixed arguments, Do not change
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
# basic config
parser.add_argument('--is_training', type=int, default=1, help='status') #required=True,
parser.add_argument('--model_id', type=str,  default='test', help='model id')#required=True,
parser.add_argument('--model', type=str,  default='LightTFF', help='model name')#required=True,
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
# srart token length
parser.add_argument('--label_len', type=int, default=48, help='start token length') #number of past target steps used as input to initialize the decoder before predicting the future.

# LightTFF, period
parser.add_argument('--period_len', type=int, default=24, help='period length')

# number of channels
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
#time feature encoding
parser.add_argument('--embed', type=str, default='learned',
                     help='time features encoding, options:[timeF, fixed, learned]')
# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers') 
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--des', type=str, default='test', help='exp description')         #test
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate') ####see tools
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start') #The percentage of the cycle (in number of steps) spent increasing the learning rate.
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', type=int, help='use multiple gpus', default=0)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

# random seed
fix_seed_list = range(2023, 2033)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False



print('Args in experiment:')
print(args)

Exp = Exp_Main



if args.is_training:
    for ii in range(args.itr): ### number of seeds, how many experiments you want to run
        #for reproducibility
        random.seed(fix_seed_list[ii])
        torch.manual_seed(fix_seed_list[ii])
        np.random.seed(fix_seed_list[ii])
        torch.cuda.manual_seed(fix_seed_list[ii])
        torch.cuda.manual_seed_all(fix_seed_list[ii])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_{}_{}_{}_seed{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.model_type,
            args.des,
            ii,
            fix_seed_list[ii])

        exp = Exp(args) 
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        mse, mae=exp.test(setting)
        print(f"MSE: {mse}, MAE: {mae}")


        torch.cuda.empty_cache()

else: 
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_{}_{}_{}_seed{}'.format(#args.model_type removed
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.model_type,
        args.des,
        ii,
        fix_seed_list[ii])

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
        



