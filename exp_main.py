# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 19:03:33 2025

@author: umroot
"""

from data_provider.data_factory import data_provider
from exp_basic import Exp_Basic
import LightTFF#SparseTSF
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        #FOR timeembed there's a case to consider for noise injection (robustness analysis)
    def _build_model(self):
        model_dict = {
            # 'Autoformer': Autoformer,
            # 'Transformer': Transformer,
            # 'Informer': Informer,
            # 'DLinear': DLinear,
            # 'Linear': Linear,
            # 'PatchTST': PatchTST,
            'LightTFF': LightTFF 
        }
        #test_model
        model = LightTFF.Model(self.args).float()#model_dict[self.args.model].Model(self.args).float() #=SparseTSF(the file inside models folder). the class model.
        #model is now instanciated as in the sparsetsf file
        if self.args.use_multi_gpu and self.args.use_gpu: #false by default
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate) #self.model refers to the model instance created in build_model
        return model_optim

    def _select_criterion(self):
        if self.args.loss == "mae":
            criterion = nn.L1Loss()
        elif self.args.loss == "mse":
            criterion = nn.MSELoss()
        elif self.args.loss == "smooth":
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            #timeembed
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, hour_index, day_index) in enumerate(vali_loader):
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                #timeembed
                # hour_index = hour_index.int().to(self.device)
                # # without week-level bank
                # if torch.all(day_index != -1):
                #     day_index = day_index.int().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp: #false
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'TimeEmb', 'SparseTSF', 'LightTFF'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'TimeEmb', 'SparseTSF', 'LightTFF'}): #yes
                        outputs = self.model(batch_x)
                        #outputs = self.model(batch_x, hour_index, day_index)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0 #0
                outputs = outputs[:, -self.args.pred_len:, f_dim:] #the model reconstructs the start tokens and predicts the future (we're interested only in the fu)
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss #valid_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True) #patience = 100

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp: #false by default
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        #plot train vs valid loss
        train_losses,valid_losses=[],[]
        epochs = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            #timeembed
            #for i,(batch_x, batch_y, batch_x_mark, batch_y_mark, hour_index, day_index) in enumerate(train_loader):
            for i,(batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device) #refers to calendar features
                batch_y_mark = batch_y_mark.float().to(self.device)
                #timeembed
                # hour_index = hour_index.int().to(self.device)
                # if torch.all(day_index != -1):
                #     day_index = day_index.int().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device) #concat the starting tokens of y and the zero placeholders

                # encoder - decoder 
                if self.args.use_amp: #false
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'TimeEmb', 'SparseTSF', 'LightTFF'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0 #'M':0 
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else: 
                    if any(substr in self.args.model for substr in {'TimeEmb', 'SparseTSF', 'LightTFF'}): #yes
                        outputs = self.model(batch_x) #we are in this case
                        #timeembed
                        #outputs = self.model(batch_x, hour_index, day_index)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0 #0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    #loss timeembed
                    # if self.args.rec_lambda:
                    #     loss_rec = criterion(outputs, batch_y)
                         # loss += self.args.rec_lambda * loss_rec
                         #        if (i + 1) % 100 == 0:
                         #            print(f"\tloss_rec: {loss_rec.item()}")
                    #if self.args.auxi_lambda: default 1
                    #if self.args.auxi_mode == "fft": #default value fft
                    #there are other cases (rfft) but we won't put them here for simplicity (l223)
                    ####################
                    #loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)
                    ################
                    #if self.mask is not None: none by default
                    #    loss_auxi *= self.mask
                    #if self.args.auxi_loss == "MAE": #mae by default (see loss equation in the paper)
                    #the other case is mse, but we omitted it for simplicity and consistency with the paper
                    #########################
                    # loss_auxi = loss_auxi.abs().mean() #loss_auxi.abs().mean() if self.args.module_first else loss_auxi.mean().abs()  # check the dim of fft
                    # loss += loss_auxi #loss += self.args.auxi_lambda * loss_auxi
                    # train_loss.append(loss.item())
                    #######################
                    #standard loss (for other models)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp: #false
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST': #false
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time)) #training time
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            train_losses.append(train_loss)
            valid_losses.append(vali_loss)
            epochs.append(epoch+1)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Vali Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST': 
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label='Training Loss', marker='o')
        plt.plot(epochs, valid_losses, label='Validation Loss', marker='o')
        
        plt.title('Training and Validation Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location="cuda:0"))


        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location="cuda:0"))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            #timeembed
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, hour_index, day_index) in enumerate(test_loader):
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                #timeembed
                # hour_index = hour_index.int().to(self.device)
                # if torch.all(day_index != -1):
                #     day_index = day_index.int().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp: #false
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'TimeEmb', 'SparseTSF', 'LightTFF'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'TimeEmb', 'SparseTSF', 'LightTFF'}):#yes
                        outputs = self.model(batch_x)
                        #outputs = self.model(batch_x, hour_index, day_index)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0 #0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                # inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:#select the 20th sample to perform visualization pred vs true
                    input = batch_x.detach().cpu().numpy()
                    #gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    #pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    #visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    visual(true[0, :, -1], pred[0, :, -1], os.path.join(folder_path, str(i) + 'ot'+'.pdf'))
                    visual(true[0, :, 0], pred[0, :, 0], os.path.join(folder_path, str(i) + 'hufl'+'.pdf'))
                    #plot problem 
                    # plt.figure()
                    # plt.plot(gt,label='true')
                    # plt.plot(pd,label='predict')
                    # plt.legend()
                    # plt.show()

        if self.args.test_flop: #false
            test_params_flop(self.model, (batch_x.shape[1],batch_x.shape[2]))
            # test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()

        # fix bug
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return mse, mae

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'TimeEmb', 'SparseTSF', 'LightTFF'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'TimeEmb', 'SparseTSF', 'LightTFF'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return