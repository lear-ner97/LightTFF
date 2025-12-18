# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 19:32:46 2025

@author: umroot
"""
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from torch.autograd import Function





class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_sizee, stride):
        super(moving_avg, self).__init__()
        self.kernel_sizee = kernel_sizee
        self.avg = nn.AvgPool1d(kernel_size=kernel_sizee, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_sizee - 1-math.floor((self.kernel_sizee - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_sizee - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # common hyperparameters 
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len 
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        
        
        #CPTF variables
        self.period_len = configs.period_len #w=24 or 24*4
        
        self.model_type = configs.model_type
        assert self.model_type in ['linear', 'mlp']
        self.seg_num_x = self.seq_len  // self.period_len #L/w
        self.seg_num_y = self.pred_len // self.period_len

        #activate self.conv1d only if static=='conv' in the text file
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2), #k=1+2*(w/2)
                                   stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False) #padding=w/2

        if self.model_type == 'linear':
            self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False) # L/w, H/w
            # self.linear = LowRank(in_features=self.seg_num_x, 
            #                                 out_features=self.seg_num_y, 
            #                                 rank=40, 
            #                                 bias=True)
        elif self.model_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(self.seg_num_x, self.d_model),
                nn.ReLU(),#relu
                nn.Linear(self.d_model, self.seg_num_y)
            )
        

        #variables of SDD
        
        self.scale=configs.scale
        self.w = nn.Parameter(self.scale * torch.randn(1, self.seq_len))   #see the effect of different constant 
        
        layers = [
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),  ######relu
                nn.Linear(self.d_model, self.pred_len)
        ]
        self.model = nn.Sequential(*layers)
    
        self.kernel_size = configs.kernel_size
        
        #option A: cross-conv filter, activate only if static=='static' in the text file
        self.bias_trend=configs.bias_trend
        self.trend_conv = nn.Conv1d(
            in_channels=7,#self.enc_in,
            out_channels=1,#self.enc_in,
            kernel_size=self.kernel_size,
            stride=2,
            padding=self.kernel_size // 2,
            #groups=self.enc_in,  # depthwise to learn per-channel trend
            bias=self.bias_trend #see the effect of setting it to true
        )
        # # Initialize weights as moving average
        nn.init.constant_(self.trend_conv.weight, 1.0 / self.kernel_size) #to bias the filter towards smoothing

        
        #option B: moving average filter, activate only if static=='ma'
        #self.moving_avg=moving_avg(self.kernel_size,stride=2)
  
        
  

        #weighted average: activate only if static=='ma' in the text file
        # self.w1 = nn.Parameter(torch.ones(self.pred_len,self.enc_in) * 0.5) #0.5 is the initial weight
        # self.w2 = nn.Parameter(torch.ones(self.pred_len,self.enc_in) * 0.5)
        # self.b = nn.Parameter(torch.zeros(self.pred_len,self.enc_in))
        


    def forward(self, x):
        batch_size = x.shape[0]        
        # normalization and permute     b,s,c
        seq_mean = torch.mean(x, dim=1).unsqueeze(1) #shape: b,1,s
        #seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5 
        x = ((x - seq_mean)).permute(0, 2, 1)  #b,c,s  / torch.sqrt(seq_var)
        
        # CPTF
        # aggregation: activate only if static=='conv' in the text file
        x1 = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x 
        ## activate the next line only if static=='ma' in the text file
        #x1 = x 

        # downsampling: b,c,s -> bc,n,w -> bc,w,n (n=L//w) 
        # the shape we want here is bc,n,w. each column corresponds to a particular day. The last reshaping was done
        # for shape alignment with the linear layer
        x1 = x1.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1) 

        # sparse forecasting
        if self.model_type == 'linear':
            x1 = self.linear(x1)  # bc,w,m
        elif self.model_type == 'mlp':
            x1 = self.mlp(x1)
        
        # upsampling: bc,w,m -> bc,m,w -> b,c,H
        x1= x1.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len).permute(0, 2, 1) ####################
        
        
        
        ### SDD
        #rfft to keep only non-redundant frequencies
        x3 = torch.fft.rfft(x, dim=2, norm='ortho') #ortho: normalization to conserve energy between time and frequency domains
        w = torch.fft.rfft(self.w, dim=1, norm='ortho')  #frequency domain filter
        x_freq_real = x3.real
        x_freq_imag = x3.imag
        

        
        # #optionA: cross-conv layer, activate only if static=='conv' in the text file
        x2=self.trend_conv(x)
        if self.kernel_size%2==1:
               x2 = F.pad(x2, (0, 1))
        
        #optionB: moving average filter, activate only if static=='ma' in the text file
        # x2=self.moving_avg(x.permute(0,2,1)).permute(0,2,1) #in case of shape error add permute at the end
        # x2 = F.pad(x2, (0, 1))
        


        x_freq_real = x_freq_real - x2 
        
        x_freq_minus_emb = torch.complex(x_freq_real, x_freq_imag)
        y = x_freq_minus_emb * w
        y_real = y.real
        y_freq_imag = y.imag
        
        y_real = y_real + x2
        
        y_freq = torch.complex(y_real, y_freq_imag)
        y = torch.fft.irfft(y_freq, n=self.seq_len, dim=2, norm="ortho")
        
        y = self.model(y).permute(0, 2, 1)

        
        
        
        
        #strategy A: simple average, activate only if static=='conv' in the text file
        y=0.5*(y+x1)#(x1+x1)#
        
        #strategy B: weighted average, activate only if static=='ma' in the text file
        # w1 = self.w1.unsqueeze(0)  # shape (1, C, H)
        # w2 = self.w2.unsqueeze(0)
        # y = w1 * x1 + w2 * y + self.b.unsqueeze(0)
        

        
        #denormalization
        y = y + seq_mean
        
        return y

