# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 22:26:09 2025

@author: umroot
"""

import torch
import matplotlib.pyplot as plt


#### you should first train the model in the main file as follows: model=exp.train(setting)
#    then you should upload the weights of the best architecture as follows:

best_model_path = "checkpoint_sl720_pl720.pth"  #####  example when L=H=720 
model.load_state_dict(torch.load(best_model_path))
model.eval()
#### to plot static and dynamic components in SDD (after you train the model)
#upload data
train_data, train_loader = exp._get_data(flag='train')
for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader): 
    batch_x = batch_x.float() 
#perform forward pass of SDD
x=batch_x.to('cuda')
seq_mean = torch.mean(x, dim=1).unsqueeze(1) #shape: b,1,s
x = ((x - seq_mean)).permute(0, 2, 1) 

x3 = torch.fft.rfft(x, dim=2, norm='ortho') #ortho: normalization to conserve energy between time and frequency domains
w = torch.fft.rfft(model.w, dim=1, norm='ortho')  #frequency domain filter
x_freq_real = x3.real
x_freq_imag = x3.imag

x2=model.trend_conv(x)
if model.kernel_size%2==1:
       x2 = F.pad(x2, (0, 1))
x2_spectrum = torch.fft.rfft(x2, dim=2, norm='ortho')
       
x_freq_real = x_freq_real - x2 
x_freq_minus_emb = torch.complex(x_freq_real, x_freq_imag)
y = x_freq_minus_emb * w
y_real = y.real
y_freq_imag = y.imag


# Extract spectra
ts_spec = x3[0, 0, :x3.shape[-1]//2].cpu().detach().numpy()
static_spec = x2_spectrum[0, 0, :].cpu().detach().numpy()
dyn_spec = y[0, 0, :].cpu().detach().numpy()

# Create side-by-side subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# --- Left plot ---
axs[0].plot(ts_spec, label="Time series spectrum")
axs[0].plot(static_spec, label="Static component spectrum")
axs[0].set_xlabel("Frequency")
axs[0].set_ylabel("Amplitude")
axs[0].legend()

# Add (a) under the first subplot
axs[0].text(0.5, -0.25, "(a)", transform=axs[0].transAxes,
            ha="center", va="center", fontsize=12)

# --- Right plot ---
axs[1].plot(dyn_spec)
axs[1].set_xlabel("Frequency")
axs[1].set_ylabel("Amplitude")

# Add (b) under the second subplot
axs[1].text(0.5, -0.25, "(b)", transform=axs[1].transAxes,
            ha="center", va="center", fontsize=12)

plt.tight_layout()
plt.show()

