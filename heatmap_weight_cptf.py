# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 22:54:21 2025

@author: umroot
"""

import torch
import matplotlib.pyplot as plt




#1-visualization of the heatmap of the Jacobian, i.e. derivative of the output wrt input
#define the two functions below
def compute_map(model, device="cuda"):
    model.eval()
    seq_len = model.seq_len

    # Input: batch=1, length=L, channels=enc_in
    x = torch.zeros(1, seq_len, model.enc_in, requires_grad=True, device=device)

    W = torch.zeros(seq_len, seq_len)

    for i in range(seq_len): #L
        model.zero_grad()

        # Forward
        y = model(x)  # shape [1,H,C]

        # Sum over channels if multivariate
        y_scalar = y[0, i].sum()

        # Compute gradient wrt input sequence
        grad = torch.autograd.grad(
            y_scalar,
            x,
            retain_graph=True,
            create_graph=False
        )[0]  # shape [1,H,C]

        # Sum gradient over channels → a single weight per time step
        W[i] = grad[0, :, :].sum(dim=1).detach().cpu()

    # Normalize to [0,1]
    W_norm = (W - W.min()) / (W.max() - W.min() + 1e-8)
    return W_norm #shape: L*H


def plot_heatmap(W_norm):
    plt.figure(figsize=(6,6))
    plt.imshow(W_norm, cmap="Reds", aspect="auto", origin="lower")
    plt.colorbar()
    plt.xlabel("Input Time Step (0–L)")
    plt.ylabel("Output Time Step (0–H)")
    #plt.title(title)
    plt.show()


#### you should first train the model in the main file as follows: model=exp.train(setting)
#    then you should upload the weights of the best architecture as follows:
best_model_path = "checkpoint_sl720_pl720.pth"  ##### example when L=H=720 
model.load_state_dict(torch.load(best_model_path))
model.eval()
##### plot the Jacobian
W_norm = compute_map(model, device="cuda")
plot_heatmap(W_norm)



