# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 17:49:30 2025

@author: umroot
"""






#2-plot the acf functions
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ---------------------------------------------------------
# Function to load, scale, compute ACF
# ---------------------------------------------------------
def compute_acf(csv_path):
    df_raw = pd.read_csv(csv_path)
    
    # Convert date column to datetime
    # df_raw['date'] = pd.to_datetime(df_raw['date'])
    # # Set index to date for resampling
    # df_raw = df_raw.set_index('date')
    # # Downsample: daily mean
    # df_raw = df_raw.resample('D').mean()
    # # Remove any NaNs introduced by resampling
    # df_raw = df_raw.dropna()
    
    scaler = StandardScaler()
    cols_data = df_raw.columns[1:]  # exclude date column
    df_data = df_raw[cols_data]

    # 1 year training window
    train_data = df_data[:12*30*24]
    scaler.fit(train_data.values)
    data = scaler.transform(df_data.values)

    train_data = data[:12*30*24]

    # Autocorrelation on the first variable (HUFL)
    acf_values = acf(train_data[:, 0], nlags=6*24) 
    return acf_values


# ---------------------------------------------------------
# Compute both ACFs
# ---------------------------------------------------------
acf_etth1 = compute_acf('data/ETT/ETTh1.csv')
acf_etth2 = compute_acf('data/ETT/ETTh2.csv')

lags1 = np.arange(len(acf_etth1))
lags2 = np.arange(len(acf_etth2))

# ---------------------------------------------------------
# Side-by-side plots
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

# =========================================
# Left subplot (ETTh1)
# =========================================
ax1.bar(lags1, acf_etth1, color='steelblue')
ax1.set_xlim([0, 6*24])
ax1.set_ylim([0, 1])
ax1.set_xlabel("Lags", fontsize=11)
ax1.set_ylabel("Autocorrelation", fontsize=11)
ax1.grid(True)

# Bottom label (a)
ax1.text(0.5, -0.70, "(a)", ha='center', va='center',
         transform=ax1.transAxes, fontsize=11)

# =========================================
# Right subplot (ETTh2)
# =========================================
ax2.bar(lags2, acf_etth2, color='steelblue')
ax2.set_xlim([0, 6*24])
ax2.set_ylim([0, 1])
ax2.set_xlabel("Lags", fontsize=11)
ax2.grid(True)

# Bottom label (b)
ax2.text(0.5, -0.70, "(b)", ha='center', va='center',
         transform=ax2.transAxes, fontsize=11)

# ---------------------------------------------------------
plt.tight_layout()
plt.show()






