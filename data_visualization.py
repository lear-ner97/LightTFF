# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 22:08:58 2025

@author: umroot
"""

### 1-plot a zoom of 1 week for both datasets etth1 and etth2 next to each other
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---- Load both datasets ----
df1_raw = pd.read_csv('data/ETT/ETTh1.csv')
df2_raw = pd.read_csv('data/ETT/ETTh2.csv')

type_load='HUFL'

# Convert date columns to datetime
df1_raw['date'] = pd.to_datetime(df1_raw['date'])
df2_raw['date'] = pd.to_datetime(df2_raw['date'])

# Zoom window (1 week)
zoom = len(df1_raw) #24 * 7

# ---- Create side-by-side plots ----
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 5))

# ========================================
# LEFT PLOT : ETTh2
# ========================================

ax2 = ax1.twinx()     # Right axis for OT

# HUFL
ax1.plot(df1_raw['date'][:zoom], df1_raw[type_load][:zoom],
         color="blue", label=type_load)
ax1.set_xlabel("Date")
ax1.set_ylabel("Load", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

# OT
ax2.plot(df1_raw['date'][:zoom], df1_raw['OT'][:zoom],
         color="red", label="OT")
ax2.set_ylabel("Temperature", color="red")
ax2.tick_params(axis="y", labelcolor="red")

# Format dates
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.tick_params(axis='x', rotation=45)

# Combined legend
ln1 = ax1.lines[0]
ln2 = ax2.lines[0]
ax1.legend([ln1, ln2], [type_load, "OT"], loc="best")

# ---- Bottom subplot label (a)
ax1.text(0.5, -0.7, "(a)",
         ha='center', va='center', transform=ax1.transAxes,
         fontsize=12)


# ========================================
# RIGHT PLOT : ETTh1
# ========================================

ax4 = ax3.twinx()     # Right axis for OT

# HUFL
ax3.plot(df2_raw['date'][:zoom], df2_raw[type_load][:zoom],
         color="blue", label=type_load)
ax3.set_xlabel("Date")
ax3.set_ylabel("Load", color="blue")
ax3.tick_params(axis="y", labelcolor="blue")

# OT
ax4.plot(df2_raw['date'][:zoom], df2_raw['OT'][:zoom],
         color="red", label="OT")
ax4.set_ylabel("Temperature", color="red")
ax4.tick_params(axis="y", labelcolor="red")

# Format dates
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
ax3.tick_params(axis='x', rotation=45)

# Combined legend
ln1 = ax3.lines[0]
ln2 = ax4.lines[0]
ax3.legend([ln1, ln2], [type_load, "OT"], loc="upper center")

# ---- Bottom subplot label (b)
ax3.text(0.5, -0.7, "(b)",
         ha='center', va='center', transform=ax3.transAxes,
         fontsize=12)

# --------------------------------------
plt.tight_layout()
plt.show()