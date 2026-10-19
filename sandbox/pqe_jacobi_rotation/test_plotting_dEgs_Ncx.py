# make dE (Efci - Epqe) vs. Num total cX gates
# for each molecule: H6 (1A), BeH2, H20, HF, LiH, N2 (color for each one)
# use VQE-BFGS, UCC quasi-newton, UCC cyc in, QNP cyc in 
# collect data for Efci and Epqe and suptract for y-ax for each method, each molecule
# collect data for total cX gates for every method, each molecule

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

# order of methods: VQE-BFGS (UCC), UCC quasi-newton, UCC cyc in, QNP cyc in
dE_H6 = [-3.235453571564+3.2360662798, -3.2354535708+3.2360662798, -3.2354535713+3.2360662798, -3.2348929907+3.2360662798]
dE_BeH2 = [-15.594733370058+15.5951175626, -15.5947333754+15.5951175626, -15.5947333748+15.5951175626, -15.5947319083+15.5951175626]
dE_H2O = [-75.013051972880+75.0131530223, -75.0130519736+75.0131530223, -75.0130519737+75.0131530223, -75.0129835481+75.0131530223]
dE_HF = [-98.597173559200+98.5971735594, -98.5971735593+98.5971735594, -98.5971735594+98.5971735594, -98.5971735594+98.5971735594]
dE_LiH = [-7.882313666206+7.8823243789, -7.8823136670+7.8823243789, -7.8823136670+7.8823243789, -7.8823144041+7.8823243789]
dE_N2 = [-107.458928931512+107.4634796198, -107.4589246758+107.4634796198, -107.4589246777+107.4634796198, -107.4543769205+107.4634796198]

cX_H6 = [13440, 13440, 13440, 630]
cX_BeH2 = [28160, 28160, 28160, 1128]
cX_H2O = [18880, 18880, 18880, 760]
cX_HF = [3520, 3520, 3520, 170]
cX_LiH = [10368, 10368, 10368, 488]
cX_N2 = [124544, 124544, 124544, 3486]

# order of molecules: as above
dE_BFGS = [-3.235453571564+3.2360662798, -15.594733370058+15.5951175626, -75.013051972880+75.0131530223, -98.597173559200+98.5971735594, -7.882313666206+7.8823243789, -107.458928931512+107.4634796198]
dE_quasi = [-3.2354535708+3.2360662798, -15.5947333754+15.5951175626, -75.0130519736+75.0131530223, -98.5971735593+98.5971735594, -7.8823136670+7.8823243789, -107.4589246758+107.4634796198]
dE_UCC_cyc = [-3.2354535713+3.2360662798, -15.5947333748+15.5951175626, -75.0130519737+75.0131530223, -98.5971735594+98.5971735594, -7.8823136670+7.8823243789, -107.4589246777+107.4634796198]
dE_QNP_cyc = [-3.2348929907+3.236066279, -15.5947319083+15.5951175626, -75.0129835481+75.0131530223, -98.5971735594+98.5971735594, -7.8823144041+7.8823243789, -107.4543769205+107.4634796198]

cX_BFGS = [13440, 28160, 18880, 3520, 10368, 124544]
cX_quasi = [13440, 28160, 18880, 3520, 10368, 124544]
cX_UCC_cyc = [13440, 28160, 18880, 3520, 10368, 124544]
cX_QNP_cyc = [630, 1128, 760, 170, 488, 3486]

fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_yscale('log')
ax.set_xscale('log')

plt.scatter(cX_H6, dE_H6, label="H6")
plt.scatter(cX_BeH2, dE_BeH2, label="BeH2")
plt.scatter(cX_H2O, dE_H2O, label="H2O")
# plt.scatter(cX_HF, dE_HF, label="HF")
plt.scatter(cX_LiH, dE_LiH, label="LiH")
plt.scatter(cX_N2, dE_N2, label="N2")
plt.axhline(y=1.2e-3, linestyle='--', color='k')
# plt.axvline(x=1e4, linestyle='--', color='k')

# plt.scatter(cX_BFGS, dE_BFGS, label="VQE BFGS")
# plt.scatter(cX_quasi, dE_quasi, label="UCC Quasi-Newton")
# plt.scatter(cX_UCC_cyc, dE_UCC_cyc, label="UCC SRU")
# plt.scatter(cX_QNP_cyc, dE_QNP_cyc, label="QNP SRU")

plt.xlabel("log(Number of cX Gates)")
plt.ylabel("log(|E_fci - E|)")
plt.title("Accuracy of Energy Convergence vs. Total cX Gates")
# plt.legend()
plt.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5)
)

plt.tight_layout()
plt.show()