# make two types of graphs
# 1 - plotting dE vs N cX gates - scatter plot for all molecules, for UCC jacobi, UCC sru, QNP sru (show shorter depth for QNP potentially)
# 2 - plotting dE vs Nres (rate of convergence) - for one molecule, H6 (not stretched at 1 A and stretched at 1.5 A)) , put curves for UCC jacobi, UCC sru (best variants), QNP sru
    # state final converged E for each method and state fci energy (exact GS energy) to show how close QNP converged final energy is to the Efci compared to the UCC ansatz
    # maybe use a table ^^
    
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

label_dir = ['H6_r1']  # List of molecule directories to process

# x-axis
Ngrad_BFGS, Nres_UCCjacobi, Nres_UCC_SRUalt_in, Nres_UCC_SRUalt_out, Nres_UCC_SRUcyc_in, Nres_UCC_SRUcyc_out, Nres_UCC_SRUgreedy_in, Nres_UCC_SRUgreedy_out, Nres_UCC_SRUrand_in, Nres_UCC_SRUrand_out = [], [], [], [], [], [], [], [], [], []
Nres_QNP_SRUalt_in, Nres_QNP_SRUalt_out, Nres_QNP_SRUcyc_in, Nres_QNP_SRUcyc_out, Nres_QNP_SRUgreedy_in, Nres_QNP_SRUgreedy_out, Nres_QNP_SRUrand_in, Nres_QNP_SRUrand_out = [], [], [], [], [], [], [], []

# y-axis
dE_BFGS, dE_UCCjacobi, dE_UCC_SRUalt_in, dE_UCC_SRUalt_out, dE_UCC_SRUcyc_in, dE_UCC_SRUcyc_out, dE_UCC_SRUgreedy_in, dE_UCC_SRUgreedy_out, dE_UCC_SRUrand_in, dE_UCC_SRUrand_out = [], [], [], [], [], [], [], [], [], []
dE_QNP_SRUalt_in, dE_QNP_SRUalt_out, dE_QNP_SRUcyc_in, dE_QNP_SRUcyc_out, dE_QNP_SRUgreedy_in, dE_QNP_SRUgreedy_out, dE_QNP_SRUrand_in, dE_QNP_SRUrand_out = [], [], [], [], [], [], [], []

base_dir = Path(__file__).resolve().parent / "summaries"

method_info = {
    "UCC": {
        "VQE_BFGS": (Ngrad_BFGS, "Ngvec_ev", dE_BFGS, "dE"),
        "PQE_jacobi": (Nres_UCCjacobi, "Nrvec_ev", dE_UCCjacobi, "dE"),
        "SRU_alternating_in": (Nres_UCC_SRUalt_in, "Nrvec_ev", dE_UCC_SRUalt_in, "dE"),
        "SRU_alternating_out": (Nres_UCC_SRUalt_out, "Nrvec_ev", dE_UCC_SRUalt_out, "dE"),
        "SRU_cyclic_in": (Nres_UCC_SRUcyc_in, "Nrvec_ev", dE_UCC_SRUcyc_in, "dE"),
        "SRU_cyclic_out": (Nres_UCC_SRUcyc_out, "Nrvec_ev", dE_UCC_SRUcyc_out, "dE"),
        "SRU_greedy_in": (Nres_UCC_SRUgreedy_in, "Nrvec_ev", dE_UCC_SRUgreedy_in, "dE"),
        "SRU_greedy_out": (Nres_UCC_SRUgreedy_out, "Nrvec_ev", dE_UCC_SRUgreedy_out, "dE"),
        "SRU_random_in": (Nres_UCC_SRUrand_in, "Nrvec_ev", dE_UCC_SRUrand_in, "dE"),
        "SRU_random_out": (Nres_UCC_SRUrand_out, "Nrvec_ev", dE_UCC_SRUrand_out, "dE"),
    },

    "QNP": {
        "SRU_alternating_in": (Nres_QNP_SRUalt_in, "Nrvec_ev", dE_QNP_SRUalt_in, "dE"),
        "SRU_alternating_out": (Nres_QNP_SRUalt_out, "Nrvec_ev", dE_QNP_SRUalt_out, "dE"),
        "SRU_cyclic_in": (Nres_QNP_SRUcyc_in, "Nrvec_ev", dE_QNP_SRUcyc_in, "dE"),
        "SRU_cyclic_out": (Nres_QNP_SRUcyc_out, "Nrvec_ev", dE_QNP_SRUcyc_out, "dE"),
        "SRU_greedy_in": (Nres_QNP_SRUgreedy_in, "Nrvec_ev", dE_QNP_SRUgreedy_in, "dE"),
        "SRU_greedy_out": (Nres_QNP_SRUgreedy_out, "Nrvec_ev", dE_QNP_SRUgreedy_out, "dE"),
        "SRU_random_in": (Nres_QNP_SRUrand_in, "Nrvec_ev", dE_QNP_SRUrand_in, "dE"),
        "SRU_random_out": (Nres_QNP_SRUrand_out, "Nrvec_ev", dE_QNP_SRUrand_out, "dE"),
    }
}

for molecule in label_dir:
    molecule_dir = base_dir / molecule

    for method_dir in molecule_dir.iterdir():
        if not method_dir.is_dir():
            continue

        method = method_dir.name

        for dat_file in method_dir.glob("*.dat"):
            filename = dat_file.stem.upper()

            # Determine ansatz from filename
            if filename.startswith("UCC"):
                ansatz = "UCC"
            elif filename.startswith("QNP"):
                ansatz = "QNP"
            else:
                print(f"Could not identify ansatz for {dat_file}")
                continue

            # Skip methods not defined for this ansatz
            if method not in method_info[ansatz]:
                continue

            list1, col1, list2, col2 = method_info[ansatz][method]

            # Read data
            df = pd.read_csv(dat_file, sep=r"\s+")

            # Find first iteration where |dE| < 1e-8
            converged = df["dE"].abs() < 1e-8

            if converged.any():
                first_converged_idx = converged[converged].index[0]

                # Keep everything from iteration 1 through convergence
                filtered = df.loc[:first_converged_idx]
            else:
                # If it never converged, keep the entire calculation
                filtered = df

            # Store x and y values
            list1.extend(filtered[col1].tolist())
            list2.extend(filtered[col2].abs().tolist())

fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_yscale('log')

# VQE BFGS (UCC)
# plt.plot(Ngrad_BFGS, dE_BFGS, marker='o', label='VQE BFGS', color='b', linestyle='--')

# Jacobi linear (UCC)
plt.plot(Nres_UCCjacobi, dE_UCCjacobi, marker='o', label='UCC Quasi-Newton (prior work)', color='g', linestyle='--')

# UCC SRU
# plt.plot(Nres_UCC_SRUalt_in, dE_UCC_SRUalt_in, marker='o', label='UCC alt in', color='m')
# plt.plot(Nres_UCC_SRUalt_out, dE_UCC_SRUalt_out, marker='D', label='UCC alt out', color='m')
plt.plot(Nres_UCC_SRUcyc_in, dE_UCC_SRUcyc_in, marker='o', label='UCC SRU (this work)', color='m', linestyle='--')
# plt.plot(Nres_UCC_SRUcyc_out, dE_UCC_SRUcyc_out, marker='D', label='UCC cyc out', color='m', linestyle='--')
# plt.plot(Nres_UCC_SRUgreedy_in, dE_UCC_SRUgreedy_in, marker='o', label='UCC greedy in', color='m', linestyle='-.')
# plt.plot(Nres_UCC_SRUgreedy_out, dE_UCC_SRUgreedy_out, marker='D', label='UCC greedy out', color='m', linestyle='-.')
# plt.plot(Nres_UCC_SRUrand_in, dE_UCC_SRUrand_in, marker='o', label='UCC rand in', color='m', linestyle=':')
# plt.plot(Nres_UCC_SRUrand_out, dE_UCC_SRUrand_out, marker='D', label='UCC rand out', color='m', linestyle=':')

# QNP SRU
# CHECK - SRU rotation update equation for QNP (might be different than UCC)
# plt.plot(Nres_QNP_SRUalt_in, dE_QNP_SRUalt_in, marker='o', label='QNP alt in', color='r')
# plt.plot(Nres_QNP_SRUalt_out, dE_QNP_SRUalt_out, marker='D', label='QNP alt out', color='r')
plt.plot(Nres_QNP_SRUcyc_in, dE_QNP_SRUcyc_in, marker='o', label='QNP SRU', color='r', linestyle='--')
# print(dE_QNP_SRUcyc_in)
# plt.plot(Nres_QNP_SRUcyc_out, dE_QNP_SRUcyc_out, marker='D', label='QNP cyc out', color='r', linestyle='--')
# plt.plot(Nres_QNP_SRUgreedy_in, dE_QNP_SRUgreedy_in, marker='o', label='QNP greedy in', color='r', linestyle='-.')
# plt.plot(Nres_QNP_SRUgreedy_out, dE_QNP_SRUgreedy_out, marker='D', label='QNP greedy out', color='r', linestyle='-.')
# plt.plot(Nres_QNP_SRUrand_in, dE_QNP_SRUrand_in, marker='o', label='QNP rand in', color='r', linestyle=':')
# plt.plot(Nres_QNP_SRUrand_out, dE_QNP_SRUrand_out, marker='D', label='QNP rand out', color='r', linestyle=':')

plt.xlabel('# Residual Vector Evals')
plt.ylabel('log(dE)')
plt.title('Change in Energy vs. Num Residual Evals for H6, r = 1 A')
plt.legend()
# plt.legend(
#     loc="center left",
#     bbox_to_anchor=(1.02, 0.5)
# )

plt.tight_layout()
plt.show()