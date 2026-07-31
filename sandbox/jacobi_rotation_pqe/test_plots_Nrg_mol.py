import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# x-axis
x_pos = [1, 2, 3, 4, 5, 6]
labels = ['HF','LiH','BeH2','H2O','Li2','H8']
labels_dir = ['HF_r0.92','LiH_r1','BeH2_r1.33','H2O_r0.96','Li2_r1','H8_r1']

# y-axis
Nres_PQEjacobi, Nres_SRUalt_in, Nres_SRUalt_out, Nres_SRUcyc_in, Nres_SRUcyc_out, Nres_SRUgreedy_in, Nres_SRUgreedy_out, Nres_SRUrand_in, Nres_SRUrand_out = [], [], [], [], [], [], [], [], []
Ng_VQEcg, Ng_VQEjacobi, Ng_VQEpowell , Ng_VQEbfgs, Ng_VQEslsqp, Ng_VQElbfgsb = [], [], [], [], [], []

base_dir = Path("summaries")

method_info = {
    "PQE_jacobi": (Nres_PQEjacobi, "Nrvec_ev"),
    "SRU_alternating_in": (Nres_SRUalt_in, "Nrvec_ev"),
    "SRU_alternating_out": (Nres_SRUalt_out, "Nrvec_ev"),
    "SRU_cyclic_in": (Nres_SRUcyc_in, "Nrvec_ev"),
    "SRU_cyclic_out": (Nres_SRUcyc_out, "Nrvec_ev"),
    "SRU_greedy_in": (Nres_SRUgreedy_in, "Nrvec_ev"),
    "SRU_greedy_out": (Nres_SRUgreedy_out, "Nrvec_ev"),
    "SRU_random_in": (Nres_SRUrand_in, "Nrvec_ev"),
    "SRU_random_out": (Nres_SRUrand_out, "Nrvec_ev"),
    "VQE_CG": (Ng_VQEcg, "Ngvec_ev"),
    "VQE_jacobi": (Ng_VQEjacobi, "Ngvec_ev"),
    "VQE_POWELL": (Ng_VQEpowell, "Ngvec_ev"),
    "VQE_BFGS": (Ng_VQEbfgs, "Ngvec_ev"),
    "VQE_SLSQP": (Ng_VQEslsqp, "Ngvec_ev"),
    "VQE_l-bfgs-b": (Ng_VQElbfgsb, "Ngvec_ev")
}

for molecule in labels_dir:
    molecule_dir = base_dir / molecule

    for method_dir in molecule_dir.iterdir():
        method = method_dir.name

        if method in method_info:
            target_list, column = method_info[method]
            dat_file = next(method_dir.glob("*.dat"), None)

            if dat_file:
                df = pd.read_csv(dat_file, sep=r"\s+")
                mask = abs(df["dE"]) < 1e-8

                if mask.any():
                    idx = df[mask].index[0]
                    value = df.loc[idx, column]
                    target_list.append(value)
                else: # handle case where convergence not reached
                    target_list.append(None)

print(Ng_VQEbfgs)
print(Ng_VQElbfgsb)

# scipy VQE
plt.plot(x_pos, Ng_VQEcg, marker='o', label='VQEcg', color='b')
plt.plot(x_pos, Ng_VQEbfgs, marker='o', label='VQEbfgs', color='b', linestyle='--')
# plt.plot(x_pos, Ng_VQElbfgsb, marker='o', label='VQElbfgsb', color='b', linestyle='-.')
# plt.plot(x_pos, Ng_VQEslsqp, marker='o', label='VQEslsqp', color='b', linestyle=':') # slsqp doesnt reach energy convergence condition
# plt.plot(x_pos, Ng_VQEpowell, marker='o', label='VQEpowell',  color='b')

# Jacobi linear (VQE, PQE)
plt.plot(x_pos, Ng_VQEjacobi, marker='o', label='VQEjacobi', color='g')
plt.plot(x_pos, Nres_PQEjacobi, marker='o', label='PQEjacobi', color='g', linestyle='--')

# SRU PQE
plt.plot(x_pos, Nres_SRUalt_in, marker='o', label='SRUalt_in', color='m')
plt.plot(x_pos, Nres_SRUalt_out, marker='D', label='SRUalt_out', color='m')
plt.plot(x_pos, Nres_SRUcyc_in, marker='o', label='SRUcyc_in', color='m', linestyle='--')
plt.plot(x_pos, Nres_SRUcyc_out, marker='D', label='SRUcyc_out', color='m', linestyle='--')
plt.plot(x_pos, Nres_SRUgreedy_in, marker='o', label='SRUgreedy_in', color='m', linestyle='-.')
plt.plot(x_pos, Nres_SRUgreedy_out, marker='D', label='SRUgreedy_out', color='m', linestyle='-.')
plt.plot(x_pos, Nres_SRUrand_in, marker='o', label='SRUrand_in', color='m', linestyle=':')
plt.plot(x_pos, Nres_SRUrand_out, marker='D', label='SRUrand_out', color='m', linestyle=':')

plt.xticks(x_pos, labels)
plt.xlabel('Molecules (increasing size)')
plt.ylabel('# Grad/Res Evals')
plt.title('Num grad/res evals vs. Molcules in Increasing Size (dE < 1e-6)')
plt.legend()
plt.show()