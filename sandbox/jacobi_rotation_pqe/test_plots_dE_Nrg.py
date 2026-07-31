import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

label_dir = ['N2_r1']

# x-axis
Nres_PQEjacobi, Nres_SRUalt_in, Nres_SRUalt_out, Nres_SRUcyc_in, Nres_SRUcyc_out, Nres_SRUgreedy_in, Nres_SRUgreedy_out, Nres_SRUrand_in, Nres_SRUrand_out = [], [], [], [], [], [], [], [], []
Ng_VQEcg, Ng_VQEjacobi, Ng_VQEpowell , Ng_VQEbfgs, Ng_VQEslsqp, Ng_VQElbfgsb = [], [], [], [], [], []

# y-axis
dE_PQEjacobi, dE_SRUalt_in, dE_SRUalt_out, dE_SRUcyc_in, dE_SRUcyc_out, dE_SRUgreedy_in, dE_SRUgreedy_out, dE_SRUrand_in, dE_SRUrand_out = [], [], [], [], [], [], [], [], []
dE_VQEcg, dE_VQEjacobi, dE_VQEpowell , dE_VQEbfgs, dE_VQEslsqp, dE_VQElbfgsb = [], [], [], [], [], []

base_dir = Path("summaries")

method_info = {
    "PQE_jacobi": (Nres_PQEjacobi, "Nrvec_ev", dE_PQEjacobi, "dE"),
    "SRU_alternating_in": (Nres_SRUalt_in, "Nrvec_ev", dE_SRUalt_in, "dE"),
    "SRU_alternating_out": (Nres_SRUalt_out, "Nrvec_ev", dE_SRUalt_out, "dE"),
    "SRU_cyclic_in": (Nres_SRUcyc_in, "Nrvec_ev", dE_SRUcyc_in, "dE"),
    "SRU_cyclic_out": (Nres_SRUcyc_out, "Nrvec_ev", dE_SRUcyc_out, "dE"),
    "SRU_greedy_in": (Nres_SRUgreedy_in, "Nrvec_ev", dE_SRUgreedy_in, "dE"),
    "SRU_greedy_out": (Nres_SRUgreedy_out, "Nrvec_ev", dE_SRUgreedy_in, "dE"),
    "SRU_random_in": (Nres_SRUrand_in, "Nrvec_ev", dE_SRUrand_in, "dE"),
    "SRU_random_out": (Nres_SRUrand_out, "Nrvec_ev", dE_SRUrand_out, "dE"),
    "VQE_CG": (Ng_VQEcg, "Ngvec_ev", dE_VQEcg, "dE"),
    "VQE_jacobi": (Ng_VQEjacobi, "Ngvec_ev", dE_VQEjacobi, "dE"),
    "VQE_POWELL": (Ng_VQEpowell, "Ngvec_ev", dE_VQEpowell, "dE"),
    "VQE_BFGS": (Ng_VQEbfgs, "Ngvec_ev", dE_VQEbfgs, "dE"),
    "VQE_SLSQP": (Ng_VQEslsqp, "Ngvec_ev", dE_VQEslsqp, "dE"),
    "VQE_l-bfgs-b": (Ng_VQElbfgsb, "Ngvec_ev", dE_VQElbfgsb, "dE")
}

for molecule in label_dir:
    molecule_dir = base_dir / molecule

    for method_dir in molecule_dir.iterdir():
        method = method_dir.name

        if method in method_info:
            list1, col1, list2, col2 = method_info[method]
            dat_file = next(method_dir.glob("*.dat"), None)

            if dat_file:
                df = pd.read_csv(dat_file, sep=r"\s+")
                filtered = df[abs(df["dE"]) < 1e-8]

            if not filtered.empty:
                list1.extend(filtered[col1].tolist())
                list2.extend(filtered[col2].tolist())
            else:
                list1.append(None)
                list2.append(None)

fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_yscale('log')

# scipy VQE
# plt.plot(Ng_VQEcg, dE_VQEcg, marker='o', label='VQEcg', color='b')
# plt.plot(Ng_VQEbfgs, dE_VQEbfgs, marker='o', label='VQEbfgs', color='b', linestyle='--')
# plt.plot(Ng_VQElbfgsb, dE_VQElbfgsb, marker='o', label='VQElbfgsb', color='b', linestyle='-.')
# plt.plot(x_pos, Ng_VQEslsqp, marker='o', label='VQEslsqp', color='b', linestyle=':') # slsqp doesnt reach energy convergence condition
# plt.plot(x_pos, Ng_VQEpowell, marker='o', label='VQEpowell',  color='b')

# Jacobi linear (VQE, PQE)
plt.plot(Ng_VQEjacobi, dE_VQEjacobi, marker='o', label='VQEjacobi', color='g')
plt.plot(Nres_PQEjacobi, dE_PQEjacobi, marker='o', label='PQEjacobi', color='g', linestyle='--')

# SRU PQE
plt.plot(Nres_SRUalt_in, dE_SRUalt_in, marker='o', label='SRUalt_in', color='m')
plt.plot(Nres_SRUalt_out, dE_SRUalt_out, marker='D', label='SRUalt_out', color='m')
plt.plot(Nres_SRUcyc_in, dE_SRUcyc_in, marker='o', label='SRUcyc_in', color='m', linestyle='--')
plt.plot(Nres_SRUcyc_out, dE_SRUcyc_out, marker='D', label='SRUcyc_out', color='m', linestyle='--')
# plt.plot(Nres_SRUgreedy_in, dE_SRUgreedy_in, marker='o', label='SRUgreedy_in', color='m', linestyle='-.')
# plt.plot(Nres_SRUgreedy_out, dE_SRUgreedy_out, marker='D', label='SRUgreedy_out', color='m', linestyle='-.')
plt.plot(Nres_SRUrand_in, dE_SRUrand_in, marker='o', label='SRUrand_in', color='m', linestyle=':')
plt.plot(Nres_SRUrand_out, dE_SRUrand_out, marker='D', label='SRUrand_out', color='m', linestyle=':')

plt.xlabel('# Grad/Res Evals')
plt.ylabel('dE')
plt.title('dE vs. Num grad/res evals for N2, r = 1 (dE < 1e-8)')
plt.legend()
plt.show()