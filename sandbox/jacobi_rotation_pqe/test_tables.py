import numpy as np
import pandas as pd

sys_strings = ["H8", "LiH", "BeH2", "H2O", "HF"]

# VQE algos
vqe_jacobi_dE, vqe_jacobi_Ngrad_ev, vqe_jacobi_Npt_ev = [], [], []

# PQE algos
pqe_jacobi_dE, pqe_jacobi_Nres_ev, pqe_jacobi_Npt_ev = [], [], []
sru_alt_in_dE, sru_alt_in_Nres, sru_alt_in_Npt = [], [], []
sru_alt_out_dE, sru_alt_out_Nres, sru_alt_out_Npt = [], [], []
sru_cyc_in_dE, sru_cyc_in_Nres, sru_cyc_in_Npt = [], [], []
sru_cyc_out_dE, sru_cyc_out_Nres, sru_cyc_out_Npt = [], [], []

for ii in range(0,5):
    sys_str = sys_strings[ii]

    df_vqe_jacobi = pd.read_csv(f'summaries/summary_vqejacobi_{sys_str}.dat', sep=r"\s+")
    vqe_jacobi_dE.append(df_vqe_jacobi['dE'].iloc[-1])
    vqe_jacobi_Ngrad_ev.append(df_vqe_jacobi["Ngvec_ev"].iloc[-1])
    vqe_jacobi_Npt_ev.append(df_vqe_jacobi["Npt_ev"].iloc[-1])

    df_pqe_jacobi = pd.read_csv(f'summaries/summary_pqejacobi_{sys_str}.dat', sep=r"\s+")
    pqe_jacobi_dE.append(df_pqe_jacobi['dE'].iloc[-1])
    pqe_jacobi_Nres_ev.append(df_pqe_jacobi["Nrvec_ev"].iloc[-1])
    pqe_jacobi_Npt_ev.append(df_pqe_jacobi["Npt_ev"].iloc[-1])

    df_sru_alt_in = pd.read_csv(f'summaries/summary_sru_alt_in_{sys_str}.dat', sep=r"\s+")
    sru_alt_in_dE.append(df_sru_alt_in['dE'].iloc[-1])
    sru_alt_in_Nres.append(df_sru_alt_in['Nrvec_ev'].iloc[-1])
    sru_alt_in_Npt.append(df_sru_alt_in['Npt_ev'].iloc[-1])

    df_sru_alt_out = pd.read_csv(f'summaries/summary_sru_alt_out_{sys_str}.dat', sep=r"\s+")
    sru_alt_out_dE.append(df_sru_alt_out['dE'].iloc[-1])
    sru_alt_out_Nres.append(df_sru_alt_out['Nrvec_ev'].iloc[-1])
    sru_alt_out_Npt.append(df_sru_alt_out['Npt_ev'].iloc[-1])

    df_sru_cyc_in = pd.read_csv(f'summaries/summary_sru_cyclic_in_{sys_str}.dat', sep=r"\s+")
    sru_cyc_in_dE.append(df_sru_cyc_in['dE'].iloc[-1])
    sru_cyc_in_Nres.append(df_sru_cyc_in['Nrvec_ev'].iloc[-1])
    sru_cyc_in_Npt.append(df_sru_cyc_in['Npt_ev'].iloc[-1])

    df_sru_cyc_out = pd.read_csv(f'summaries/summary_sru_cyclic_out_{sys_str}.dat', sep=r"\s+")
    sru_cyc_out_dE.append(df_sru_cyc_out['dE'].iloc[-1])
    sru_cyc_out_Nres.append(df_sru_cyc_out['Nrvec_ev'].iloc[-1])
    sru_cyc_out_Npt.append(df_sru_cyc_out['Npt_ev'].iloc[-1])

avg_vqe_jacobi_dE = np.mean(vqe_jacobi_dE)
avg_vqe_jacobi_Ngrad = np.mean(vqe_jacobi_Ngrad_ev)
avg_vqe_jacobi_Npt = np.mean(vqe_jacobi_Npt_ev)

avg_pqe_jacobi_dE = np.mean(pqe_jacobi_dE)
avg_pqe_jacobi_Nres = np.mean(pqe_jacobi_Nres_ev)
avg_pqe_jacobi_Npt = np.mean(pqe_jacobi_Npt_ev)

avg_sru_alt_in_dE = np.mean(sru_alt_in_dE)
avg_sru_alt_in_Nres = np.mean(sru_alt_in_Nres)
avg_sru_alt_in_Npt = np.mean(sru_alt_in_Npt)

avg_sru_alt_out_dE = np.mean(sru_alt_out_dE)
avg_sru_alt_out_Nres = np.mean(sru_alt_out_Nres)
avg_sru_alt_out_Npt = np.mean(sru_alt_out_Npt)

avg_sru_cyc_in_dE = np.mean(sru_cyc_in_dE)
avg_sru_cyc_in_Nres = np.mean(sru_cyc_in_Nres)
avg_sru_cyc_in_Npt = np.mean(sru_cyc_in_Npt)

avg_sru_cyc_out_dE = np.mean(sru_cyc_out_dE)
avg_sru_cyc_out_Nres = np.mean(sru_cyc_out_Nres)
avg_sru_cyc_out_Npt = np.mean(sru_cyc_out_Npt)

data = {
    "Optimizer": ["VQE Jacobi", "PQE Jacobi", "SRU Alternating In", "SRU Alternating Out", "SRU Cyclic In", "SRU Cyclic Out"],
    "ΔE0": [avg_vqe_jacobi_dE, avg_pqe_jacobi_dE, avg_sru_alt_in_dE, avg_sru_alt_out_dE, avg_sru_cyc_in_dE, avg_sru_cyc_out_dE],
    "Num Grad/Res Evals": [avg_vqe_jacobi_Ngrad, avg_pqe_jacobi_Nres, avg_sru_alt_in_Nres, avg_sru_alt_out_Nres, avg_sru_cyc_in_Nres, avg_sru_cyc_out_Nres],
    "Num Pauli Term Evals": [avg_vqe_jacobi_Npt, avg_pqe_jacobi_Npt, avg_sru_alt_in_Npt, avg_sru_alt_out_Npt, avg_sru_cyc_in_Npt, avg_sru_cyc_out_Npt]
}

df = pd.DataFrame(data)
print(df)