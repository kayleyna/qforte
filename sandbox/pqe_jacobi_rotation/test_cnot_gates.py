import qforte as qf
import numpy as np
from qforte.utils.trotterization import trotterize

molecules = {
    "H4": {
        "r": 1,
        "symmetry": "d2h",
        "build_geom": lambda r: [
            ('H', (0., 0., i*r)) for i in range(4)
        ]
    },

    # "H6": {
    #     "r": 1,
    #     "symmetry": "d2h",
    #     "build_geom": lambda r: [
    #         ('H', (0., 0., i*r)) for i in range(6)
    #     ]
    # },

    # "N2": {
    #     "r": 1.75,
    #     "symmetry": "d2h",
    #     "build_geom": lambda r: [
    #         ('N',  (0., 0., -0.54855568036781*r)),
    #         ('N', (0., 0.,  0.54855568036781*r))
    #     ]
    # },

    # "LiH": {
    #     "r": 1.6,
    #     "symmetry": "c2v",
    #     "build_geom": lambda r: [
    #         ('Li', (0., 0., 0.0*r)),
    #         ('H',  (0., 0., 1.0*r))
    #     ]
    # },

    # "BeH2": {
    #     "r": 1.33,
    #     "symmetry": "d2h",
    #     "build_geom": lambda r: [
    #         ('H',  (0., 0., -1.0*r)),
    #         ('Be', (0., 0.,  0.0*r)),
    #         ('H',  (0., 0.,  1.0*r))
    #     ]
    # },

    # "H2O": {
    #     "r": 0.96,
    #     "symmetry": "c2v",
    #     "build_geom": lambda r: [
    #         ('O', (0., 0., 0.0*r)),
    #         ('H', ( 0.7907*r, 0., 0.6122*r)),
    #         ('H', (-0.7907*r, 0., 0.6122*r))
    #     ]
    # },

    # "HF": {
    #     "r": 0.92,
    #     "symmetry": "c2v",
    #     "build_geom": lambda r: [
    #         ('F', (0., 0., 0.0*r)),
    #         ('H', (0., 0., 1.0*r))
    #     ]
    # },
}

for mol_name, mol_data in molecules.items():
    r = mol_data["r"]
    geom = mol_data["build_geom"](r)

    print(f"\nRunning {mol_name} | r = {r}")


    mol = qf.system_factory(
        build_type='psi4', 
        mol_geometry=geom, 
        symmetry=mol_data["symmetry"],
        basis='sto-3g',
        run_fci=1,
        charge=mol_data.get("charge", 0),
        multiplicity=mol_data.get("multiplicity", 1))

print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref) # num electrons
sz = 0 # spin Z component
norb = int(len(ref) / 2) # number orbitals

print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")

fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fock_comp = qf.Computer(norb * 2)
graph = qf.FCIGraph(int(norb / 2), int(norb / 2), norb)

reference = 'hf'

fci_comp.hartree_fock()

# ===> evovle pool basic <====

ah = True
adj = False

print("\n Time Evo Settings")
print("===========================")
print(f" time:      {1.0}")
print(f" r:         {1}")
print(f" order:     {1}")
print(f" antiherm:  {ah}")
print(f" adjoint:   {adj}")
print("\n")

sd_pool = qf.SQOpPool()
sd_pool.set_orb_spaces(ref)
sd_pool.fill_pool("SD")
print(sd_pool)

A = sd_pool.get_qubit_operator('commuting_grp_lex', qubit_excitations = False)

U, phase1 = trotterize(A, trotter_number=1)
print(f"Num CNOT gates in UCC circuit: {U.get_num_cnots()}")

# print("\nEntire UCC circuit")
# print("==================")

# for i, gate in enumerate(U.gates()):
#     print(f"{i}: {gate}")

# write for loop over operators and add up each count
tot_cnot_UCC = 0
for coeff, op in sd_pool.terms():
    tot_cnot_UCC += op.count_cnot_for_exponential_full()
print(f"Num CNOT gates in UCC circuit: {tot_cnot_UCC}")

tot_cnot_QNP = 0
for coeff, op in sd_pool.terms():
    tot_cnot_QNP += op.count_cnot_for_qnp()
print(f"Num CNOT gates in Givens/QNP circuit: {tot_cnot_QNP}")
