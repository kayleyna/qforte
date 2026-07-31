import qforte as qf
import numpy as np
r = 1.0

geom = [
    ('H', (0., 0., 0.0*r)), 
    ('H', (0., 0., 1.0*r)),
    ('H', (0., 0., 2.0*r)),
    ('H', (0., 0., 3.0*r))
]

timer = qf.local_timer()

timer.reset()
mol = qf.system_factory(
    build_type='psi4', 
    mol_geometry=geom, 
    symmetry='d2h',
    basis='sto-3g',
    run_fci=1)
timer.record("mol build")

# alg_fci1 uses UCC matrix by design
# timer.reset()
# alg_fci1 = qf.UCCNPQE(
#     mol,
#     computer_type = 'fci',
#     verbose=False)
# timer.record("alg setup fci")

# timer.reset()
# alg_fci1.run(
#     opt_thresh=1.0e-8, 
#     pool_type='SD',
#     optimizer = 'jacobi',
#     # sru_type = 'inside',
#     # sru_sweep_type='cyclic',
#     )
# timer.record("run alg fci")

# print(f' Efci:    {mol.fci_energy:+12.10f}')
# print(f' Edif:    {alg_fci1._Egs - mol.fci_energy:+12.10f}')

# print("\n Total Script Time \n")
# print(timer)

# # below figure out how to create QNPor and QNPpx unitary matrices to try instead of UCC
# timer.reset()
# alg_fci2 = qf.UCCNPQE(
#     mol,
#     computer_type = 'fci',
#     verbose=False)
# timer.record("alg setup fci")

# timer.reset()
# alg_fci1.run(
#     opt_thresh=1.0e-8, 
#     pool_type='SD',
#     optimizer = 'jacobi',
#     # sru_type = 'inside',
#     # sru_sweep_type='cyclic',
#     )
# timer.record("run alg fci")

# print(f' Efci:    {mol.fci_energy:+12.10f}')
# print(f' Edif:    {alg_fci2._Egs - mol.fci_energy:+12.10f}')

# print("\n Total Script Time \n")
# print(timer)


# def qnp_or(theta):
#     control_idx_1 = 1
#     target_idx_1 = 3
#     control_idx_2 = 2
#     target_idx_2 = 4

#     givens_gate_1 = qf.gate('G', target_idx_1, control_idx_1, theta)
#     givens_gate_2 = qf.gate('G', target_idx_2, control_idx_2, theta)

#     circ = qf.Circuit()
#     circ.add_gate(givens_gate_1)
#     circ.add_gate(givens_gate_2)

#     return circ

def qnp_or(theta):
    control_idx = [1, 2]
    target_idx = [3, 4]

    qnp_or_circ = qf.Circuit()

    for i in range(len(control_idx)):
        hadamard = qf.gate('H', control_idx[i])
        cX_gate = qf.gate('cX', target_idx[i], control_idx[i])
        Ry_c = qf.gate('Ry', control_idx[i], theta)
        Ry_t = qf.gate('Ry', target_idx[i], theta)

        givens_circ = qf.Circuit()
        givens_circ.add_gate(hadamard)
        givens_circ.add_gate(cX_gate)
        givens_circ.add_gate(Ry_c)
        givens_circ.add_gate(Ry_t)
        givens_circ.add_gate(cX_gate)
        givens_circ.add_gate(hadamard)

        # add circuit to circuit??


qnp_or(np.pi/2)

# def qnp_px(theta):