# comparing QNP_px on FCI and Fock computers
import qforte as qf
import numpy as np

def get_qubit_givens_pairs(ref):
    n_occ_spo = int(sum(ref) // 2)
    n_spo = int(len(ref) // 2)
    
    pairs = []
    
    for i in range(n_occ_spo):
        for a in range(n_occ_spo, n_spo):
            qa_pair = (2*i,2*a)
            qb_pair = (2*i+1, 2*a+1)
            pairs.append(qa_pair)
            pairs.append(qb_pair)
            
    return pairs
    

# Define the reference and geometry lists.
geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    # ('H', (0., 0., 5.0)), 
    # ('H', (0., 0., 6.0)),
    # ('H', (0., 0., 7.0)), 
    # ('H', (0., 0., 8.0)),
    # ('H', (0., 0., 9.0)), 
    # ('H', (0., 0.,10.0))
    ]

# Get the molecule object that now contains both the fermionic and qubit Hamiltonians.
mol = qf.system_factory(build_type='psi4', mol_geometry=geom, basis='sto-3g', run_fci=1)

print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref) # num electrons
sz = 0 # spin Z component
norb = int(len(ref) / 2) # number orbitals

print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")

q1 = 1 # source/control 1
q2 = 3 # target 1
q3 = 5 # source 2
q4 = 7 # target 2

theta = np.pi/2
qnp_px = qf.gate("QNP_PX", q1, q2, q3, q4, theta)

fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
fock_comp = qf.Computer(norb * 2)
graph = qf.FCIGraph(int(norb / 2), int(norb / 2), norb)

# reference = 'random'
reference = 'hf'

if(reference == 'hf'):
    fci_comp.hartree_fock()
    
    Uhf = qf.utils.state_prep.build_Uprep(ref, 'occupation_list')
    fock_comp.apply_circuit(Uhf)
    
elif(reference == 'random'):
    # FCI Computer case
    np.random.seed(42)
    random_array = np.random.rand(fci_comp.get_state().shape()[0], fci_comp.get_state().shape()[1])
    random = np.array(random_array, dtype = np.dtype(np.complex128))

    Crand = qf.Tensor(fci_comp.get_state().shape(), "Crand")
    Crand.fill_from_nparray(random.ravel(), Crand.shape())
    rand_nrm = Crand.norm()
    Crand.scale(1/rand_nrm)

    fci_comp.set_state(Crand)
    
    # FOCK Computer case
    state = []

    C = fci_comp.get_state()

    # maybe check for correct signs - if time
    for alpha_idx in range(graph.get_lena()):
        for beta_idx in range(graph.get_lenb()):

            coeff = C.get([alpha_idx, beta_idx])

            if abs(coeff) < 1e-12:
                continue

            alpha_bits = int(graph.get_astr_at_idx(alpha_idx))
            beta_bits = int(graph.get_bstr_at_idx(beta_idx))

            qb = qf.QubitBasis()

            for i in range(norb):
                qb.set_bit(2 * i,     bool((alpha_bits >> i) & 1))
                qb.set_bit(2 * i + 1, bool((beta_bits >> i) & 1))

            state.append((qb, coeff))

    fock_comp.set_state(state)
    
# # check prepared state
# print(fci_comp)
# print(fock_comp)

apply_many_qnps = False

if (apply_many_qnps):
    pairs = get_qubit_givens_pairs(ref)
    
    for pair in pairs:
        print(f"pair: {pair}")
        
        pair_givens = qf.gate("Givens", pair[1], pair[0], theta)
        
        fci_comp.apply_givens_rotation(pair[0], pair[1], theta)
        fock_comp.apply_gate(pair_givens)
else:
    fci_comp.apply_qnp_px(q1, q2, q3, q4, theta)
    fock_comp.apply_gate(qnp_px)

print(f"fci_comp^ {fci_comp}")
print(fock_comp)

do_phase_compare = True

tensor_diff = fock_comp.get_fci_tensor_diff(fci_comp, do_phase_compare)
print(f"tensor diff^ {tensor_diff}")

tensor_diff_norm = fock_comp.get_fci_tensor_diff_norm(fci_comp, do_phase_compare)
print(f"tensor diff norm: {tensor_diff_norm}")