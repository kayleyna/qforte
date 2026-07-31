import copy
import numpy as np
import random
from scipy.optimize import minimize

def diis(diis_max_dim, t_diis, e_diis):
    """This function implements the direct inversion of iterative subspace
    (DIIS) convergence accelerator. Draws heavy insiration from Daniel
    Smith's ccsd_diss.py code in psi4 numpy
    """

    if len(t_diis) > diis_max_dim:
        del t_diis[0]
        del e_diis[0]

    diis_dim = len(t_diis) - 1

    # Construct diis B matrix (following Crawford Group github tutorial)
    B = np.ones((diis_dim+1, diis_dim+1)) * -1
    bsol = np.zeros(diis_dim+1)

    B[-1, -1] = 0.0
    bsol[-1] = -1.0
    for i, ei in enumerate(e_diis):
        for j, ej in enumerate(e_diis):
            B[i,j] = np.dot(np.real(ei), np.real(ej))

    B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

    x = np.linalg.lstsq(B, bsol, rcond=None)[0][:-1]

    t_new = np.zeros(( len(t_diis[0]) ))
    for l in range(diis_dim):
        temp_ary = x[l] * np.asarray(t_diis[l+1])
        t_new = np.add(t_new, temp_ary)

    return copy.deepcopy(list(np.real(t_new)))

def jacobi_solver(self):
    """
    This function minimizes the norm of the residual/gradient vector
    by using a quasi-Newton update procedure for the amplitudes
    """

    sys_str = "HF"

    t_diis = [copy.deepcopy(self._tamps)]
    e_diis = []
    Ek0 = self.energy_feval(self._tamps)

    if self.__class__.__name__ in ['UCCNPQE', 'SPQE']:
        print('\n    k iteration         Energy               dE           Nrvec ev      Nrm ev*        Npt ev            ||r||')

        self._print_summary_file = True
        if (self._print_summary_file):
                f = open("summary_pqejacobi.dat", "w+", buffering=1)
                f.write('k_iteration        Energy              dE              Nrvec_ev    Nrm_ev*        Npt_ev            ||r||')
                # f.write('\n#--------------------------------------------------------------------------------------------------')
                f.close()

    elif self.__class__.__name__ in ['UCCNVQE', 'ADAPTVQE']:
        print('\n    k iteration         Energy               dE           Ngvec ev      Ngm ev*        Npt ev            ||g||')

        self._print_summary_file = True
        if (self._print_summary_file):
            f = open("summary_vqejacobi.dat", "w+", buffering=1)
            f.write('k_iteration        Energy              dE              Ngvec_ev    Ngm_ev*        Npt_ev            ||g||')
            # f.write('\n#--------------------------------------------------------------------------------------------------')
            f.close()
    print('-------------------------------------------------------------------------------------------------------------------', flush=True)

    energy_change = []
    for k in range(1, self._opt_maxiter+1):

        t_old = copy.deepcopy(self._tamps)

        #do regular update
        if self.__class__.__name__ in ['UCCNPQE', 'SPQE']:
            r_k = self.get_residual_vector(self._tamps)
        elif self.__class__.__name__ in ['UCCNVQE', 'ADAPTVQE']:
            r_k = self.gradient_ary_feval(self._tamps)
        r_k = self.get_res_over_mpdenom(r_k)

        self._tamps = list(np.add(self._tamps, r_k))

        Ek = self.energy_feval(self._tamps)
        dE = Ek - Ek0
        Ek0 = Ek

        energy_change.append(dE)

        n_pt_measures = 2*self._Nl*k*len(self._tamps) + self._Nl*k

        if self.__class__.__name__ in ['UCCNPQE', 'SPQE']:
            print(f'     {k:7}        {Ek:+12.10f}      {dE:+12.10f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {n_pt_measures:10}        {self._res_vec_norm:+12.10f}')

            if (self._print_summary_file):
                f = open("summary_pqejacobi.dat", "a", buffering=1)
                f.write(f'\n{k:7}       {Ek:+12.10f}        {dE:+12.10f}        {self._res_vec_evals:4}     {self._res_m_evals:6}       {n_pt_measures:10}        {self._res_vec_norm:+12.10f}')
                f.close()

            # if (self._res_vec_norm < self._opt_thresh):
            #     self._Egs = Ek
            #     break

            if (abs(dE) < self._opt_thresh):
                self._Egs = Ek
                break

        elif self.__class__.__name__ in ['UCCNVQE', 'ADAPTVQE']:
            print(f'     {k:7}        {Ek:+12.10f}      {dE:+12.10f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {n_pt_measures:10}        {self._curr_grad_norm:+12.10f}')

            if (self._print_summary_file):
                f = open("summary_vqejacobi.dat", "a", buffering=1)
                f.write(f'\n{k:7}        {Ek:+12.10f}      {dE:+12.12f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {n_pt_measures:10}        {self._curr_grad_norm:+12.12f}')
                f.close()

            # if(self._curr_grad_norm < self._opt_thresh): # write in kwarg for using grad norm/dE
            #     self._Egs = Ek
            #     break

            if (abs(dE) < self._opt_thresh):
                self._Egs = Ek
                break

        t_diis.append(copy.deepcopy(self._tamps))
        e_diis.append(np.subtract(copy.deepcopy(self._tamps), t_old))

        if(k >= 1 and self._diis_max_dim >= 2):
            self._tamps = diis(self._diis_max_dim, t_diis, e_diis)

    self._Egs = Ek
    self._dEs = energy_change
    if k == self._opt_maxiter:
        print("\nMaximum number of Jacobi iterations reached!")
    if hasattr(self, '_energies'):
        self._energies.append(Ek)
    if hasattr(self, '_n_classical_params'):
        self._n_classical_params = len(self._tamps)
    if hasattr(self, '_n_pauli_measures_k'):
        self._n_pauli_measures_k += self._Nl*k * (2*len(self._tamps) + 1)
    if hasattr(self, '_n_pauli_trm_measures'):
        self._n_pauli_trm_measures += 2*self._Nl*k*len(self._tamps) + self._Nl*k
    if hasattr(self, '_n_pauli_trm_measures_lst'):
        self._n_pauli_trm_measures_lst.append(self._n_pauli_measures_k)
    if hasattr(self, '_n_cnot'):
        if(self._computer_type == 'fock'):
            self._n_cnot = self.build_Uvqc().get_num_cnots()
        else:
            # TODO: Build resource estimator
            self._n_cnot = 'N/A'
    if hasattr(self, '_n_cnot_lst'):
        if(self._computer_type == 'fock'):
            self._n_cnot_lst.append(self.build_Uvqc().get_num_cnots())
        else:
            # TODO: Build resource estimator
            self._n_cnot_lst.append('N/A')

def subspace_rotation_solver(self):
    """
    This function minimizes the norm of the residual vector
    by using a 2x2 Jacobi rotation update procedure for the amplitudes
    """

    sys_str = "H8"
    sweep_str = "alt"
    in_out = "in"

    t_diis = [copy.deepcopy(self._tamps)]
    e_diis = []
    Ek0 = self.energy_feval(self._tamps)


    if self.__class__.__name__ in ['UCCNPQE', 'SPQE']:
        print('\n    k iteration         Energy               dE           Nrvec ev      Nrm ev*        Npt ev            ||r||')

        self._print_summary_file = True
        if (self._print_summary_file):
                f = open("summary_sru.dat", "w+", buffering=1)
                f.write('k_iteration        Energy              dE              Nrvec_ev    Nrm_ev*        Npt_ev            ||r||')
                # f.write('\n#--------------------------------------------------------------------------------------------------')
                f.close()

        print('-------------------------------------------------------------------------------------------------------------------', flush=True)
    else: 
        raise ValueError('subspace_rotation_solver is not implemented for this algorithm.')

    energy_change = []

    random.seed(5)
    
    for k in range(1, self._opt_maxiter+1):
        t_old = copy.deepcopy(self._tamps)

        if self.__class__.__name__ in ['UCCNPQE', 'SPQE']:
            r_k = self.get_residual_vector(self._tamps, inc_nres_vec=False)
        else:
            raise ValueError('subspace_rotation_solver is not implemented for this algorithm.')
            
        delta_tmu = []
        r = []

        # c0 = self.energy_feval(self._tamps) # c0 = Eo
        # print(f"c_o {c_0} E0 {E0}")

        if self._sru_type.lower() == 'outside': # parameters calculated outside the sweep loop
            c0_out = self.energy_feval(self._tamps) 
            c_mu_out = np.zeros(len(r_k), dtype=np.complex128)

            for mu in range(len(r_k)):
                c_mu_out[mu] = self.get_cmu_fci(mu, self._tamps)
        
            if self._sru_sweep_type.lower() == 'cyclic':
                c0 = c0_out
                c_mu = c_mu_out
                r = r_k

                for mu in range(len(self._tamps)):
                    delta_tmu_R = 0.5 * np.arctan((2 * r_k[mu]) / (c0 - c_mu[mu]))
                    delta_tmu.append(delta_tmu_R)
                    self._tamps[mu] += delta_tmu_R
            elif self._sru_sweep_type.lower() == 'alternating':
                c0 = c0_out
                if k % 2 == 0:
                    c_mu = c_mu_out
                    r = r_k

                    for mu in range(len(self._tamps)):
                        delta_tmu_R = 0.5 * np.arctan((2 * r_k[mu]) / (c0 - c_mu[mu]))
                        delta_tmu.append(delta_tmu_R)
                        self._tamps[mu] += delta_tmu_R
                else:
                    c_mu = list(reversed(c_mu_out))
                    r = list(reversed(r_k))
                    mus = list(reversed(range(len(self._tamps))))

                    for mu in range(len(self._tamps)):
                        delta_tmu_R = 0.5 * np.arctan((2 * r[mu]) / (c0 - c_mu[mu]))
                        delta_tmu.append(delta_tmu_R)
                        self._tamps[mus[mu]] += delta_tmu_R
            elif self._sru_sweep_type.lower() == 'random':
                c0 = c0_out
                c_mu = c_mu_out
                r = r_k
                mus = list(range(len(self._tamps)))
              
                x = list(zip(c_mu, r, mus))
                random.shuffle(x)
                c_mu, r, mus = zip(*x)
              
                for mu in range(len(self._tamps)):
                    delta_tmu_R = 0.5 * np.arctan((2 * r[mu]) / (c0 - c_mu[mu]))
                    delta_tmu.append(delta_tmu_R)
                    self._tamps[mus[mu]] += delta_tmu_R
            elif self._sru_sweep_type.lower() == 'greedy':
                c0 = c0_out
                c_mu = np.real(c_mu_out)
                r = np.real(r_k)
                # c_mu.sort()
                # r.sort()
                mus_ascending = sorted(range(len(np.abs(r))), key=lambda i: np.real(r[i]))
              
                for mu in mus_ascending:
                    delta_tmu_R = 0.5 * np.arctan((2 * r[mu]) / (c0 - c_mu[mu]))
                    delta_tmu.append(delta_tmu_R)
                    self._tamps[mu] += delta_tmu_R
        elif self._sru_type.lower() == 'inside': # parameters calculated inside sweep loop
            if self._sru_sweep_type.lower() == 'cyclic':
                for mu in range(len(self._tamps)):
                    c0 = self.energy_feval(self._tamps) # c0 = E0
                    c_mu = self.get_cmu_fci(mu, self._tamps)
                    r_k_cyc = self.get_residual_vector(self._tamps, inc_nres_vec=False)
                    rmu = r_k_cyc[mu] 
                    r.append(rmu)

                    delta_tmu_R = 0.5 * np.arctan((2 * rmu) / (c0 - c_mu))
                    delta_tmu.append(delta_tmu_R)
                    self._tamps[mu] += delta_tmu_R
            elif self._sru_sweep_type.lower() == 'alternating':
                if k % 2 == 0:
                    for mu in range(len(self._tamps)):
                        c0 = self.energy_feval(self._tamps)
                        c_mu = self.get_cmu_fci(mu, self._tamps)
                        r_k_alt = self.get_residual_vector(self._tamps, inc_nres_vec=False)
                        rmu = r_k_alt[mu]
                        r.append(rmu)

                        delta_tmu_R = 0.5 * np.arctan((2 * rmu) / (c0 - c_mu))
                        delta_tmu.append(delta_tmu_R)
                        self._tamps[mu] += delta_tmu_R
                else:
                    mus = range(len(self._tamps))
                    for mu in reversed(mus):
                        c0 = self.energy_feval(self._tamps)
                        c_mu = self.get_cmu_fci(mu, self._tamps)
                        r_k_alt = self.get_residual_vector(self._tamps, inc_nres_vec=False)
                        rmu = r_k_alt[mu]
                        r.append(rmu)

                        delta_tmu_R = 0.5 * np.arctan((2 * rmu) / (c0 - c_mu))
                        delta_tmu.append(delta_tmu_R)
                        self._tamps[mu] += delta_tmu_R
            elif self._sru_sweep_type.lower() == 'random':
                mus = list(range(len(self._tamps)))
                random.shuffle(mus)
              
                for mu in mus:
                    c0 = self.energy_feval(self._tamps)
                    c_mu = self.get_cmu_fci(mu, self._tamps)
                    r_k_rand = self.get_residual_vector(self._tamps, inc_nres_vec=False)
                    rmu = r_k_rand[mu]
                    r.append(rmu)
                  
                    delta_tmu_R = 0.5 * np.arctan((2 * rmu) / (c0 - c_mu))
                    delta_tmu.append(delta_tmu_R)
                    self._tamps[mu] += delta_tmu_R
            elif self._sru_sweep_type.lower() == 'greedy':
                mus_ascending = sorted(range(len(np.abs(np.real(r_k)))), key=lambda i: np.real(r_k[i]))
                for mu in mus_ascending:
                    r_k_greedy = self.get_residual_vector(self._tamps, inc_nres_vec=False)
                    c0 = self.energy_feval(self._tamps)
                    c_mu = self.get_cmu_fci(mu, self._tamps)
                    rmu = r_k_greedy[mu]
                    r.append(rmu)

                    delta_tmu_R = 0.5 * np.arctan((2 * rmu) / (c0 - c_mu))
                    delta_tmu.append(delta_tmu_R)
                    self._tamps[mu] += delta_tmu_R

        self._res_vec_norm = np.linalg.norm(r)
        self._res_vec_evals += 1
        self._res_m_evals += len(self._tamps)

        Ek = self.energy_feval(self._tamps)
        dE = Ek - Ek0
        Ek0 = Ek

        energy_change.append(dE)

        n_pt_measures = 2*self._Nl*k*len(self._tamps) + self._Nl*k

        if self.__class__.__name__ in ['UCCNPQE', 'SPQE']:
            print(f'     {k:7}        {Ek:+12.10f}      {dE:+12.10f}      {self._res_vec_evals:4}        {self._res_m_evals:6}       {n_pt_measures:10}        {self._res_vec_norm:+12.10f}')

            if (self._print_summary_file):
                f = open("summary_sru.dat", "a", buffering=1)
                f.write(f'\n{k:7}       {Ek:+12.10f}        {dE:+12.10f}        {self._res_vec_evals:4}     {self._res_m_evals:6}       {n_pt_measures:10}        {self._res_vec_norm:+12.10f}')
                f.close()

            # if(self._res_vec_norm < self._opt_thresh):
            #     self._Egs = Ek
            #     break

            if (abs(dE) < self._opt_thresh):
                self._Egs = Ek
                break

        t_diis.append(copy.deepcopy(self._tamps))
        e_diis.append(np.subtract(copy.deepcopy(self._tamps), t_old))

        if(k >= 1 and self._diis_max_dim >= 2):
            self._tamps = diis(self._diis_max_dim, t_diis, e_diis)

    self._Egs = Ek
    self._dEs = energy_change
    if k == self._opt_maxiter:
        print("\nMaximum number of Jacobi iterations reached!")
    if hasattr(self, '_energies'):
        self._energies.append(Ek)
    if hasattr(self, '_n_classical_params'):
        self._n_classical_params = len(self._tamps)
    if hasattr(self, '_n_pauli_measures_k'):
        self._n_pauli_measures_k += self._Nl*k * (2*len(self._tamps) + 1)
    if hasattr(self, '_n_pauli_trm_measures'):
        self._n_pauli_trm_measures += 2*self._Nl*k*len(self._tamps) + self._Nl*k
    if hasattr(self, '_n_pauli_trm_measures_lst'):
        self._n_pauli_trm_measures_lst.append(self._n_pauli_measures_k)
    if hasattr(self, '_n_cnot'):
        if(self._computer_type == 'fock'):
            self._n_cnot = self.build_Uvqc().get_num_cnots()
        else:
            # TODO: Build resource estimator
            self._n_cnot = 'N/A'
    if hasattr(self, '_n_cnot_lst'):
        if(self._computer_type == 'fock'):
            self._n_cnot_lst.append(self.build_Uvqc().get_num_cnots())
        else:
            # TODO: Build resource estimator
            self._n_cnot_lst.append('N/A')

def scipy_solver(self, function_to_minimize):

    # Construct arguments to hand to the minimizer.
    opts = {}

    # Options common to all minimization algorithms
    opts['disp'] = True
    opts['maxiter'] = self._opt_maxiter

    # Optimizer-specific options
    if self._optimizer.lower() in ['bfgs', 'cg', 'l-bfgs-b']:
        opts['gtol'] = self._opt_thresh
    if self._optimizer.lower() == ['nelder-mead']:
        opts['fatol'] = self._opt_thresh
        opts['adaptive'] = True
    if self._optimizer.lower() in ['l-bfgs-b', 'slsqp']:
        opts['ftol'] = self._opt_thresh
    if self._optimizer.lower() in ['powell']:
        opts['ftol'] = self._opt_thresh
        opts['xtol'] = self._opt_thresh

    x0 = copy.deepcopy(self._tamps)
    self._prev_energy = self.energy_feval(x0)
    self._k_counter = 0

    res = minimize(function_to_minimize, x0,
            method=self._optimizer,
            options=opts,
            callback=self.report_iteration)
    
    # n_pt_measures = 2*self._Nl*self._k_counter*len(self._tamps) + self._Nl*self._k_counter

    if(res.success):
        print('  => Minimization successful!')
    else:
        print('  => WARNING: minimization result may not be tightly converged.')

    self._tamps = list(res.x)
    self._Egs = self.energy_feval(self._tamps)
    if hasattr(self, '_energies'):
        self._energies.append(self._Egs)
    if hasattr(self, '_n_classical_params'):
        self._n_classical_params = len(self._tamps)
    if hasattr(self, '_n_pauli_measures_k'):
        self._n_pauli_measures_k += self._Nl*self._k_counter * (2*len(self._tamps) + 1)
    if hasattr(self, '_n_pauli_trm_measures'):
        self._n_pauli_trm_measures += 2*self._Nl*self._k_counter*len(self._tamps) + self._Nl*self._k_counter
    if hasattr(self, '_n_pauli_trm_measures_lst'):
        self._n_pauli_trm_measures_lst.append(self._n_pauli_measures_k)
    if hasattr(self, '_n_cnot'):
        self._n_cnot = self.build_Uvqc().get_num_cnots()
    if hasattr(self, '_n_cnot_lst'):
        self._n_cnot_lst.append(self.build_Uvqc().get_num_cnots())
