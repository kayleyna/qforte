import qforte as qf
import numpy as np
import os
from pathlib import Path

molecules = {
    # "H4": {
    #     "r": 1,
    #     "symmetry": "d2h",
    #     "build_geom": lambda r: [
    #         ('H', (0., 0., i*r)) for i in range(4)
    #     ]
    # },

    # "H6": {
    #     "r": 1,
    #     "symmetry": "d2h",
    #     "build_geom": lambda r: [
    #         ('H', (0., 0., i*r)) for i in range(6)
    #     ]
    # },

    "N2": {
        "r": 1.75,
        "symmetry": "d2h",
        "build_geom": lambda r: [
            ('N',  (0., 0., -0.54855568036781*r)),
            ('N', (0., 0.,  0.54855568036781*r))
        ]
    },

    "LiH": {
        "r": 1.6,
        "symmetry": "c2v",
        "build_geom": lambda r: [
            ('Li', (0., 0., 0.0*r)),
            ('H',  (0., 0., 1.0*r))
        ]
    },

    "BeH2": {
        "r": 1.33,
        "symmetry": "d2h",
        "build_geom": lambda r: [
            ('H',  (0., 0., -1.0*r)),
            ('Be', (0., 0.,  0.0*r)),
            ('H',  (0., 0.,  1.0*r))
        ]
    },

    "H2O": {
        "r": 0.96,
        "symmetry": "c2v",
        "build_geom": lambda r: [
            ('O', (0., 0., 0.0*r)),
            ('H', ( 0.7907*r, 0., 0.6122*r)),
            ('H', (-0.7907*r, 0., 0.6122*r))
        ]
    },

    "HF": {
        "r": 0.92,
        "symmetry": "c2v",
        "build_geom": lambda r: [
            ('F', (0., 0., 0.0*r)),
            ('H', (0., 0., 1.0*r))
        ]
    },
}

# optimizers
methods = {
    # VQE
    # "VQE_jacobi": {
    #     "class": qf.UCCNVQE,
    #     "run_kwargs": {"optimizer": "jacobi"}
    # },

    # "VQE_CG": {
    #     "class": qf.UCCNVQE,
    #     "run_kwargs": {"optimizer": "cg"}
    # },

    # "VQE_POWELL": {
    #     "class": qf.UCCNVQE,
    #     "run_kwargs": {"optimizer": "powell"}
    # },

    # "VQE_BFGS": {
    #     "class": qf.UCCNVQE,
    #     "run_kwargs": {"optimizer": "bfgs"}
    # },

    # "VQE_SLSQP": {
    #     "class": qf.UCCNVQE,
    #     "run_kwargs": {"optimizer": "slsqp"}
    # },

    # "VQE_l-bfgs-b": {
    #     "class": qf.UCCNVQE,
    #     "run_kwargs": {"optimizer": "l-bfgs-b"}
    # },

    # more to add, but testing code

    # PQE
    # "PQE_jacobi": {
    #     "class": qf.UCCNPQE,
    #     "run_kwargs": {"optimizer": "jacobi"}
    # },

    # SRU Variants (PQE)
    "SRU_cyclic_in": {
        "class": qf.UCCNPQE,
        "run_kwargs": {
            "optimizer": "sru",
            "sru_type": "inside",
            "sru_sweep_type": "cyclic"
        }
    },

    # "SRU_cyclic_out": {
    #     "class": qf.UCCNPQE,
    #     "run_kwargs": {
    #         "optimizer": "sru",
    #         "sru_type": "outside",
    #         "sru_sweep_type": "cyclic"
    #     }
    # },

    # "SRU_alternating_in": {
    #     "class": qf.UCCNPQE,
    #     "run_kwargs": {
    #         "optimizer": "sru",
    #         "sru_type": "inside",
    #         "sru_sweep_type": "alternating"
    #     }
    # },

    # "SRU_alternating_out": {
    #     "class": qf.UCCNPQE,
    #     "run_kwargs": {
    #         "optimizer": "sru",
    #         "sru_type": "outside",
    #         "sru_sweep_type": "alternating"
    #     }
    # },

    # "SRU_random_in": {
    #     "class": qf.UCCNPQE,
    #     "run_kwargs": {
    #         "optimizer": "sru",
    #         "sru_type": "inside",
    #         "sru_sweep_type": "random"
    #     }
    # },

    # "SRU_random_out": {
    #     "class": qf.UCCNPQE,
    #     "run_kwargs": {
    #         "optimizer": "sru",
    #         "sru_type": "outside",
    #         "sru_sweep_type": "random"
    #     }
    # },

    # "SRU_greedy_in": {
    #     "class": qf.UCCNPQE,
    #     "run_kwargs": {
    #         "optimizer": "sru",
    #         "sru_type": "inside",
    #         "sru_sweep_type": "greedy"
    #     }
    # },

    # "SRU_greedy_out": {
    #     "class": qf.UCCNPQE,
    #     "run_kwargs": {
    #         "optimizer": "sru",
    #         "sru_type": "outside",
    #         "sru_sweep_type": "greedy"
    #     }
    # }
}

# automatic data collection
results = []
timer = qf.local_timer()

script_dir = Path(__file__).resolve().parent

base_dir = script_dir / "summaries"
base_dir.mkdir(exist_ok=True)

original_dir = Path.cwd()

for mol_name, mol_data in molecules.items():
    r = mol_data["r"]
    geom = mol_data["build_geom"](r)

    for method_name, method_data in methods.items():
        print(f"\nRunning {mol_name} | r = {r} | {method_name}")

        run_dir = base_dir / f"{mol_name}_r{r}" / method_name
        run_dir.mkdir(parents=True, exist_ok=True)

        os.chdir(run_dir)

        timer.reset()
        mol = qf.system_factory(
            build_type='psi4', 
            mol_geometry=geom, 
            symmetry=mol_data["symmetry"],
            basis='sto-3g',
            run_fci=1,
            charge=mol_data.get("charge", 0),
            multiplicity=mol_data.get("multiplicity", 1))
        timer.record("mol build")

        Efci = mol.fci_energy

        timer.reset()
        alg_fci = method_data["class"](
            mol,
            computer_type = 'fci',
            verbose=False
        )
        timer.record("alg fci setup")

        timer.reset()
        
        generator = 'qnp' # ucc or qnp
        
        alg_fci.run(
            opt_thresh=1.0e-12,
            pool_type='SD',
            generator_type=generator,
            **method_data["run_kwargs"]
        )
        timer.record("alg fci run")

        E = alg_fci._Egs
        dE0 = E - Efci

        # rename data files for specific method/molecule
        data_file = Path("data.dat")

        if data_file.exists():
            prefix = generator.upper()   # "UCC" or "QNP"

            new_name = (
                f"{prefix}summary_{mol_name}_r{r}_{method_name}.dat"
            )

            data_file.rename(new_name)
        else:
            print("WARNING: data.dat not found for", method_name)

        results.append({
            "molecule": mol_name,
            "r": r,
            "method": method_name,
            "E": E,
            "Efci": Efci,
            "E_diff": dE0
        })

        print(f' Efci:    {mol.fci_energy:+12.10f}')
        print(f' Edif:    {E - mol.fci_energy:+12.10f}')

        print("\n Total Script Time \n")
        print(timer)

        os.chdir(original_dir)

print("\nAll runs complete")