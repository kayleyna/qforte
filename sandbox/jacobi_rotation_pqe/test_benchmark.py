import qforte as qf
import numpy as np
import os
from pathlib import Path

molecules = {
    # "H8": {
    #     "r": 1,
    #     "symmetry": "d2h",
    #     "build_geom": lambda r: [
    #         ('H', (0., 0., i*r)) for i in range(8)
    #     ]
    # },

    # "H12": {
    #     "r": 1,
    #     "symmetry": "d2h",
    #     "build_geom": lambda r: [
    #         ('H', (0., 0., i*r)) for i in range(12)
    #     ]
    # },

    # "H12": {
    #     "r": 1.5,
    #     "symmetry": "d2h",
    #     "build_geom": lambda r: [
    #         ('H', (0., 0., i*r)) for i in range(12)
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

    # slim molecules

#    "LiH": {
#        "r": 1,
#        "symmetry": "c2v", 
#        "build_geom": lambda r: [
#            ('Li',  (0., 0., -0.79739733931541*r)),
#            ('H',  (0., 0., 0.79739733931541*r))
#        ]
#    },

#    "FNH3": {
#        "r": 1,
#        "symmetry": "c1",
#        "charge": 0,
#        "multiplicity": 2,
#        "build_geom": lambda r: [
#            ('F',  (-0.20511592550368*r, 1.19194565532739*r, -0.69629008079903*r)),
#            ('N',  (-1.07146811414627*r, -0.00912579911730*r, 1.50279445312235*r)),
#            ('H',  (-0.61757273596162*r, -0.91698370489959*r, 1.47414850528117*r)),
#            ('H',  (-0.60090008716373*r, 0.48687620550560*r, 2.25355568011495*r)),
#            ('H', (-0.76307211569594*r, 0.48318494574625*r,  0.60817262672433*r))
#        ]
#    },

   "FOH2": {
       "r": 1,
       "symmetry": "c1",
        "charge": 0,
       "multiplicity": 1,
       "build_geom": lambda r: [
           ('F',  (-1.23638891242469*r, 0.01223899892089*r, 0.0*r)),
           ('O',  (1.19756591471928*r, -0.10808699225716*r, 0.0*r)),
           ('H',  (1.41539689966303*r, 0.82701394114129*r, 0.0*r)),
           ('H', (0.13482999027236*r, -0.08437799420910*r,  0.0*r))
       ]
   },

#    "ClNH3": {
#        "r": 1,
#        "symmetry": "c1",
#        "charge": 0,
#        "multiplicity": 2,
#        "build_geom": lambda r: [
#            ('Cl',  (-0.05269979638981*r, 1.43421208803173*r, -1.14799698841848*r)),
#            ('N',  (-1.10346055168117*r, -0.08276287392249*r, 1.64126467353726*r)),
#            ('H',  (-0.60746386687678*r, -0.96403353177218*r, 1.55976716921984*r)),
#            ('H',  (-0.58603408852298*r, 0.44582925854889*r, 2.33563922438047*r)),
#            ('H', (-0.90847067552967*r, 0.40265237120158*r,  0.75370710625386*r))
#        ]
#    },

#    "ClOH2": {
#        "r": 1,
#        "symmetry": "cs",
#        "charge": 0,
#        "multiplicity": 1,
#        "build_geom": lambda r: [
#            ('Cl',  (-1.35104290390905*r, 0.03529399743754*r, 0.0*r)),
#            ('O',  (1.74135387616602*r, -0.08657199361778*r, 0.0*r)),
#            ('H',  (1.82908987020235*r, 0.87012993818364*r, 0.0*r)),
#            ('H', (0.75492994641567*r, -0.17939998742839*r,  0.0*r))
#        ]
#    },

#    "ClFH": {
#        "r": 1,
#        "symmetry": "c1",
#        "charge": 0,
#        "multiplicity": 2,
#        "build_geom": lambda r: [
#            ('Cl',  (-5.21765810957416*r, 1.03354112670284*r, 0.01665424877709*r)),
#            ('F',  (-2.34796102313037*r, 1.10896776135797*r, -0.00280929968149*r)),
#            ('H', (-3.32450095394741*r, 1.08822231285660*r,  0.00254397975313*r))
#        ]
#    },

#    "HCl2": {
#        "r": 1,
#        "symmetry": "c1",
#        "charge": 0,
#        "multiplicity": 2,
#        "build_geom": lambda r: [
#            ('Cl',  (-1.08432462318399*r, -0.37737494323059*r, 0.80154741328151*r)),
#            ('H',  (-0.91892469462938*r, 0.27674120031814*r, -0.59444767755731*r)),
#            ('Cl', (-0.75489008614226*r, 0.93320937358275*r,  -1.99831438788966*r))
#        ]
#    },

#    "HFN3": {
#        "r": 1,
#        "symmetry": "c1",
#        "charge": 0,
#        "multiplicity": 2,
#        "build_geom": lambda r: [
#            ('N',  (-6.06210715973508*r, 1.07714819331761*r, 0.12402690101301*r)),
#            ('N',  (-4.94801505846677*r, 1.43880826780977*r, 0.40939130116296*r)),
#            ('N',  (-7.18169604006818*r, 0.75794177597438*r, -0.14863492933740*r)),
#            ('F',  (-2.79019322181358*r, 0.08212056391567*r, 0.39201212194111*r)),
#            ('H', (-3.63394740197038*r, 0.58534223853320*r,  0.39040002219802*r))
#        ]
#    },
}

# optimizers
methods = {
    # VQE
    "VQE_jacobi": {
        "class": qf.UCCNVQE,
        "run_kwargs": {"optimizer": "jacobi"}
    },

    "VQE_CG": {
        "class": qf.UCCNVQE,
        "run_kwargs": {"optimizer": "cg"}
    },

    # "VQE_POWELL": {
    #     "class": qf.UCCNVQE,
    #     "run_kwargs": {"optimizer": "powell"}
    # },

    "VQE_BFGS": {
        "class": qf.UCCNVQE,
        "run_kwargs": {"optimizer": "bfgs"}
    },

    "VQE_SLSQP": {
        "class": qf.UCCNVQE,
        "run_kwargs": {"optimizer": "slsqp"}
    },

    "VQE_l-bfgs-b": {
        "class": qf.UCCNVQE,
        "run_kwargs": {"optimizer": "l-bfgs-b"}
    },

    # more to add, but testing code

    # PQE
    "PQE_jacobi": {
        "class": qf.UCCNPQE,
        "run_kwargs": {"optimizer": "jacobi"}
    },

    # SRU Variants (PQE)
    "SRU_cyclic_in": {
        "class": qf.UCCNPQE,
        "run_kwargs": {
            "optimizer": "sru",
            "sru_type": "inside",
            "sru_sweep_type": "cyclic"
        }
    },

    "SRU_cyclic_out": {
        "class": qf.UCCNPQE,
        "run_kwargs": {
            "optimizer": "sru",
            "sru_type": "outside",
            "sru_sweep_type": "cyclic"
        }
    },

    "SRU_alternating_in": {
        "class": qf.UCCNPQE,
        "run_kwargs": {
            "optimizer": "sru",
            "sru_type": "inside",
            "sru_sweep_type": "alternating"
        }
    },

    "SRU_alternating_out": {
        "class": qf.UCCNPQE,
        "run_kwargs": {
            "optimizer": "sru",
            "sru_type": "outside",
            "sru_sweep_type": "alternating"
        }
    },

    "SRU_random_in": {
        "class": qf.UCCNPQE,
        "run_kwargs": {
            "optimizer": "sru",
            "sru_type": "inside",
            "sru_sweep_type": "random"
        }
    },

    "SRU_random_out": {
        "class": qf.UCCNPQE,
        "run_kwargs": {
            "optimizer": "sru",
            "sru_type": "outside",
            "sru_sweep_type": "random"
        }
    },

    "SRU_greedy_in": {
        "class": qf.UCCNPQE,
        "run_kwargs": {
            "optimizer": "sru",
            "sru_type": "inside",
            "sru_sweep_type": "greedy"
        }
    },

    "SRU_greedy_out": {
        "class": qf.UCCNPQE,
        "run_kwargs": {
            "optimizer": "sru",
            "sru_type": "outside",
            "sru_sweep_type": "greedy"
        }
    }
}

# automatic data collection
results = []
timer = qf.local_timer()

base_dir = Path("summaries")
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
        alg_fci.run(
            opt_thresh=1.0e-12,
            pool_type='SD',
            **method_data["run_kwargs"]
        )
        timer.record("alg fci run")

        E = alg_fci._Egs
        dE0 = E - Efci

        # rename data files for specific method/molecule
        data_file = Path("data.dat")

        if data_file.exists():
            new_name = f"summary_{mol_name}_r{r}_{method_name}.dat"
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