# index graphs for SD excitations using a Givens gate
# single excitations --> single givens
# double excitations --> double givens == QNP_or??

import qforte as qf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

def spin_orb_to_spatial(spin_orb):
    """Return (is_alpha, spatial_orbital) for qforte's spin-orbital indexing."""
    spin_orb = int(spin_orb)
    return spin_orb % 2 == 0, spin_orb // 2


def read_givens_mapping_for_sq_term(graph, sq_term):
    """Read FCIGraph mappings for a spin-preserving single or double SQ term.

    Supported excitation patterns are single alpha->alpha, single beta->beta,
    same-spin doubles, and a spin-preserving double containing one alpha->alpha
    move plus one beta->beta move. True spin-flip terms are rejected.  The returned ``sectors`` list contains one mapping for
    each spin sector changed by the operator.
    """
    coeff, crea_ops, ann_ops = sq_term
    crea_ops = [int(index) for index in crea_ops]
    ann_ops = [int(index) for index in ann_ops]

    if len(crea_ops) != len(ann_ops) or len(crea_ops) not in (1, 2):
        raise ValueError(
            "Expected a single or double excitation with the same number of "
            f"creation and annihilation operators; got creation={crea_ops}, "
            f"annihilation={ann_ops}"
        )

    by_spin = {
        True: {"creation": [], "annihilation": []},
        False: {"creation": [], "annihilation": []},
    }
    for spin_orb in crea_ops:
        is_alpha, spatial_orb = spin_orb_to_spatial(spin_orb)
        by_spin[is_alpha]["creation"].append(spatial_orb)
    for spin_orb in ann_ops:
        is_alpha, spatial_orb = spin_orb_to_spatial(spin_orb)
        by_spin[is_alpha]["annihilation"].append(spatial_orb)

    sectors = []
    for is_alpha in (True, False):
        target_orbs = by_spin[is_alpha]["creation"]
        source_orbs = by_spin[is_alpha]["annihilation"]

        if len(target_orbs) != len(source_orbs):
            spin_name = "alpha" if is_alpha else "beta"
            raise ValueError(
                "The excitation must conserve alpha and beta electron number. "
                f"The {spin_name} sector has targets={target_orbs} and "
                f"sources={source_orbs}."
            )
        if not target_orbs:
            continue

        count, source, target, parity = graph.make_givens_mapping_each(
            is_alpha,
            source_orbs,
            target_orbs,
        )
        sectors.append({
            "is_alpha": is_alpha,
            "spin_sector": "alpha" if is_alpha else "beta",
            "source_orbs": list(source_orbs),
            "target_orbs": list(target_orbs),
            "count": int(count),
            "source": list(source),
            "target": list(target),
            "parity": list(parity),
        })

    if not sectors:
        raise ValueError("The SQ term does not contain an excitation.")

    rank = len(crea_ops)
    spin_sector = "+".join(sector["spin_sector"] for sector in sectors)
    return {
        "coeff": coeff,
        "rank": rank,
        "excitation_type": "single" if rank == 1 else "double",
        "spin_sector": spin_sector,
        "creation_spin_orbs": crea_ops,
        "annihilation_spin_orbs": ann_ops,
        "sectors": sectors,
    }


def expand_mapping_to_fci_givens_pairs(mapping, graph):
    """Expand one- or two-sector mappings into FCI determinant pairs."""
    len_alpha = int(graph.get_lena())
    len_beta = int(graph.get_lenb())
    sectors = mapping["sectors"]
    givens_pairs = []

    def sector_entries(sector):
        return list(zip(sector["source"], sector["target"], sector["parity"]))

    if len(sectors) == 1:
        sector = sectors[0]
        for source_idx, target_idx, parity in sector_entries(sector):
            if sector["is_alpha"]:
                for beta_idx in range(len_beta):
                    source_coord = (source_idx, beta_idx)
                    target_coord = (target_idx, beta_idx)
                    givens_pairs.append({
                        "qubit_1_coord": source_coord,
                        "qubit_2_coord": target_coord,
                        "source_coord": source_coord,
                        "target_coord": target_coord,
                        "parity": parity,
                        "color_key": "positive" if parity > 0 else "negative",
                    })
            else:
                for alpha_idx in range(len_alpha):
                    source_coord = (alpha_idx, source_idx)
                    target_coord = (alpha_idx, target_idx)
                    givens_pairs.append({
                        "qubit_1_coord": source_coord,
                        "qubit_2_coord": target_coord,
                        "source_coord": source_coord,
                        "target_coord": target_coord,
                        "parity": parity,
                        "color_key": "positive" if parity > 0 else "negative",
                    })

    elif len(sectors) == 2:
        alpha_sector = next(s for s in sectors if s["is_alpha"])
        beta_sector = next(s for s in sectors if not s["is_alpha"])
        for a_source, a_target, a_parity in sector_entries(alpha_sector):
            for b_source, b_target, b_parity in sector_entries(beta_sector):
                parity = a_parity * b_parity
                source_coord = (a_source, b_source)
                target_coord = (a_target, b_target)
                givens_pairs.append({
                    "qubit_1_coord": source_coord,
                    "qubit_2_coord": target_coord,
                    "source_coord": source_coord,
                    "target_coord": target_coord,
                    "parity": parity,
                    "color_key": "positive" if parity > 0 else "negative",
                })
    else:
        raise ValueError(f"Expected one or two spin sectors, got {len(sectors)}")

    altered_matrix_coords = sorted({
        coord
        for pair in givens_pairs
        for coord in (pair["qubit_1_coord"], pair["qubit_2_coord"])
    })
    return givens_pairs, altered_matrix_coords


def make_givens_color_matrix(givens_pairs, graph):
    """Create a matrix with color codes for untouched, source, and target cells."""
    len_alpha = int(graph.get_lena())
    len_beta = int(graph.get_lenb())
    color_matrix = np.zeros((len_alpha, len_beta), dtype=int)

    for pair in givens_pairs:
        source_alpha, source_beta = pair["qubit_1_coord"]
        target_alpha, target_beta = pair["qubit_2_coord"]

        color_matrix[source_alpha, source_beta] = 1

        if color_matrix[target_alpha, target_beta] == 1:
            color_matrix[target_alpha, target_beta] = 3
        else:
            color_matrix[target_alpha, target_beta] = 2

    return color_matrix


def bitstring_label(index, bitstring, norb):
    """Format a determinant-string index with its orbital occupation bitstring."""
    occ = format(int(bitstring), f"0{norb}b")[::-1]
    return f"{index}: {occ}"


def determinant_axis_labels(graph):
    """Build alpha and beta axis labels from FCIGraph determinant strings."""
    norb = int(graph.get_astr_at_idx(0)).bit_length()
    for bitstring in list(graph.get_astr()) + list(graph.get_bstr()):
        norb = max(norb, int(bitstring).bit_length())

    alpha_labels = [
        bitstring_label(index, bitstring, norb)
        for index, bitstring in enumerate(graph.get_astr())
    ]
    beta_labels = [
        bitstring_label(index, bitstring, norb)
        for index, bitstring in enumerate(graph.get_bstr())
    ]

    return alpha_labels, beta_labels


def plot_givens_color_matrix(color_matrix, givens_pairs, mapping, graph, output_path):
    """Plot alpha rows by beta columns and save the color-coded matrix."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = [
        "#d9d9d9c6",  # untouched
        "#0aa137",  # qubit 1 / source
        "#b3190b",  # qubit 2 / target
        "#96407e",  # both source and target
    ]
    labels = [
        "untouched",
        "qubit 1 / source",
        "qubit 2 / target",
        "source and target",
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), cmap.N)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(color_matrix, cmap=cmap, norm=norm, origin="upper")
    
    # Draw one arrow for every unique determinant-pair mapping.  Singles and
    # same-spin doubles are vertical/horizontal; doubles containing one alpha move
    # and one beta move are diagonal. These are not spin-flip excitations.
    unique_pairs = sorted({
        (pair["qubit_1_coord"], pair["qubit_2_coord"])
        for pair in givens_pairs
    })
    for (source_row, source_col), (target_row, target_col) in unique_pairs:
        ax.annotate(
            "",
            xy=(target_col, target_row),
            xytext=(source_col, source_row),
            arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.5),
            zorder=5,
        )

    alpha_labels, beta_labels = determinant_axis_labels(graph)

    ax.set_xlabel("beta determinant string")
    ax.set_ylabel("alpha determinant string")
    moves = ", ".join(
        f"{sector['spin_sector']} {sector['source_orbs']} -> {sector['target_orbs']}"
        for sector in mapping["sectors"]
    )
    ax.set_title(
        f"FCI determinant mapping for Givens {mapping['excitation_type']} excitation\n"
        f"{moves}"
    )

    ax.set_xticks(np.arange(color_matrix.shape[1]))
    ax.set_yticks(np.arange(color_matrix.shape[0]))
    ax.set_xticklabels(beta_labels, rotation=45, ha="right")
    ax.set_yticklabels(alpha_labels)
    ax.set_xticks(np.arange(-0.5, color_matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, color_matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    legend_handles = [
        Patch(facecolor=color, edgecolor="black", label=label)
        for color, label in zip(colors, labels)
    ]
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    fig.subplots_adjust(left=0.18, right=0.72, top=0.86, bottom=0.22)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.45)
    plt.close(fig)


# build H4 molecule
geom = [
    ('H', (0., 0., 1.0)), 
    ('H', (0., 0., 2.0)),
    ('H', (0., 0., 3.0)), 
    ('H', (0., 0., 4.0)),
    # ('H', (0., 0., 5.0)), 
    # ('H', (0., 0., 6.0))
]

mol = qf.system_factory(build_type='psi4', mol_geometry=geom, basis='sto-3g', run_fci=1)
 
print("\n Initial FCIcomp Stuff")
print("===========================")
ref = mol.hf_reference

nel = sum(ref) # num electrons
sz = 0 # spin Z component
norb = int(len(ref) / 2) # number orbitals

print(f" nqbit:     {norb*2}")
print(f" nel:       {nel}")
 
fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)

# build FCI graph object
graph = qf.FCIGraph(int(norb/2), int(norb/2), norb)

# build UCCsd pool for molecule
pool = qf.SQOpPool()
pool.set_orb_spaces(ref)
pool.fill_pool("SD")
print(pool)

# make individual second quantized operators representative of the ones in pool
# Select the first pool operator containing a double-excitation SQ term.
# Change ``desired_rank`` to 1 to inspect a single excitation instead.
desired_rank = 2
# op1 = next(
#     op for op in pool.terms()
#     if any(len(term[1]) == desired_rank for term in op[1].terms())
# )

target_mu = 10
op1 = pool.terms()[target_mu]

print(op1[0])
print(op1[1])

subop1 = op1[1].terms()
print(subop1)
sq_term = next(term for term in subop1 if len(term[1]) == desired_rank)

# write subfunction that reads info from each operator --> source, targ, parity lists
mapping_info = read_givens_mapping_for_sq_term(graph, sq_term)
print("\n Givens mapping info for first pool operator")
print("============================================")
print(f" excitation type:    {mapping_info['excitation_type']}")
print(f" spin sector(s):     {mapping_info['spin_sector']}")
for sector in mapping_info["sectors"]:
    print(f" {sector['spin_sector']} spatial move: {sector['source_orbs']} -> {sector['target_orbs']}")
    print(f"   count:  {sector['count']}")
    print(f"   source: {sector['source']}")
    print(f"   target: {sector['target']}")
    print(f"   parity: {sector['parity']}")

# take the indices from the subfunction and mapping function and group them into Givens pairs
givens_pairs, altered_matrix_coords = expand_mapping_to_fci_givens_pairs(mapping_info, graph)
print("\n Givens-pair data prepared for later plotting")
print("=============================================")
print(f" matrix shape:           ({graph.get_lena()}, {graph.get_lenb()})")
print(f" Givens pair count:       {len(givens_pairs)}")
print(f" altered matrix coords:   {altered_matrix_coords}")
print(f" first Givens pairs:      {givens_pairs[:10]}")

# read in indices and color coordinate them
color_matrix = make_givens_color_matrix(givens_pairs, graph)
print("\n Color-code matrix")
print("==================")
print(color_matrix)

# create matplot script
output_path = Path(__file__).resolve().parent / "double_ab_givens_SQop_index_map.png"
plot_givens_color_matrix(color_matrix, givens_pairs, mapping_info, graph, output_path)
print(f"\nSaved color-coded matrix graph to: {output_path}")