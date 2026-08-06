# index graphs for spin-preserving single and double excitations using a UCC circuit

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


def read_ucc_mapping_for_sq_term(graph, sq_term):
    """Read UCC/FCIGraph mappings for a spin-preserving single or double term.

    Supported cases:
      * one alpha -> alpha excitation
      * one beta -> beta excitation
      * two alpha -> alpha excitations
      * two beta -> beta excitations
      * one alpha -> alpha plus one beta -> beta excitation

    A term is rejected only when it changes the number of alpha or beta
    electrons, i.e. when it contains a genuine spin flip.
    """
    coeff, crea_ops, ann_ops = sq_term
    crea_ops = [int(index) for index in crea_ops]
    ann_ops = [int(index) for index in ann_ops]

    if len(crea_ops) != len(ann_ops) or len(crea_ops) not in (1, 2):
        raise ValueError(
            "Expected a single or double excitation with equal numbers of "
            f"creation and annihilation operators; got creation={crea_ops}, "
            f"annihilation={ann_ops}."
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
                "Spin-flipping excitation encountered. The term must conserve "
                f"the number of {spin_name} electrons, but has targets="
                f"{target_orbs} and sources={source_orbs}."
            )

        if not target_orbs:
            continue

        count, source, target, parity = graph.make_mapping_each(
            is_alpha,
            target_orbs,  # daggered/creation spatial orbitals
            source_orbs,  # undaggered/annihilation spatial orbitals
        )

        count = int(count)
        sectors.append({
            "is_alpha": is_alpha,
            "spin_sector": "alpha" if is_alpha else "beta",
            "source_orbs": list(source_orbs),
            "target_orbs": list(target_orbs),
            "count": count,
            "source": [int(value) for value in list(source)[:count]],
            "target_bits": [int(value) for value in list(target)[:count]],
            "parity": list(parity)[:count],
        })

    if not sectors:
        raise ValueError("The SQ term does not contain an excitation.")

    rank = len(crea_ops)
    return {
        "coeff": coeff,
        "rank": rank,
        "excitation_type": "single" if rank == 1 else "double",
        "spin_sector": "+".join(s["spin_sector"] for s in sectors),
        "creation_spin_orbs": crea_ops,
        "annihilation_spin_orbs": ann_ops,
        "sectors": sectors,
    }


def _sector_index_entries(sector, graph):
    """Convert make_mapping_each target bitstrings into determinant indices."""
    strings = graph.get_astr() if sector["is_alpha"] else graph.get_bstr()
    bit_to_index = {int(bit): idx for idx, bit in enumerate(strings)}

    entries = []
    for source_idx, target_bit, parity in zip(
        sector["source"], sector["target_bits"], sector["parity"]
    ):
        target_bit = int(target_bit)
        if target_bit not in bit_to_index:
            raise KeyError(
                f"Target determinant bitstring {target_bit} was not found in "
                f"the {sector['spin_sector']} determinant list."
            )
        entries.append((int(source_idx), bit_to_index[target_bit], parity))
    return entries


def expand_mapping_to_fci_pairs(mapping, graph):
    """Expand one- or two-sector UCC mappings into full FCI matrix pairs."""
    len_alpha = int(graph.get_lena())
    len_beta = int(graph.get_lenb())
    sectors = mapping["sectors"]
    ucc_pairs = []

    if len(sectors) == 1:
        sector = sectors[0]
        for source_idx, target_idx, parity in _sector_index_entries(sector, graph):
            if sector["is_alpha"]:
                for beta_idx in range(len_beta):
                    ucc_pairs.append({
                        "qubit_1_coord": (source_idx, beta_idx),
                        "qubit_2_coord": (target_idx, beta_idx),
                        "parity": parity,
                    })
            else:
                for alpha_idx in range(len_alpha):
                    ucc_pairs.append({
                        "qubit_1_coord": (alpha_idx, source_idx),
                        "qubit_2_coord": (alpha_idx, target_idx),
                        "parity": parity,
                    })

    elif len(sectors) == 2:
        alpha_sector = next(sector for sector in sectors if sector["is_alpha"])
        beta_sector = next(sector for sector in sectors if not sector["is_alpha"])

        alpha_entries = _sector_index_entries(alpha_sector, graph)
        beta_entries = _sector_index_entries(beta_sector, graph)

        # One alpha excitation and one beta excitation act simultaneously.
        # Combining the two determinant-string maps gives diagonal connections
        # in the alpha-by-beta FCI matrix.
        for a_source, a_target, a_parity in alpha_entries:
            for b_source, b_target, b_parity in beta_entries:
                ucc_pairs.append({
                    "qubit_1_coord": (a_source, b_source),
                    "qubit_2_coord": (a_target, b_target),
                    "parity": a_parity * b_parity,
                })
    else:
        raise ValueError(f"Expected one or two spin sectors, got {len(sectors)}.")

    altered_matrix_coords = sorted({
        coord
        for pair in ucc_pairs
        for coord in (pair["qubit_1_coord"], pair["qubit_2_coord"])
    })
    return ucc_pairs, altered_matrix_coords


def make_ucc_color_matrix(ucc_pairs, graph):
    """Create a matrix with color codes for untouched, source, and target cells."""
    color_matrix = np.zeros(
        (int(graph.get_lena()), int(graph.get_lenb())), dtype=int
    )

    for pair in ucc_pairs:
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
    all_strings = list(graph.get_astr()) + list(graph.get_bstr())
    norb = max(1, max(int(bitstring).bit_length() for bitstring in all_strings))

    alpha_labels = [
        bitstring_label(index, bitstring, norb)
        for index, bitstring in enumerate(graph.get_astr())
    ]
    beta_labels = [
        bitstring_label(index, bitstring, norb)
        for index, bitstring in enumerate(graph.get_bstr())
    ]
    return alpha_labels, beta_labels


def plot_ucc_color_matrix(color_matrix, ucc_pairs, mapping, graph, output_path):
    """Plot alpha rows by beta columns and save the color-coded matrix."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = ["#d9d9d9c6", "#0aa137", "#b3190b", "#96407e"]
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

    # Vertical arrows: alpha-only mapping.
    # Horizontal arrows: beta-only mapping.
    # Diagonal arrows: simultaneous alpha and beta excitation.
    unique_pairs = sorted({
        (pair["qubit_1_coord"], pair["qubit_2_coord"])
        for pair in ucc_pairs
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
    moves = ", ".join(
        f"{sector['spin_sector']} {sector['source_orbs']} -> {sector['target_orbs']}"
        for sector in mapping["sectors"]
    )

    ax.set_xlabel("beta determinant string")
    ax.set_ylabel("alpha determinant string")
    ax.set_title(
        f"FCI determinant mapping for UCC {mapping['excitation_type']} excitation\n"
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
    ("H", (0.0, 0.0, 1.0)),
    ("H", (0.0, 0.0, 2.0)),
    ("H", (0.0, 0.0, 3.0)),
    ("H", (0.0, 0.0, 4.0)),
    # ("H", (0.0, 0.0, 5.0)),
    # ("H", (0.0, 0.0, 6.0)),
]

mol = qf.system_factory(
    build_type="psi4", mol_geometry=geom, basis="sto-3g", run_fci=1
)

print("\n Initial FCIcomp Stuff")
print("==========================")
ref = mol.hf_reference

nel = sum(ref)
sz = 0
norb = int(len(ref) / 2)

print(f" nqbit:     {norb * 2}")
print(f" nel:       {nel}")

fci_comp = qf.FCIComputer(nel=nel, sz=sz, norb=norb)
graph = qf.FCIGraph(int(norb / 2), int(norb / 2), norb)

pool = qf.SQOpPool()
pool.set_orb_spaces(ref)
pool.fill_pool("SD")
print(pool)

# Select the first pool operator containing the desired excitation rank.
# Use desired_rank = 1 for a single or desired_rank = 2 for a double.
desired_rank = 1
selected_pool_op = None
selected_sq_term = None

for pool_op in pool.terms():
    for sq_term in pool_op[1].terms():
        if len(sq_term[1]) != desired_rank or len(sq_term[2]) != desired_rank:
            continue
        try:
            read_ucc_mapping_for_sq_term(graph, sq_term)
        except ValueError:
            continue
        selected_pool_op = pool_op
        selected_sq_term = sq_term
        break
    if selected_sq_term is not None:
        break

if selected_sq_term is None:
    raise RuntimeError(
        f"No spin-preserving rank-{desired_rank} SQ term was found in the pool."
    )
    
# target_mu = 2
# selected_pool_op = pool.terms()[target_mu]
# selected_sq_term= selected_pool_op[1].terms()[0]

print("\nSelected pool operator")
print("======================")
print(selected_pool_op[0])
print(selected_pool_op[1])
print("Selected SQ term:", selected_sq_term)

mapping_info = read_ucc_mapping_for_sq_term(graph, selected_sq_term)
print("\nUCC mapping information")
print("=======================")
print(f" excitation type: {mapping_info['excitation_type']}")
print(f" spin sectors:    {mapping_info['spin_sector']}")
for sector in mapping_info["sectors"]:
    print(
        f" {sector['spin_sector']}: {sector['source_orbs']} -> "
        f"{sector['target_orbs']}"
    )
    print(f"   count:       {sector['count']}")
    print(f"   source idx:  {sector['source']}")
    print(f"   target bits: {sector['target_bits']}")
    print(f"   parity:      {sector['parity']}")

ucc_pairs, altered_matrix_coords = expand_mapping_to_fci_pairs(mapping_info, graph)
print("\nUCC-pair data prepared for plotting")
print("===================================")
print(f" matrix shape:          ({graph.get_lena()}, {graph.get_lenb()})")
print(f" UCC pair count:        {len(ucc_pairs)}")
print(f" altered matrix coords: {altered_matrix_coords}")
print(f" first UCC pairs:       {ucc_pairs[:10]}")

color_matrix = make_ucc_color_matrix(ucc_pairs, graph)
print("\nColor-code matrix")
print("=================")
print(color_matrix)

output_path = Path(__file__).resolve().parent / "first_UCCop_index_map_single.png"
plot_ucc_color_matrix(color_matrix, ucc_pairs, mapping_info, graph, output_path)
print(f"\nSaved color-coded matrix graph to: {output_path}")