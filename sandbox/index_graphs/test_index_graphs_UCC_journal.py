# index graphs for spin-preserving single and double excitations using a UCC circuit
# looking at H4 and H6 for pure-a and mixed a-b excitations

import qforte as qf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import FancyArrowPatch, Rectangle


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

class MixedPairLegendHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent,
        width, height, fontsize, trans
    ):
        box_width = 0.26 * width
        box_height = 0.80 * height

        vertical_shift = -0.25 * height
        box_y = ydescent + 0.10 * height + vertical_shift
        center_y = ydescent + 0.50 * height + vertical_shift

        blue_box = Rectangle(
            (xdescent, box_y), box_width, box_height,
            facecolor="#15579b", edgecolor="black", transform=trans,
        )
        red_box = Rectangle(
            (xdescent + width - box_width, box_y),
            box_width, box_height,
            facecolor="#dc2b2f", edgecolor="black", transform=trans,
        )
        arrow = FancyArrowPatch(
            (xdescent + box_width, center_y),
            (xdescent + width - box_width, center_y),
            arrowstyle="<->",
            color="black",
            linewidth=1.8,
            mutation_scale=13,
            transform=trans,
        )

        return [blue_box, red_box, arrow]

def plot_ucc_color_matrix(color_matrix, ucc_pairs, mapping, graph, output_path):
    """Plot alpha rows by beta columns in the style of the reference figure."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Colors matched to the second reference figure:
    # 0 = unaffected
    # 1 = source determinant
    # 2 = target determinant
    # 3 = source and target
    colors = [
        "#bdbdbd",  # gray
        "#15579b",  # blue
        "#dc2b2f",  # red
        # "#8e0fa5",  # purple
    ]

    labels = [
        "no read/write",
        "source determinant",
        "target determinant",
        # "source and target",
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(
        np.arange(-0.5, len(colors) + 0.5),
        cmap.N
    )

    # Figure size similar to the second graph
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.imshow(
        color_matrix,
        cmap=cmap,
        norm=norm,
        origin="upper",
        aspect="auto",
    )

    # ---------------------------------------------------------
    # UCC excitation arrows
    # ---------------------------------------------------------
    unique_pairs = sorted({
        (pair["qubit_1_coord"], pair["qubit_2_coord"])
        for pair in ucc_pairs
    })

    # for (source_row, source_col), (target_row, target_col) in unique_pairs:
    #     ax.annotate(
    #         "",
    #         xy=(target_col, target_row),
    #         xytext=(source_col, source_row),
    #         arrowprops=dict(
    #             arrowstyle="<->",
    #             color="black",
    #             linewidth=2.0,
    #             mutation_scale=18,
    #         ),
    #         zorder=5,
    #     )
    
    # n_rows = color_matrix.shape[0]
    # center_pair_index = 0

    # for (source_row, source_col), (target_row, target_col) in unique_pairs:

    #     is_top_pair = source_row in (0, 1) and target_row in (0, 1)
    #     is_bottom_pair = (
    #         source_row in (n_rows - 2, n_rows - 1)
    #         and target_row in (n_rows - 2, n_rows - 1)
    #     )

    #     if is_top_pair or is_bottom_pair:
    #         curve = 0.0  # straight
    #     else:
    #         curve = 0.18 if center_pair_index % 2 == 0 else -0.18
    #         center_pair_index += 1
            
    #     dx = target_col - source_col
    #     dy = target_row - source_row
    #     length = np.hypot(dx, dy)

    #     extension = 0.12  # increase to 0.18 if you want more length

    #     arrow_start = (
    #         source_col - extension * dx / length,
    #         source_row - extension * dy / length,
    #     )
    #     arrow_end = (
    #         target_col + extension * dx / length,
    #         target_row + extension * dy / length,
    #     )

    #     ax.annotate(
    #         "",
    #         xy=arrow_end,
    #         xytext=arrow_start,
    #         arrowprops=dict(
    #             arrowstyle="<->",
    #             color="black",
    #             linewidth=2.0,
    #             mutation_scale=18,
    #             connectionstyle=f"arc3,rad={curve}",
    #         ),
    #         zorder=5,
    #     )
    
    n_rows = color_matrix.shape[0]
    center_pair_index = 0

    for (source_row, source_col), (target_row, target_col) in unique_pairs:

        # Outermost top pair
        is_outer_top_pair = (
            {source_row, target_row} == {0, 1}
        )

        # Outermost bottom pair
        is_outer_bottom_pair = (
            {source_row, target_row} == {n_rows - 2, n_rows - 1}
        )

        # "Top" pairs in the middle blocks:
        # 5 <-> 7
        # 11 <-> 13
        is_middle_top_pair = (
            {source_row, target_row} in ({5, 7}, {11, 13})
        )

        if is_outer_top_pair or is_outer_bottom_pair:
            # Outer pairs stay straight and black
            curve = 0.0
            arrow_color = "black"

        elif is_middle_top_pair:
            # TOP arrows in the middle blocks = cream
            curve = -0.28 if center_pair_index % 2 == 0 else 0.28
            arrow_color = "black"
            center_pair_index += 1

        else:
            # BOTTOM arrows in the middle blocks = black
            curve = 0.28 if center_pair_index % 2 == 0 else -0.28
            arrow_color = "black"  # "#F5E6C8"
            center_pair_index += 1

        dx = target_col - source_col
        dy = target_row - source_row
        length = np.hypot(dx, dy)

        extension = 0.15

        arrow_start = (
            source_col - extension * dx / length,
            source_row - extension * dy / length,
        )

        arrow_end = (
            target_col + extension * dx / length,
            target_row + extension * dy / length,
        )

        ax.annotate(
            "",
            xy=arrow_end,
            xytext=arrow_start,
            arrowprops=dict(
                arrowstyle="<->",
                color=arrow_color,
                linewidth=2.0,
                mutation_scale=18,
                connectionstyle=f"arc3,rad={curve}",
            ),
            zorder=5,
        )


    
    # ---------------------------------------------------------
    # Axis labels
    # ---------------------------------------------------------
    ax.set_xlabel(
        r"$\beta$-spin bitstring index $I_\beta$",
        fontsize=23,
        labelpad=12,
    )

    ax.set_ylabel(
        r"$\alpha$-spin bitstring index $I_\alpha$",
        fontsize=23,
        labelpad=12,
    )

    # No title -- the second reference figure does not have one.
    ax.set_title("")

    # ---------------------------------------------------------
    # Tick labels
    # ---------------------------------------------------------
    # The reference graph uses determinant indices rather than
    # the determinant bitstrings.
    even_x = np.arange(0, color_matrix.shape[1], 2)
    even_y = np.arange(0, color_matrix.shape[0], 2)

    ax.set_xticks(even_x)
    ax.set_xticklabels([str(i) for i in even_x], fontsize=20)

    ax.set_yticks(even_y)
    ax.set_yticklabels([str(i) for i in even_y], fontsize=20)

    # ---------------------------------------------------------
    # Grid
    # ---------------------------------------------------------
    ax.set_xticks(
        np.arange(-0.5, color_matrix.shape[1], 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, color_matrix.shape[0], 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linewidth=2.0,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # Major tick appearance
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=18,
        width=1.8,
        length=7,
    )

    # ---------------------------------------------------------
    # Axis border
    # ---------------------------------------------------------
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color("black")

    # ---------------------------------------------------------
    # Legend
    # ---------------------------------------------------------
    mixed_pair_handle = object()

    legend_handles = [
        Patch(facecolor="#bdbdbd", edgecolor="black", label="no read/write"),
        mixed_pair_handle,
        # Patch(facecolor="#8e0fa5", edgecolor="black", label="source and target"),
    ]

    ax.legend(
        handles=legend_handles,
        labels=[
            "no read/write",
            "mixed coefficient pairs",
            "source and target",
        ],
        handler_map={type(mixed_pair_handle): MixedPairLegendHandler()},
        loc="center left",
        bbox_to_anchor=(1.03, 0.45),
        frameon=False,
        fontsize=20,
        handlelength=2.8,
        handleheight=1.2,
        labelspacing=1.0,
        borderaxespad=0,
    )

    # Leave enough room for the large labels and legend
    fig.subplots_adjust(
        left=0.13,
        right=0.75,
        top=0.97,
        bottom=0.13,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.2,
    )

    plt.close(fig)



# build H4 or H6 molecule
geom = [
    ("H", (0.0, 0.0, 1.0)),
    ("H", (0.0, 0.0, 2.0)),
    ("H", (0.0, 0.0, 3.0)),
    ("H", (0.0, 0.0, 4.0)),
    ("H", (0.0, 0.0, 5.0)),
    ("H", (0.0, 0.0, 6.0)),
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
    
# target_mu = 21
# selected_pool_op = pool.terms()[target_mu]
# selected_sq_term = selected_pool_op[1].terms()[0]

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

# Save graph in a separate UCC_gpu_graphs folder
output_dir = Path(__file__).resolve().parent / "UCC_gpu_graphs"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "H6_UCC_pure_alpha_0.png"

plot_ucc_color_matrix(
    color_matrix,
    ucc_pairs,
    mapping_info,
    graph,
    output_path,
)

print(f"\nSaved color-coded matrix graph to: {output_path}")