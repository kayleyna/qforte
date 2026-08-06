# Unified FCI determinant-index graphs for Givens, QNP-OR, QNP-PX, and UCC
#
# One SQOpPool is built once. Select a pool operator with TARGET_MU and choose
# which circuit/gate mapping to plot with CIRCUIT_TYPE.
#
# Supported CIRCUIT_TYPE values:
#   "givens"  : one single excitation only
#   "qnp_or"  : two sequential Givens rotations for a double excitation
#   "qnp_px"  : direct source-double <-> target-double rotation
#   "ucc"     : UCC single or double excitation mapping
#
# QNP qubit convention used here:
#   q1, q2 = creation/target spin orbitals
#   q3, q4 = annihilation/source spin orbitals

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qforte as qf
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

CIRCUIT_TYPE = "qnp_px"  # "givens", "qnp_or", "qnp_px", or "ucc"
TARGET_MU = 10  # index in pool.terms()
DESIRED_RANK = 2  # 1 for singles, 2 for doubles
POOL_TYPE = "SD"

GEOMETRY = [
    ("H", (0.0, 0.0, 1.0)),
    ("H", (0.0, 0.0, 2.0)),
    ("H", (0.0, 0.0, 3.0)),
    ("H", (0.0, 0.0, 4.0)),
    ("H", (0.0, 0.0, 5.0)),
    ("H", (0.0, 0.0, 6.0)),
]

BASIS = "sto-3g"

OUTPUT_DIR = Path(__file__).resolve().parent

# Automatically create one folder per circuit type.
PLOT_DIR = OUTPUT_DIR / CIRCUIT_TYPE
PLOT_DIR.mkdir(exist_ok=True)

# =============================================================================
# GENERAL SQ-OPERATOR HELPERS
# =============================================================================

def spin_orb_to_spatial(spin_orb):
    """Return (is_alpha, spatial_orbital) for interleaved spin-orbital indices."""
    spin_orb = int(spin_orb)
    return spin_orb % 2 == 0, spin_orb // 2


def split_sq_term_by_spin(sq_term):
    """Parse one SQ term into alpha/beta source and target spatial orbitals."""
    coeff, creation_spin_orbs, annihilation_spin_orbs = sq_term
    creation_spin_orbs = [int(q) for q in creation_spin_orbs]
    annihilation_spin_orbs = [int(q) for q in annihilation_spin_orbs]

    if len(creation_spin_orbs) != len(annihilation_spin_orbs):
        raise ValueError(
            "Creation and annihilation ranks differ: "
            f"{creation_spin_orbs} versus {annihilation_spin_orbs}."
        )
    if len(creation_spin_orbs) not in (1, 2):
        raise ValueError("Only single and double excitations are supported.")

    by_spin = {
        True: {"creation": [], "annihilation": []},
        False: {"creation": [], "annihilation": []},
    }

    for q in creation_spin_orbs:
        is_alpha, spatial = spin_orb_to_spatial(q)
        by_spin[is_alpha]["creation"].append(spatial)

    for q in annihilation_spin_orbs:
        is_alpha, spatial = spin_orb_to_spatial(q)
        by_spin[is_alpha]["annihilation"].append(spatial)

    for is_alpha in (True, False):
        targets = by_spin[is_alpha]["creation"]
        sources = by_spin[is_alpha]["annihilation"]
        if len(targets) != len(sources):
            spin_name = "alpha" if is_alpha else "beta"
            raise ValueError(
                f"Spin-flipping term: {spin_name} targets={targets}, "
                f"sources={sources}."
            )

    return {
        "coeff": coeff,
        "rank": len(creation_spin_orbs),
        "excitation_type": "single" if len(creation_spin_orbs) == 1 else "double",
        "creation_spin_orbs": creation_spin_orbs,
        "annihilation_spin_orbs": annihilation_spin_orbs,
        "by_spin": by_spin,
    }


def select_pool_sq_term(pool, target_mu, desired_rank):
    """Select a pool operator manually and return one matching SQ term."""
    pool_terms = list(pool.terms())

    if not 0 <= target_mu < len(pool_terms):
        raise IndexError(
            f"TARGET_MU={target_mu} is invalid. "
            f"Valid pool indices are 0 through {len(pool_terms) - 1}."
        )

    pool_coeff, sq_operator = pool_terms[target_mu]
    matching_terms = [
        term
        for term in sq_operator.terms()
        if len(term[1]) == desired_rank and len(term[2]) == desired_rank
    ]

    if not matching_terms:
        ranks = [(len(term[1]), len(term[2])) for term in sq_operator.terms()]
        raise ValueError(
            f"Pool operator {target_mu} has no rank-{desired_rank} SQ term. "
            f"Available ranks are {ranks}."
        )

    return (pool_coeff, sq_operator), matching_terms[0]


def list_pool_operators(pool):
    """Print pool indices, excitation ranks, and operators for manual selection."""
    print("\nAvailable pool operators")
    print("========================")
    for mu, (coeff, sq_operator) in enumerate(pool.terms()):
        ranks = sorted({
            len(term[1])
            for term in sq_operator.terms()
            if len(term[1]) == len(term[2])
        })
        print(f"mu={mu:3d}  ranks={ranks}  coeff={coeff}  operator={sq_operator}")


# =============================================================================
# LOW-LEVEL GRAPH HELPERS
# =============================================================================

def graph_strings(graph, is_alpha):
    return list(graph.get_astr() if is_alpha else graph.get_bstr())


def make_single_sector_mapping(graph, is_alpha, source_orb, target_orb):
    """Use QForte's Givens helper for one same-spin orbital move."""
    count, source, target, parity = graph.make_givens_mapping_each(
        is_alpha,
        [int(source_orb)],
        [int(target_orb)],
    )
    count = int(count)
    return [
        (int(s), int(t), int(p))
        for s, t, p in zip(
            list(source)[:count],
            list(target)[:count],
            list(parity)[:count],
        )
    ]


def make_ucc_sector_mapping(graph, is_alpha, source_orbs, target_orbs):
    """Use QForte's general UCC mapping helper for one spin sector."""
    count, source, target_bits, parity = graph.make_mapping_each(
        is_alpha,
        list(target_orbs),  # creation/daggered orbitals
        list(source_orbs),  # annihilation/undaggered orbitals
    )
    count = int(count)

    strings = graph_strings(graph, is_alpha)
    bit_to_index = {int(bit): idx for idx, bit in enumerate(strings)}

    entries = []
    for source_idx, target_bit, sign in zip(
        list(source)[:count],
        list(target_bits)[:count],
        list(parity)[:count],
    ):
        target_bit = int(target_bit)
        if target_bit not in bit_to_index:
            raise KeyError(
                f"Target bitstring {target_bit} is absent from the "
                f"{'alpha' if is_alpha else 'beta'} determinant list."
            )
        entries.append(
            (int(source_idx), int(bit_to_index[target_bit]), int(sign))
        )
    return entries


def add_pair(pairs, source_coord, target_coord, parity=1, transition=""):
    """Append one standardized, bidirectional graph edge."""
    pairs.append({
        "qubit_1_coord": tuple(source_coord),
        "qubit_2_coord": tuple(target_coord),
        "source_coord": tuple(source_coord),
        "target_coord": tuple(target_coord),
        "parity": int(parity),
        "transition": transition,
    })


def unique_pairs(pairs):
    """Remove duplicate undirected edges while retaining the first metadata."""
    result = {}
    for pair in pairs:
        edge = tuple(sorted((
            tuple(pair["qubit_1_coord"]),
            tuple(pair["qubit_2_coord"]),
        )))
        result.setdefault(edge, pair)
    return list(result.values())


# =============================================================================
# GIVENS MAPPING
# =============================================================================

def make_givens_pairs(graph, parsed):
    """Map one single excitation through a single Givens rotation."""
    if parsed["rank"] != 1:
        raise ValueError(
            "A single Givens gate represents one single excitation. "
            "Use qnp_or, qnp_px, or ucc for a double excitation."
        )

    len_alpha = int(graph.get_lena())
    len_beta = int(graph.get_lenb())
    pairs = []

    for is_alpha in (True, False):
        targets = parsed["by_spin"][is_alpha]["creation"]
        sources = parsed["by_spin"][is_alpha]["annihilation"]
        if not targets:
            continue

        entries = make_single_sector_mapping(
            graph, is_alpha, sources[0], targets[0]
        )

        for source_idx, target_idx, parity in entries:
            if is_alpha:
                for beta_idx in range(len_beta):
                    add_pair(
                        pairs,
                        (source_idx, beta_idx),
                        (target_idx, beta_idx),
                        parity,
                        "alpha Givens",
                    )
            else:
                for alpha_idx in range(len_alpha):
                    add_pair(
                        pairs,
                        (alpha_idx, source_idx),
                        (alpha_idx, target_idx),
                        parity,
                        "beta Givens",
                    )

    return unique_pairs(pairs)


# =============================================================================
# QNP-OR MAPPING
# =============================================================================

def ordered_qnp_qubits(parsed):
    """Return q1,q2,q3,q4 in the convention used by QNP-PX.

    q1,q2 are creation/target spin orbitals and q3,q4 are
    annihilation/source spin orbitals. Pairing is performed within each
    spin sector so QNP-OR's two Givens rotations preserve spin.
    """
    if parsed["rank"] != 2:
        raise ValueError("QNP-OR and QNP-PX require a double excitation.")

    moves = []
    for is_alpha in (True, False):
        targets = parsed["by_spin"][is_alpha]["creation"]
        sources = parsed["by_spin"][is_alpha]["annihilation"]
        moves.extend(zip(targets, sources, [is_alpha] * len(targets)))

    if len(moves) != 2:
        raise ValueError(f"Expected exactly two orbital moves, found {moves}.")

    target_qubits = []
    source_qubits = []
    for target_spatial, source_spatial, is_alpha in moves:
        spin_offset = 0 if is_alpha else 1
        target_qubits.append(2 * int(target_spatial) + spin_offset)
        source_qubits.append(2 * int(source_spatial) + spin_offset)

    q1, q2 = target_qubits
    q3, q4 = source_qubits
    return q1, q2, q3, q4


def make_qnp_or_pairs(graph, parsed):
    """Map QNP-OR exactly as two sequential Givens rotations.

    FCIComputer::apply_qnp_or(q1,q2,q3,q4,theta) applies:
        Givens(q1,q3,theta)
        Givens(q2,q4,theta)

    The index graph is therefore the union of the edges generated by those
    two Givens rotations. For an alpha-beta double, these edges form the
    familiar four-state rectangles.
    """
    q1, q2, q3, q4 = ordered_qnp_qubits(parsed)
    move_pairs = [(q3, q1), (q4, q2)]  # source -> target
    len_alpha = int(graph.get_lena())
    len_beta = int(graph.get_lenb())
    pairs = []

    for move_number, (source_q, target_q) in enumerate(move_pairs, start=1):
        if source_q % 2 != target_q % 2:
            raise ValueError(
                f"QNP-OR move {source_q}->{target_q} changes spin."
            )

        is_alpha = source_q % 2 == 0
        source_spatial = source_q // 2
        target_spatial = target_q // 2
        entries = make_single_sector_mapping(
            graph, is_alpha, source_spatial, target_spatial
        )

        for source_idx, target_idx, parity in entries:
            if is_alpha:
                for beta_idx in range(len_beta):
                    add_pair(
                        pairs,
                        (source_idx, beta_idx),
                        (target_idx, beta_idx),
                        parity,
                        f"QNP-OR Givens {move_number}",
                    )
            else:
                for alpha_idx in range(len_alpha):
                    add_pair(
                        pairs,
                        (alpha_idx, source_idx),
                        (alpha_idx, target_idx),
                        parity,
                        f"QNP-OR Givens {move_number}",
                    )

    return unique_pairs(pairs), (q1, q2, q3, q4)


# =============================================================================
# QNP-PX MAPPING
# =============================================================================

def make_qnp_px_pairs(graph, parsed):
    """Use QForte's exact make_qnp_px_matching_each helper."""
    q1, q2, q3, q4 = ordered_qnp_qubits(parsed)

    sourcea, targeta, sourceb, targetb, phase = (
        graph.make_qnp_px_matching_each(q1, q2, q3, q4)
    )

    pairs = []
    for sa, ta, sb, tb, sign in zip(
        sourcea, targeta, sourceb, targetb, phase
    ):
        add_pair(
            pairs,
            (int(sa), int(sb)),
            (int(ta), int(tb)),
            int(sign),
            "QNP-PX pair",
        )

    return unique_pairs(pairs), (q1, q2, q3, q4)


# =============================================================================
# UCC MAPPING
# =============================================================================

def make_ucc_pairs(graph, parsed):
    """Map a UCC single or double excitation using make_mapping_each."""
    len_alpha = int(graph.get_lena())
    len_beta = int(graph.get_lenb())

    sectors = []
    for is_alpha in (True, False):
        targets = parsed["by_spin"][is_alpha]["creation"]
        sources = parsed["by_spin"][is_alpha]["annihilation"]
        if not targets:
            continue

        sectors.append({
            "is_alpha": is_alpha,
            "entries": make_ucc_sector_mapping(
                graph, is_alpha, sources, targets
            ),
        })

    pairs = []

    if len(sectors) == 1:
        sector = sectors[0]
        for source_idx, target_idx, parity in sector["entries"]:
            if sector["is_alpha"]:
                for beta_idx in range(len_beta):
                    add_pair(
                        pairs,
                        (source_idx, beta_idx),
                        (target_idx, beta_idx),
                        parity,
                        "UCC alpha sector",
                    )
            else:
                for alpha_idx in range(len_alpha):
                    add_pair(
                        pairs,
                        (alpha_idx, source_idx),
                        (alpha_idx, target_idx),
                        parity,
                        "UCC beta sector",
                    )

    elif len(sectors) == 2:
        alpha_sector = next(s for s in sectors if s["is_alpha"])
        beta_sector = next(s for s in sectors if not s["is_alpha"])

        for a_source, a_target, a_parity in alpha_sector["entries"]:
            for b_source, b_target, b_parity in beta_sector["entries"]:
                add_pair(
                    pairs,
                    (a_source, b_source),
                    (a_target, b_target),
                    a_parity * b_parity,
                    "UCC alpha-beta double",
                )
    else:
        raise ValueError(f"Expected one or two spin sectors, got {len(sectors)}.")

    return unique_pairs(pairs)


# =============================================================================
# COMMON PLOTTING
# =============================================================================

def make_color_matrix(pairs, graph):
    """Mark untouched, source, target, and source+target determinant cells."""
    matrix = np.zeros(
        (int(graph.get_lena()), int(graph.get_lenb())),
        dtype=int,
    )

    for pair in pairs:
        source = pair["qubit_1_coord"]
        target = pair["qubit_2_coord"]

        sr, sc = source
        tr, tc = target

        matrix[sr, sc] = 3 if matrix[sr, sc] == 2 else 1
        matrix[tr, tc] = 3 if matrix[tr, tc] == 1 else 2

    return matrix


def bitstring_label(index, bitstring, norb):
    occ = format(int(bitstring), f"0{norb}b")[::-1]
    return f"{index}: {occ}"


def determinant_axis_labels(graph):
    strings = list(graph.get_astr()) + list(graph.get_bstr())
    norb = max(1, max(int(bitstring).bit_length() for bitstring in strings))

    alpha_labels = [
        bitstring_label(i, bitstring, norb)
        for i, bitstring in enumerate(graph.get_astr())
    ]
    beta_labels = [
        bitstring_label(i, bitstring, norb)
        for i, bitstring in enumerate(graph.get_bstr())
    ]
    return alpha_labels, beta_labels


def excitation_description(parsed):
    pieces = []
    for is_alpha in (True, False):
        targets = parsed["by_spin"][is_alpha]["creation"]
        sources = parsed["by_spin"][is_alpha]["annihilation"]
        if sources:
            name = "alpha" if is_alpha else "beta"
            pieces.append(f"{name} {sources} -> {targets}")
    return ", ".join(pieces)


def plot_index_graph(
    color_matrix,
    pairs,
    parsed,
    graph,
    circuit_type,
    output_path,
    qnp_qubits=None,
):
    """Plot a common determinant-index graph for any supported circuit type."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = [
        "#d9d9d9c6",
        "#0aa137",
        "#b3190b",
        "#96407e",
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

    for pair in pairs:
        source_row, source_col = pair["qubit_1_coord"]
        target_row, target_col = pair["qubit_2_coord"]
        ax.annotate(
            "",
            xy=(target_col, target_row),
            xytext=(source_col, source_row),
            arrowprops={
                "arrowstyle": "<->",
                "color": "black",
                "linewidth": 1.5,
            },
            zorder=5,
        )

    display_names = {
        "givens": "Givens",
        "qnp_or": "QNP-OR",
        "qnp_px": "QNP-PX",
        "ucc": "UCC",
    }
    subtitle = excitation_description(parsed)
    if qnp_qubits is not None:
        subtitle += f"\n(q1,q2,q3,q4) = {qnp_qubits}"

    ax.set_title(
        f"H6 FCI determinant mapping for "
        f"{display_names[circuit_type]} "
        f"{parsed['excitation_type']} excitation\n"
        f"{subtitle}"
    )
    ax.set_xlabel("beta determinant string")
    ax.set_ylabel("alpha determinant string")

    alpha_labels, beta_labels = determinant_axis_labels(graph)
    ax.set_xticks(np.arange(color_matrix.shape[1]))
    ax.set_yticks(np.arange(color_matrix.shape[0]))
    ax.set_xticklabels(beta_labels, rotation=45, ha="right")
    ax.set_yticklabels(alpha_labels)

    ax.set_xticks(
        np.arange(-0.5, color_matrix.shape[1], 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, color_matrix.shape[0], 1),
        minor=True,
    )
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

    fig.subplots_adjust(
        left=0.18,
        right=0.72,
        top=0.84,
        bottom=0.22,
    )
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.45,
    )
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    if CIRCUIT_TYPE not in {"givens", "qnp_or", "qnp_px", "ucc"}:
        raise ValueError(
            f"Unknown CIRCUIT_TYPE={CIRCUIT_TYPE!r}. "
            "Choose 'givens', 'qnp_or', 'qnp_px', or 'ucc'."
        )

    molecule = qf.system_factory(
        build_type="psi4",
        mol_geometry=GEOMETRY,
        basis=BASIS,
        run_fci=1,
    )
    reference = molecule.hf_reference
    nel = int(sum(reference))
    norb = int(len(reference) / 2)

    print("\nSystem")
    print("======")
    print(f"qubits:    {2 * norb}")
    print(f"electrons: {nel}")
    print(f"reference: {reference}")

    graph = qf.FCIGraph(
        int(norb / 2),
        int(norb / 2),
        norb,
    )

    # Build the pool exactly once and reuse it for every circuit mapping.
    pool = qf.SQOpPool()
    pool.set_orb_spaces(reference)
    pool.fill_pool(POOL_TYPE)

    # Uncomment when choosing TARGET_MU:
    # list_pool_operators(pool)

    selected_pool_op, sq_term = select_pool_sq_term(
        pool,
        TARGET_MU,
        DESIRED_RANK,
    )
    parsed = split_sq_term_by_spin(sq_term)

    print("\nSelection")
    print("=========")
    print(f"circuit type: {CIRCUIT_TYPE}")
    print(f"pool index:   {TARGET_MU}")
    print(f"pool coeff:   {selected_pool_op[0]}")
    print(f"operator:     {selected_pool_op[1]}")
    print(f"SQ term:      {sq_term}")
    print(f"rank:         {parsed['rank']}")
    print(f"moves:        {excitation_description(parsed)}")

    qnp_qubits = None

    if CIRCUIT_TYPE == "givens":
        pairs = make_givens_pairs(graph, parsed)

    elif CIRCUIT_TYPE == "qnp_or":
        pairs, qnp_qubits = make_qnp_or_pairs(graph, parsed)

    elif CIRCUIT_TYPE == "qnp_px":
        pairs, qnp_qubits = make_qnp_px_pairs(graph, parsed)

    else:
        pairs = make_ucc_pairs(graph, parsed)

    altered_coords = sorted({
        coord
        for pair in pairs
        for coord in (
            pair["qubit_1_coord"],
            pair["qubit_2_coord"],
        )
    })

    print("\nMapping")
    print("=======")
    if qnp_qubits is not None:
        print(f"(q1,q2,q3,q4): {qnp_qubits}")
    print(f"pair count:     {len(pairs)}")
    print(f"altered coords: {altered_coords}")
    print(f"first pairs:    {pairs[:10]}")

    color_matrix = make_color_matrix(pairs, graph)
    output_path = PLOT_DIR / (
        f"H6_mu{TARGET_MU}_{parsed['excitation_type']}_index_map.png"
    )

    plot_index_graph(
        color_matrix,
        pairs,
        parsed,
        graph,
        CIRCUIT_TYPE,
        output_path,
        qnp_qubits=qnp_qubits,
    )
    print(f"\nSaved index graph to: {output_path}")


if __name__ == "__main__":
    main()