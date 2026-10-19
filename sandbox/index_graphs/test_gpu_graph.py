import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Patch
from matplotlib.patches import FancyArrowPatch, Rectangle
from pathlib import Path


# ============================================================
# Locate the folder containing the H6 graphs
# ============================================================

base_dir = Path(__file__).parent

graph_dir = base_dir / "UCC_gpu_graphs"


# ============================================================
# Graph files
# ============================================================

graphs = {
    "H6 pure-alpha": graph_dir / "H6_UCC_pure_alpha_0.png",
    "H6 mixed-alpha-beta": graph_dir / "H6_UCC_mixed_alpha_beta_21.png",
}


# ============================================================
# Load image and remove the legend
# ============================================================

def load_and_crop(image_path):
    img = mpimg.imread(image_path)

    height, width = img.shape[:2]

    # Keep only the graph portion and remove the right-side legend.
    # Adjust this number if necessary.
    crop_width = int(width * 0.63)

    cropped = img[:, :crop_width]

    return cropped


class MixedPairLegendHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent,
        width, height, fontsize, trans
    ):
        box_width = 0.23 * width
        box_height = 0.68 * height
        box_y = ydescent - 0.20 * height
        center_y = ydescent + 0.14 * height

        blue_box = Rectangle(
            (xdescent, box_y), box_width, box_height,
            facecolor="#15579b", edgecolor="black",
            linewidth=1, transform=trans,
        )
        red_box = Rectangle(
            (xdescent + width - box_width, box_y),
            box_width, box_height,
            facecolor="#dc2b2f", edgecolor="black",
            linewidth=1, transform=trans,
        )
        arrow = FancyArrowPatch(
            (xdescent + box_width + 0.04 * width, center_y),
            (xdescent + width - box_width - 0.04 * width, center_y),
            arrowstyle="<->",
            color="black",
            linewidth=1.8,
            mutation_scale=13,
            transform=trans,
        )

        return [blue_box, red_box, arrow]


# ============================================================
# Create the 2x1 figure
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 14),
)


plot_order = [
    "H6 mixed-alpha-beta",
    "H6 pure-alpha",
]


for ax, graph_name in zip(axes, plot_order):

    image = load_and_crop(graphs[graph_name])

    ax.imshow(image)
    
    panel_label = "(a)" if graph_name == "H6 mixed-alpha-beta" else "(b)"
    ax.text(
        0.02, 0.98,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="black",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
            pad=1.5,
        ),
    )

    # Remove matplotlib axes because they already exist in image
    ax.axis("off")
    
    pos_b = axes[1].get_position()
    axes[1].set_position([
        pos_b.x0,
        pos_b.y0 + 0.025,
        pos_b.width,
        pos_b.height,
    ])


# ============================================================
# Create one shared legend
# ============================================================

mixed_pair_handle = object()

legend_elements = [
    Patch(facecolor="#bdbdbd", edgecolor="black", label="no read/write"),
    mixed_pair_handle,
]


fig.legend(
    handles=legend_elements,
    labels=["no read/write", "mixed coefficient pairs"],
    handler_map={type(mixed_pair_handle): MixedPairLegendHandler()},
    loc="upper center",
    bbox_to_anchor=(0.5, 0.95),
    ncol=2,
    fontsize=14,
    frameon=False,
    handlelength=3.8,   # key change: widens blue ↔ red symbol
    handleheight=1.3,
    handletextpad=0.7,
    columnspacing=2.0,
)

# ============================================================
# Adjust spacing
# ============================================================

plt.subplots_adjust(
    left=0.03,
    right=0.97,
    top=0.90,
    bottom=0.03,
    hspace=0.08,
)

# ============================================================
# Save final figure
# ============================================================

output_file = graph_dir / "UCC_combined_H6_2x1.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved combined graph to: {output_file}")
