import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import textwrap

# ===============================
# CONFIG
# ===============================
OUTPUT = "resnet18_clean.png"

BOX_W  = 1.4     # tăng độ rộng box
BOX_H  = 2.0
HGAP   = 0.50    # khoảng cách giữa box
FONT   = 8

COLORS = {
    "stem":   "#d9d9d9",
    "blue":   "#b3d1ff",
    "green":  "#bfe8bf",
    "orange": "#ffd8b3",
    "gray":   "#c7c7d9",
    "yellow": "#ffe68a"
}

# ===============================
# BLOCK LIST
# ===============================
blocks = [
    ("Input", COLORS["stem"]),

    # Stem
    ("7×7\nConv 64", COLORS["blue"]),
    ("3×3\nConv 64", COLORS["blue"]),
    ("3×3\nConv 64", COLORS["blue"]),

    # Layer1
    ("3×3\nConv128/2", COLORS["green"]),
    ("3×3\nConv128",   COLORS["green"]),
    ("3×3\nConv128",   COLORS["green"]),
    ("3×3\nConv128",   COLORS["green"]),

    # Layer2
    ("3×3\nConv256/2", COLORS["orange"]),
    ("3×3\nConv256",   COLORS["orange"]),
    ("3×3\nConv256",   COLORS["orange"]),
    ("3×3\nConv256",   COLORS["orange"]),

    # Layer3
    ("3×3\nConv512/2", COLORS["gray"]),
    ("3×3\nConv512",   COLORS["gray"]),
    ("3×3\nConv512",   COLORS["gray"]),
    ("3×3\nConv512",   COLORS["gray"]),

    ("AvgPool", COLORS["yellow"]),
    ("FC",       COLORS["yellow"]),
    ("Softmax",  COLORS["yellow"])
]

# Skip connections (source -> target)
skip_pairs = [
    (2, 4), (4, 6),
    (6, 8), (8, 10),
    (10, 12), (12, 14),
    (14, 16), (16, 18),
]

# Labels cho các đoạn quan trọng
NOTES = [
    (1, "Stem"),
    (3, "Layer1 (Residual Block x2)"),
    (7, "Layer2 (Residual Block x2, Downsample /2)"),
    (11, "Layer3 (Residual Block x2, Downsample /2)"),
    (15, "Layer4 (Residual Block x2, Downsample /2)"),
    (18, "Classifier Head")
]


# ===============================
# DRAW FUNCTION
# ===============================
def draw_resnet18():
    fig, ax = plt.subplots(figsize=(18, 4))

    y_center = 0
    x = 0
    box_positions = []

    # ---- Draw all blocks ----
    for label, color in blocks:
        # Wrap label cho đẹp
        wrapped = "\n".join(textwrap.wrap(label, 12))

        rect = Rectangle(
            (x, y_center - BOX_H/2),
            BOX_W, BOX_H,
            facecolor=color,
            edgecolor="black"
        )
        ax.add_patch(rect)

        ax.text(
            x + BOX_W/2,
            y_center,
            wrapped,
            ha="center",
            va="center",
            fontsize=FONT
        )

        box_positions.append((x, y_center))
        x += BOX_W + HGAP

    # ---- Straight arrows ----
    for i in range(len(box_positions) - 1):
        xs, ys = box_positions[i]
        xt, yt = box_positions[i + 1]

        arrow = FancyArrowPatch(
            (xs + BOX_W, ys),
            (xt, yt),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1
        )
        ax.add_patch(arrow)

    # ---- Skip connections ----
    for s, t in skip_pairs:
        xs, ys = box_positions[s]
        xt, yt = box_positions[t]

        skip = FancyArrowPatch(
            (xs + BOX_W, ys),
            (xt, ys + 0.5),
            connectionstyle="arc3,rad=1.2",
            arrowstyle="-|>",
            linestyle="--",
            mutation_scale=9,
            linewidth=1
        )
        ax.add_patch(skip)

    # ---- Add Notes ----
    for idx, text in NOTES:
        x_note, _ = box_positions[idx]
        ax.text(
            x_note + BOX_W/2,
            y_center + 2.2,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    ax.axis("off")
    ax.relim()
    ax.autoscale_view()
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=300)
    print("[INFO] Saved:", OUTPUT)


if __name__ == "__main__":
    draw_resnet18()
