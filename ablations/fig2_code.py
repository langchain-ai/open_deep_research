import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ---------------------------------------------------
# Input files
# ---------------------------------------------------
FILES = {
    "GPT-4.1": "ablations/comparing_v2_v3_gpt4.1.md",
    "GPT-4.1-mini": "ablations/comparing_v2_v3_gpt4.1mini.md",
}

OUTPUT_PDF = "turn3_headroom_scatter_icml.pdf"
OUTPUT_PNG = "turn3_headroom_scatter_icml.png"


# ---------------------------------------------------
# Parse markdown tables
# ---------------------------------------------------
def parse_t2_t3_markdown(path: str, model: str) -> pd.DataFrame:
    text = Path(path).read_text()

    rows = []
    current_category = None

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("## Tasks where v3 > v2"):
            current_category = "T3 > T2"
            continue

        if line.startswith("## Tasks where v2 >= v3"):
            current_category = "T2 >= T3"
            continue

        m = re.match(
            r"^\|\s*(task_\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([+-]?[0-9.]+)\s*\|$",
            line,
        )

        if m and current_category is not None:
            task, t2_ns, t3_ns, delta = m.groups()
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "t2_ns": float(t2_ns),
                    "t3_ns": float(t3_ns),
                    "delta": float(delta),
                    "category": current_category,
                }
            )

    df = pd.DataFrame(rows)

    # Sanity checks
    assert len(df) == 50, f"{model}: expected 50 tasks, found {len(df)}"
    assert (df["delta"].round(2) == (df["t3_ns"] - df["t2_ns"]).round(2)).all(), \
        f"{model}: delta mismatch detected"

    return df


# ---------------------------------------------------
# Load data
# ---------------------------------------------------
dfs = []
for model, path in FILES.items():
    dfs.append(parse_t2_t3_markdown(path, model))
df = pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------
# Optional printed summary
# ---------------------------------------------------
summary = (
    df.assign(low_t2=df["t2_ns"] < 50)
      .groupby(["model", "category"])
      .agg(
          n=("task", "count"),
          mean_t2=("t2_ns", "mean"),
          mean_delta=("delta", "mean"),
          below_50=("low_t2", "sum"),
      )
      .reset_index()
)

print(summary)


# ---------------------------------------------------
# Plot configuration
# ---------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 13,
})

X_SPLIT = 50
XMIN, XMAX = 0, 105
YMIN, YMAX = -50, 30

# Good size for a two-panel figure in a paper
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

marker_config = {
    "T3 > T2": {
        "marker": "o",
        "label": r"$T_3 > T_2$",
        "s": 60,
        "alpha": 0.85,
        "linewidths": 0.8,
    },
    "T2 >= T3": {
        "marker": "x",
        "label": r"$T_2 \geq T_3$",
        "s": 85,
        "alpha": 0.95,
        "linewidths": 1.8,
    },
}

for ax, model in zip(axes, ["GPT-4.1", "GPT-4.1-mini"]):
    sub = df[df["model"] == model]

    # Highlight top-left region: t2 < 50 and delta > 0
    ax.add_patch(
        Rectangle(
            (XMIN, 0),
            X_SPLIT - XMIN,
            YMAX - 0,
            facecolor="tab:blue",
            alpha=0.08,
            edgecolor="none",
            zorder=0,
        )
    )

    # Highlight bottom-right region: t2 >= 50 and delta <= 0
    ax.add_patch(
        Rectangle(
            (X_SPLIT, YMIN),
            XMAX - X_SPLIT,
            0 - YMIN,
            facecolor="tab:orange",
            alpha=0.08,
            edgecolor="none",
            zorder=0,
        )
    )

    # Plot scatter by category
    for category in ["T3 > T2", "T2 >= T3"]:
        part = sub[sub["category"] == category]
        cfg = marker_config[category]

        ax.scatter(
            part["t2_ns"],
            part["delta"],
            marker=cfg["marker"],
            s=cfg["s"],
            alpha=cfg["alpha"],
            linewidths=cfg["linewidths"],
            label=cfg["label"],
            zorder=3,
        )

    # Reference lines
    ax.axhline(0, linestyle="--", linewidth=1.2, zorder=2)
    ax.axvline(X_SPLIT, linestyle="--", linewidth=1.2, zorder=2)

    # Axes styling
    ax.set_title(model, pad=10)
    ax.set_xlabel("Turn 2 Normalized Score")
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7, zorder=1)

axes[0].set_ylabel(r"$\Delta$ Normalized Score $(T_3 - T_2)$")

# # Shared legend
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(
#     handles,
#     labels,
#     loc="upper center",
#     bbox_to_anchor=(0.5, 1.02),
#     ncol=2,
#     frameon=False,
# )
for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="0.75",
        facecolor="white",
        borderpad=0.35,
        labelspacing=0.35,
        handletextpad=0.5,
        fontsize=12,
    )

fig.subplots_adjust(
    top=0.93,
    bottom=0.13,
    left=0.08,
    right=0.995,
    wspace=0.06,
)
# Save and show
plt.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.05)
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.show()