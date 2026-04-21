from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "images"


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle=None,
    *,
    fc="#ffffff",
    ec="#2f4b66",
    lw=2.2,
    title_color="#17324d",
    subtitle_color="#486581",
    title_size=19,
    subtitle_size=12,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.60,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.32,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=subtitle_color,
        )
    return patch


def color_card(ax, x, y, w, h, title, subtitle, fill):
    return rounded_box(
        ax,
        x,
        y,
        w,
        h,
        title,
        subtitle,
        fc=fill,
        ec="none",
        lw=0,
        title_color="white",
        subtitle_color="white",
        title_size=19,
        subtitle_size=13,
    )


def add_agent_cluster(ax, cx, cy, label, accent="#7aa5d8"):
    offsets = [(-0.30, 0.08), (0.0, 0.18), (0.30, 0.08), (0.0, -0.18)]
    for dx, dy in offsets:
        head = Circle(
            (cx + dx, cy + dy),
            0.09,
            facecolor="#f6d7ad",
            edgecolor="#243b53",
            linewidth=2,
        )
        body = FancyBboxPatch(
            (cx + dx - 0.12, cy + dy - 0.23),
            0.24,
            0.14,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor=accent,
            edgecolor="#243b53",
            linewidth=2,
        )
        ax.add_patch(head)
        ax.add_patch(body)
    ax.text(cx, cy - 0.46, label, ha="center", va="center", fontsize=15, color="#102a43")


def add_arrow(ax, p1, p2, *, color="#1f5f8b", lw=2.8, dashed=False):
    arrow = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=lw,
        linestyle=(0, (6, 4)) if dashed else "solid",
        color=color,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def draw_paper():
    fig, ax = plt.subplots(figsize=(14, 7.6), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(
        6,
        6.55,
        "Scientist-Reviewer Coordination via a Shared Knowledge Graph",
        ha="center",
        va="center",
        fontsize=25,
        fontweight="bold",
        color="#102a43",
    )

    ax.text(3.1, 5.75, "Scientist-side discovery loop", ha="center", fontsize=13, color="#486581")
    ax.text(9.0, 5.75, "Reviewer-side filtering", ha="center", fontsize=13, color="#486581")

    rounded_box(ax, 1.0, 4.6, 2.2, 0.95, "Law proposal")
    rounded_box(ax, 3.8, 4.6, 2.2, 0.95, "Experiment")
    rounded_box(ax, 6.9, 4.6, 2.2, 0.95, "Review")
    rounded_box(ax, 9.6, 4.6, 1.75, 0.95, "Accept / reject", title_size=17)

    rounded_box(
        ax,
        3.1,
        1.95,
        5.8,
        1.25,
        "Shared knowledge graph",
        "accepted laws, reviewer feedback, and reusable context",
        fc="#eef6ff",
        ec="#1f5f8b",
        lw=2.5,
        title_size=21,
        subtitle_size=13,
    )

    add_agent_cluster(ax, 1.3, 1.55, "LLM-based scientist agents", accent="#b7cff6")
    add_agent_cluster(ax, 10.7, 1.55, "LLM-based reviewer agents", accent="#d2e4b4")

    add_arrow(ax, (3.2, 5.08), (3.8, 5.08), color="#ef7d32")
    add_arrow(ax, (6.0, 5.08), (6.9, 5.08), color="#d9b310")
    add_arrow(ax, (9.1, 5.08), (9.6, 5.08), color="#88b91f")
    add_arrow(ax, (1.55, 2.55), (1.55, 4.62), color="#4f9d1d", lw=3.0)
    add_arrow(ax, (10.1, 2.55), (10.1, 4.62), color="#4f9d1d", lw=3.0)
    add_arrow(ax, (10.45, 4.55), (8.55, 3.15), color="#1f5f8b", dashed=True)
    add_arrow(ax, (4.4, 1.95), (2.0, 2.55), color="#1f5f8b", dashed=True)

    ax.text(8.8, 3.45, "accepted updates", fontsize=12, color="#486581")
    ax.text(1.95, 2.75, "retrieve shared context", fontsize=12, color="#486581")

    ax.text(
        6,
        0.55,
        "Scientist agents generate and test candidate laws; reviewer agents decide what becomes shared memory.",
        ha="center",
        fontsize=13,
        color="#334e68",
    )

    fig.savefig(OUT / "scientist_reviewer_pipeline_paper.png", bbox_inches="tight")
    fig.savefig(OUT / "scientist_reviewer_pipeline_paper.svg", bbox_inches="tight")
    plt.close(fig)


def draw_presentation():
    fig, ax = plt.subplots(figsize=(14.5, 8.2), dpi=180)
    fig.patch.set_facecolor("#f6fbff")
    ax.set_facecolor("#f6fbff")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    ax.text(
        6,
        6.6,
        "Collaborative Discovery Loop",
        ha="center",
        va="center",
        fontsize=27,
        fontweight="bold",
        color="#17324d",
    )

    scientist_band = FancyBboxPatch(
        (0.7, 4.2),
        5.5,
        1.5,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor="#f5fbf3",
        edgecolor="none",
    )
    reviewer_band = FancyBboxPatch(
        (6.4, 4.2),
        4.8,
        1.5,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor="#fff7ef",
        edgecolor="none",
    )
    ax.add_patch(scientist_band)
    ax.add_patch(reviewer_band)

    ax.text(3.45, 5.85, "Scientist agents", ha="center", fontsize=13, color="#486581")
    ax.text(8.8, 5.85, "Reviewer agents", ha="center", fontsize=13, color="#486581")

    color_card(ax, 1.1, 4.55, 2.15, 0.92, "Law proposal", "", "#1e7d2b")
    color_card(ax, 3.65, 4.55, 2.15, 0.92, "Experiment", "design", "#d9b310")
    color_card(ax, 6.85, 4.55, 1.9, 0.92, "Review", "", "#ef7d32")
    color_card(ax, 9.1, 4.55, 2.0, 0.92, "Accept / reject", "", "#97c11f")

    color_card(ax, 2.85, 1.9, 2.3, 0.98, "Experiment", "execution", "#4f9d1d")
    rounded_box(
        ax,
        5.4,
        2.0,
        3.15,
        1.22,
        "Shared knowledge graph",
        "laws, relations, and accepted updates",
        fc="white",
        ec="#6ea8d8",
        lw=2.8,
        title_size=20,
        subtitle_size=12.5,
    )

    add_agent_cluster(ax, 1.4, 1.25, "Scientist agents", accent="#78aef5")
    add_agent_cluster(ax, 10.65, 1.25, "Reviewer agents", accent="#aacd5f")

    add_arrow(ax, (3.25, 5.01), (3.65, 5.01), color="#d9b310", lw=3.0)
    add_arrow(ax, (5.8, 5.01), (6.85, 5.01), color="#ef7d32", lw=3.0)
    add_arrow(ax, (8.75, 5.01), (9.1, 5.01), color="#88b91f", lw=3.0)
    add_arrow(ax, (2.0, 1.95), (2.55, 4.55), color="#4f9d1d", lw=3.1)
    add_arrow(ax, (10.15, 4.55), (8.4, 3.15), color="#88b91f", lw=2.8)
    add_arrow(ax, (6.0, 2.0), (4.95, 2.38), color="#1f5f8b", dashed=True, lw=2.6)

    ax.text(8.65, 3.45, "accepted updates", fontsize=12.5, color="#486581")
    ax.text(4.3, 2.1, "shared context", fontsize=12.5, color="#486581")

    ax.text(
        6,
        0.45,
        "The knowledge graph links proposal, experimentation, review, and reuse across the two agent groups.",
        ha="center",
        fontsize=13,
        color="#334e68",
    )

    fig.savefig(
        OUT / "scientist_reviewer_pipeline_presentation.png",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    fig.savefig(
        OUT / "scientist_reviewer_pipeline_presentation.svg",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    draw_paper()
    draw_presentation()
