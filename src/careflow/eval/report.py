"""Results table and the quality-vs-latency figure for block B."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Reference palette (dataviz skill, references/palette.md): one series, so one hue; text in ink tokens.
SERIES = "#2a78d6"
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"


def pareto_front(df: pd.DataFrame, x: str, y: str) -> pd.Series:
    """True for systems not dominated (lower x and higher y is better); rows with NaN are never on the front."""
    flags = []
    valid = df[[x, y]].notna().all(axis=1)
    for i, r in df.iterrows():
        if not valid[i]:
            flags.append(False)
            continue
        dominated = (valid & (df[x] <= r[x]) & (df[y] >= r[y]) & ((df[x] < r[x]) | (df[y] > r[y]))).any()
        flags.append(not dominated)
    return pd.Series(flags, index=df.index)


def quality_latency_plot(df: pd.DataFrame, path: str | Path, reference: str = "human") -> None:
    """Scatter: x = mean latency per reply (s, log), y = win-rate vs `reference` with 95% CI."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = df.dropna(subset=["latency_s", "winrate"])
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0.5, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.99, 0.49, f"parity with {reference}", transform=ax.get_yaxis_transform(), color=INK_2, fontsize=8,
            va="top", ha="right")
    yerr = [df["winrate"] - df["ci_low"], df["ci_high"] - df["winrate"]]
    ax.errorbar(df["latency_s"], df["winrate"], yerr=yerr, fmt="none", ecolor=SERIES, elinewidth=1.5,
                capsize=0, alpha=0.5, zorder=2)
    ax.scatter(df["latency_s"], df["winrate"], s=64, color=SERIES, edgecolor=SURFACE, linewidth=2, zorder=3)
    # Alternate labels above/below in latency order so neighbouring points do not collide.
    for k, (_, r) in enumerate(df.sort_values("latency_s").iterrows()):
        dy = 8 if k % 2 == 0 else -12
        ax.annotate(r["system"], (r["latency_s"], r["winrate"]), xytext=(5, dy), textcoords="offset points",
                    fontsize=7.5, color=INK)
    ax.set_xscale("log")
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.margins(x=0.15)
    ax.set_xlabel("Latency per reply, s (log scale, M5 Pro, MLX 4-bit)", color=INK_2, fontsize=9)
    ax.set_ylabel(f"Win-rate vs {reference} (HEART judge)", color=INK_2, fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_title("Quality vs latency, ESConv test sample", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


ORACLE = "#eb6834"  # categorical slot 2 (reference palette): oracle frontier vs fixed systems (slot 1)


def oracle_frontier_plot(table: pd.DataFrame, frontier: list, path: str | Path, reference: str) -> None:
    """Fixed systems as points (slot 1) and the cross-fitted oracle budget frontier as a line (slot 2)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    fixed_df = table[~table["route"].str.startswith("oracle")].sort_values("latency_s")
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0.5, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.99, 0.49, f"parity with {reference}", transform=ax.get_yaxis_transform(), color=INK_2,
            fontsize=8, va="top", ha="right")
    fx = [f.cost for f in frontier]
    fy = [f.quality for f in frontier]
    ax.plot(fx, fy, color=ORACLE, lw=2, zorder=3, drawstyle="steps-post", label="oracle routing (cross-fitted)")
    ax.errorbar(fixed_df["latency_s"], fixed_df["quality"],
                yerr=[fixed_df["quality"] - fixed_df["ci_low"], fixed_df["ci_high"] - fixed_df["quality"]],
                fmt="none", ecolor=SERIES, elinewidth=1.5, alpha=0.5, zorder=2)
    ax.scatter(fixed_df["latency_s"], fixed_df["quality"], s=64, color=SERIES, edgecolor=SURFACE, linewidth=2,
               zorder=4, label="fixed system")
    for k, (_, r) in enumerate(fixed_df.iterrows()):
        ax.annotate(r["route"], (r["latency_s"], r["quality"]), xytext=(5, 8 if k % 2 == 0 else -12),
                    textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.margins(x=0.15)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean latency per reply, s (log scale, M5 Pro, MLX 4-bit)", color=INK_2, fontsize=9)
    ax.set_ylabel(f"Win-rate vs {reference} (HEART judge)", color=INK_2, fontsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK)
    ax.set_title("Routing headroom: oracle frontier vs fixed systems", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
