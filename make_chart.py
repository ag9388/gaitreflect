"""README 用のグラフ (PNG): 証拠レベル別・不具合の数別の「3 回以内に解けた割合」。

  python3 make_chart.py results/summary.json chart.png

色はページの L1/L2/L3 と同じ順序付きの青 3 段 (validate_palette --ordinal で確認済み)。
棒にはすべて「k/n」を書くので、色だけに頼らない。
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from pathlib import Path  # noqa: E402

for _f in (Path(__file__).resolve().parent / "assets" / "fonts").glob("Inter-*.ttf"):
    font_manager.fontManager.addfont(str(_f))

LV = [("L1", "L1 · measurements only", "#6da7ec"),
      ("L2", "L2 · + indices & bone motion", "#2a78d6"),
      ("L3", "L3 · + near-answer hints", "#104281")]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#dcdbd5", "#fcfcfb"


def main():
    summ = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    n = summ["levels"]["L1"]["n"]
    groups = [("1 defect", lambda lv: summ["by_count"]["1"][lv]),
              ("2 defects", lambda lv: summ["by_count"]["2"][lv]),
              ("3 defects", lambda lv: summ["by_count"]["3"][lv]),
              (f"All {n}", lambda lv: [summ["levels"][lv]["solved_any"], summ["levels"][lv]["n"]])]
    plt.rcParams.update({"font.family": "Inter", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    bw, gap = 0.24, 0.02
    for gi, (label, get) in enumerate(groups):
        for li, (lv, _, col) in enumerate(LV):
            k, m = get(lv)
            p = k / m if m else 0
            x = gi + (li - 1) * (bw + gap)
            ax.bar(x, p, width=bw, color=col, edgecolor=SURF, linewidth=1, zorder=3)
            ax.text(x, p + 0.025, f"{k}/{m}", ha="center", va="bottom", fontsize=8, color=INK2, zorder=4)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], color=INK2)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"], color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.set_title("Solved within 3 rounds, by evidence level", loc="left", fontsize=11, color=INK, pad=24)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in LV]
    ax.legend(handles, [name for _, name, _ in LV], loc="upper left", bbox_to_anchor=(0, 1.13), ncol=3,
              frameon=False, fontsize=8.5, labelcolor=INK2, handlelength=1.0, columnspacing=1.4)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    main()
