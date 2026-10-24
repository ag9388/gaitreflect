"""投稿・README 用の画像を作る (matplotlib)。

  python3 make_post_images.py knees  v1.npz v2.npz out.png      # v1 と v2 の膝の向き (真横)
  python3 make_post_images.py banner hero.npz out.png           # タイトル画像 (Righteous)
  python3 make_post_images.py tokens summary.json out.png       # レベル別の 1 エピソードあたりトークン

フォントは assets/fonts の Righteous (タイトル) と Inter (本文)。ページと同じ。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

import render_gif as R  # noqa: E402

HERE = Path(__file__).resolve().parent
FONTS = HERE / "assets" / "fonts"
for f in FONTS.glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
TITLE = font_manager.FontProperties(fname=str(FONTS / "Righteous.ttf"))
BODY = "Inter"
INK, INK2, INK3, SURF, ACCENT, BAD, GOOD = "#0b0b0b", "#52514e", "#7a7974", "#fcfcfb", "#2a78d6", "#b42318", "#1f7a3a"
plt.rcParams.update({"font.family": BODY, "font.size": 10})


def side(ax, m, k, title, zoom=2.0):
    R.draw(ax, m, k, R.com_of(m["V"][k], m["faces"])[1], "")
    ax.set_facecolor(SURF)
    ax.view_init(elev=4, azim=0)
    ax.set_box_aspect((1.6, 1.6, 1.25), zoom=zoom)
    ax.set_title(title, fontsize=9, color=INK2, pad=-6)


def knees(v1p, v2p, out):
    v1, v2 = R.load(v1p), R.load(v2p)
    fig = plt.figure(figsize=(10, 5.6), dpi=130)
    fig.patch.set_facecolor(SURF)
    rows = [(v1, [1, 4, 7, 10], "v1", "thigh swings back,\nshin bends forward:\na bird's knee", BAD),
            (v2, [0, 3, 6, 9], "v2", "thigh swings forward,\nshin bends back:\na human knee", GOOD)]
    for r, (m, ks, tag, cap, col) in enumerate(rows):
        y0 = 0.47 - r * 0.45
        for c, k in enumerate(ks):
            ax = fig.add_axes([0.21 + c * 0.195, y0, 0.2, 0.42], projection="3d")
            side(ax, m, k, "")
            fig.text(0.31 + c * 0.195, y0 - 0.005, f"frame {k + 1}", fontsize=8, color=INK3, ha="center")
        fig.text(0.03, y0 + 0.27, tag, fontproperties=TITLE, fontsize=32, color=col, va="center")
        fig.text(0.03, y0 + 0.12, cap, fontsize=9.5, color=INK2, va="center", linespacing=1.5)
    fig.text(0.03, 0.965, "Same side view, walking to the right", fontsize=11.5, color=INK, fontweight="semibold", va="top")
    fig.text(0.97, 0.965, "orange = face (v2 only)", fontsize=9, color=INK3, va="top", ha="right")
    fig.savefig(out, facecolor=SURF)
    print("wrote", out)


def banner(hero_p, out):
    m = R.load(hero_p)
    m["outside"] = m["outside"] * 0      # バナーでは「一瞬だけ足の外」の赤い点を出さない
    fig = plt.figure(figsize=(12, 3.6), dpi=130)
    fig.patch.set_facecolor(SURF)
    t = fig.text(0.04, 0.66, "Gait", fontproperties=TITLE, fontsize=64, color=INK, va="center")
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(fig.transFigure.inverted())     # "Gait" の幅だけ右に続ける
    fig.text(bb.x1, 0.66, "Reflect", fontproperties=TITLE, fontsize=64, color=ACCENT, va="center")
    fig.text(0.045, 0.37, "Beam repairs broken 3D robot rigs.", fontsize=15, color=INK, fontweight="semibold")
    fig.text(0.045, 0.25, "Blender grades every fix  ·  Bullet physics decides who falls", fontsize=12, color=INK2)
    fig.text(0.045, 0.10, "v2  ·  30 robots  ·  3 evidence levels  ·  90 Beam runs", fontsize=10.5, color=INK3)
    for i, k in enumerate([2, 6, 10, 14]):
        ax = fig.add_axes([0.52 + i * 0.115, 0.0, 0.17, 1.0], projection="3d")
        side(ax, m, k, "", zoom=1.7)
        ax.view_init(elev=8, azim=14)
    fig.savefig(out, facecolor=SURF)
    print("wrote", out)


def tokens(summary_p, out):
    s = json.load(open(summary_p))
    lv = [("L1", "#6da7ec"), ("L2", "#2a78d6"), ("L3", "#104281")]
    names = {"L1": "measurements only", "L2": "+ indices & bone motion", "L3": "+ near-answer hints"}
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    vals = [s["levels"][k]["tokens_per_episode"] for k, _ in lv]
    ys = list(range(len(lv)))[::-1]
    for y, (k, col), v in zip(ys, lv, vals):
        ax.barh(y, v, height=0.56, color=col, zorder=3)
        ax.text(v + max(vals) * 0.015, y, f"{v:,}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(ys, [f"{k} · {names[k]}" for k, _ in lv], color=INK2)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.grid(axis="x", color="#dcdbd5", linewidth=0.8, zorder=0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#dcdbd5")
    ax.tick_params(length=0, colors=INK2)
    ax.set_title("Beam tokens per episode (incl. reasoning)", loc="left", fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURF)
    print("wrote", out)


def loophole(run1_p, run2_p, out):
    """1 回目 (抜け穴あり) と 2 回目 (塞いだ後) の「3 回以内に合格」をレベル別に並べる。"""
    r1, r2 = json.load(open(run1_p)), json.load(open(run2_p))
    series = [("Run 1 · loophole open", r1, "#eb6834"), ("Run 2 · loophole closed", r2, "#2a78d6")]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=150)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    bw = 0.34
    for gi, lv in enumerate(["L1", "L2", "L3"]):
        for si, (_, s, col) in enumerate(series):
            k, n = s["levels"][lv]["solved_any"], s["levels"][lv]["n"]
            x = gi + (si - 0.5) * (bw + 0.03)
            ax.bar(x, k / n, width=bw, color=col, edgecolor=SURF, linewidth=1, zorder=3)
            ax.text(x, k / n + 0.025, f"{k}/{n}", ha="center", va="bottom", fontsize=8.5, color=INK2, zorder=4)
    ax.set_xticks(range(3), ["L1 · measurements only", "L2 · + bone motion", "L3 · + hints"], color=INK2)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"], color=INK2)
    ax.grid(axis="y", color="#dcdbd5", linewidth=0.8, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#dcdbd5")
    ax.tick_params(length=0)
    ax.set_title("Solved within 3 rounds: before and after closing the loophole", loc="left", fontsize=11, color=INK, pad=24)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in series]
    ax.legend(handles, [n for n, _, _ in series], loc="upper left", bbox_to_anchor=(0, 1.13), ncol=2, frameon=False,
              fontsize=8.5, labelcolor=INK2, handlelength=1.0)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"knees": knees, "banner": banner, "tokens": tokens, "loophole": loophole}[cmd](*args)
