"""bake.py の .npz から GIF を描く (matplotlib、Blender 不要)。

  python3 render_gif.py out.gif a.npz "Broken input" b.npz "L2 repair" ... [--frames 96] [--width 300]

- カメラは斜め前 (ロボットの右前方) から、腰の前後位置を追いかける
- 胴体はグレー、顔のバイザーと胸のプレート (正面の目印) はオレンジ、1.5 倍以上伸びた面は赤
- 床の水色の多角形 = 足の支えの範囲 (接地している頂点の凸包)。点 = 重心の真下。外に出たら赤
"""

from __future__ import annotations

import argparse
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from PIL import Image  # noqa: E402

BODY = np.array([0.62, 0.66, 0.74])
FRONT = np.array([1.0, 0.50, 0.12])
HOT = np.array([0.88, 0.28, 0.24])
LIGHT = np.array([0.45, 0.65, 0.6])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def hull2d(pts):
    pts = sorted(set(map(tuple, np.round(pts, 4))))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def load(p):
    d = np.load(p)
    faces = [[i for i in f if i >= 0] for f in d["faces"]]
    V = d["verts"]
    # 面積加重の重心 (ハーネスと同じ)
    return {"V": V, "faces": faces, "mat": d["face_mat"], "stretch": d["stretch"],
            "outside": d["outside"], "onset": int(d["fall_onset"]),
            "kind": str(d["fall_kind"]) if "fall_kind" in d else "tip"}


def com_of(V, faces):
    acc, tot = np.zeros(3), 0.0
    for f in faces:
        p = V[f]
        a = 0.5 * np.linalg.norm(np.cross(p[2] - p[0], p[1] - p[0])) + 0.5 * np.linalg.norm(np.cross(p[3] - p[0], p[2] - p[0])) if len(f) == 4 else 0.0
        acc += p.mean(0) * a
        tot += a
    return acc / max(tot, 1e-9)


def draw(ax, m, k, yc, title):
    k = min(k, len(m["V"]) - 1)
    V = m["V"][k]
    ax.cla()
    ax.set_axis_off()
    ax.computed_zorder = False          # 床 → 支えの範囲 → 重心の影 → ロボット の順に重ねる
    ax.set_box_aspect((1.6, 1.6, 1.25), zoom=1.35)
    ax.view_init(elev=12, azim=18)
    ax.set_xlim(-0.8, 0.8); ax.set_ylim(yc - 0.8, yc + 0.8); ax.set_zlim(0, 1.95)
    # 床のグリッド (前へ進んでいるのがわかるように)
    for g in np.arange(np.floor(yc - 0.8), yc + 0.81, 0.25):
        ax.plot([-0.8, 0.8], [g, g], [0, 0], color="#c9cfd8", lw=0.6, zorder=1)
    for g in np.arange(-0.75, 0.76, 0.25):
        ax.plot([g, g], [yc - 0.8, yc + 0.8], [0, 0], color="#c9cfd8", lw=0.6, zorder=1)
    # 支えの範囲と重心の影
    contacts = V[V[:, 2] <= 0.03][:, :2]
    falling = m["onset"] and k + 1 >= m["onset"]
    if len(contacts) >= 3 and not (m["onset"] and k + 1 > m["onset"]):
        h = hull2d(contacts)
        if len(h) >= 3:
            ax.add_collection3d(Poly3DCollection([[(x, y, 0.001) for x, y in h]], facecolor=(0.35, 0.7, 0.95, 0.45),
                                                 edgecolor=(0.2, 0.5, 0.8, 0.9), lw=0.8, zorder=2))
    c = com_of(V, m["faces"])
    out = m["outside"][k] if k < len(m["outside"]) else 0.0
    bad = out > 0.05 or falling
    ax.scatter([c[0]], [c[1]], [0.002], s=26, c=["#e0473c" if bad else "#1f6fd1"], depthshade=False, zorder=3)
    # メッシュ (簡易な陰影)
    polys, cols = [], []
    st = m["stretch"][k]
    for i, f in enumerate(m["faces"]):
        p = V[f]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        ln = np.linalg.norm(n)
        shade = 0.55 + 0.45 * abs(float(n @ LIGHT) / ln) if ln > 1e-12 else 0.7
        base = HOT if st[i] >= 1.5 else (FRONT if m["mat"][i] == 1 else BODY)
        polys.append(p)
        cols.append(np.clip(base * shade, 0, 1))
    pc = Poly3DCollection(polys, facecolors=cols, edgecolors=(0.2, 0.22, 0.26, 0.35), linewidths=0.3, zorder=4)
    ax.add_collection3d(pc)
    ax.set_title(title, fontsize=9, pad=-4)
    if m["onset"] and k + 1 >= m["onset"]:
        msg = ("feet off the floor → dropped (Bullet physics)" if m["kind"] == "airborne"
               else "can't stand → falls (Bullet physics)")
        ax.text2D(0.5, 0.02, msg, transform=ax.transAxes, ha="center", fontsize=7.5, color="#c0392b")
    elif k + 1 > 60:
        ax.text2D(0.5, 0.02, "walk cycle ended (frame 60)", transform=ax.transAxes, ha="center",
                  fontsize=7.5, color="#666")


def render(out, models, titles, frames=96, width=300, fps=24, labels=None):
    n = len(models)
    fig = plt.figure(figsize=(width * n / 100, width * 1.15 / 100), dpi=100)
    axes = [fig.add_subplot(1, n, i + 1, projection="3d") for i in range(n)]
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94, wspace=0)
    imgs = []
    # カメラは各ロボットの腰の前後位置を、なめらかに追う
    track = []
    for m in models:
        ys = np.array([com_of(m["V"][min(k, len(m["V"]) - 1)], m["faces"])[1] for k in range(frames)])
        ker = np.ones(9) / 9
        track.append(np.convolve(np.pad(ys, 4, mode="edge"), ker, mode="valid"))
    for k in range(frames):
        for i, (ax, m) in enumerate(zip(axes, models)):
            draw(ax, m, k, track[i][k], titles[i])
        txt = fig.texts
        for t in list(txt):
            t.remove()
        fig.text(0.01, 0.005, f"frame {k + 1}", fontsize=7, color="#555")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        imgs.append(Image.open(buf).convert("RGB"))
    plt.close(fig)
    # 全フレームで同じパレットを使う (フレームごとに色表が変わると色化けする)
    w, h = imgs[0].size
    sheet = Image.new("RGB", (w, h * 4))
    for j, k in enumerate(np.linspace(0, len(imgs) - 1, 4).astype(int)):
        sheet.paste(imgs[k], (0, h * j))
    pal = sheet.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    frames_p = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    frames_p[0].save(out, save_all=True, append_images=frames_p[1:], duration=int(1000 / fps), loop=0,
                     optimize=False, disposal=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--width", type=int, default=300)
    a = ap.parse_args()
    files, titles = a.pairs[0::2], a.pairs[1::2]
    render(a.out, [load(f) for f in files], titles, a.frames, a.width)


if __name__ == "__main__":
    main()
