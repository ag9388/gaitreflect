"""ベンチマーク 1 問分の .blend を、仕様 JSON から生成する (Blender 側)。v2

  python3 blender_run.py - gen_episode.py -- spec.json out.blend result.json

v1 からの変更:
  - 前方は +Y、左は -X (v1 は L が +X で、前を向いたときの左右が逆だった)
  - 膝は人間の向き: もも (thigh) が前 (+Y) に振れ、すね (shin) は後ろに曲がる
    (Blender の X 回転で thigh = +θ, shin = -κ。すねの X 回転は「負 = 屈曲」)
  - 立脚の足が地面に止まるように腰 (root) の位置を脚の長さから逆算し、前へ進む
  - 頭と、顔のバイザー・胸のプレートに色を付けて前後がわかるようにした
  - 不具合:
      leak    : 隣り合わないボーンへのウェイト漏れ (v1 と同じ)
      balance : 上半身 (torso) が倒れ込み、重心が足の支えの範囲 (support polygon) から外れ続ける
                (v1 の「root をキーフレームで沈める転倒」は廃止)
      knee    : 膝の過伸展。すねが前に反りすぎて、可動域 (-150°..+5°) の上限を超える
result.json には、実際に選ばれた頂点番号と、模範解答の修正スクリプト (oracle_code) を書く。
"""

import json
import math
import sys

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

FRAMES = 60
BONE_PARENTS = {
    "root": None, "torso": "root",
    "thigh.L": "root", "shin.L": "thigh.L", "foot.L": "shin.L",
    "thigh.R": "root", "shin.R": "thigh.R", "foot.R": "shin.R",
    "arm.L": "torso", "arm.R": "torso",
}
SIDES = (("L", -1), ("R", 1))          # 前方 +Y を向いたとき、左は -X
SHIN_LIMIT = (-150.0, 5.0)             # すねの X 回転: 負 = 膝を曲げる (人間の向き)、正 = 過伸展
THIGH_LIMIT = (-50.0, 60.0)            # ももの X 回転: 正 = 前に振る
TORSO_TILT_LIMIT = (-20.0, 20.0)       # 上半身の左右の傾き (torso のローカル Z 回転 = world Y 軸回り)
ANKLE_H = 0.04
FORWARD = Vector((0.0, 1.0, 0.0))


def add_box(bm, center, size, segments_z=1, mat=0):
    cx, cy, cz = center
    sx, sy, sz = (s / 2 for s in size)
    rings = []
    for k in range(segments_z + 1):
        z = cz - sz + 2 * sz * k / segments_z
        rings.append([bm.verts.new((cx + x, cy + y, z)) for x, y in ((-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy))])
    faces = []
    for a, b in zip(rings, rings[1:]):
        for i in range(4):
            j = (i + 1) % 4
            faces.append(bm.faces.new((a[i], a[j], b[j], b[i])))
    faces.append(bm.faces.new(list(reversed(rings[0]))))
    faces.append(bm.faces.new(rings[-1]))
    for f in faces:
        f.material_index = mat
    return [v for r in rings for v in r]


def leg_ankle(theta, kappa, lt, ls):
    """腰から見た足首の位置 (前方 y, 上 z)。theta: もも前振り, kappa: 膝の屈曲 (後ろ向き)。"""
    y = lt * math.sin(theta) + ls * math.sin(theta - kappa)
    z = -(lt * math.cos(theta) + ls * math.cos(theta - kappa))
    return y, z


def gait_fn(r, hip, knee):
    """時刻 t (フレーム - 1、小数可) の関節角と腰の位置を返す関数を作る。立脚の足が止まるように腰を逆算する。"""
    P, A = r["period"], r["amp"]
    Ks, Kw = math.radians(r["knee_stance"]), math.radians(r["knee_swing"])
    lt, ls = hip - knee, knee - ANKLE_H
    half = P / 2
    S = 2 * (lt + ls) * math.sin(A)         # 1 歩の長さ

    def at(t):
        legs = {}
        for s, _ in SIDES:
            ph = (t / P + (0.0 if s == "L" else 0.5)) % 1.0
            if ph < 0.5:                    # 立脚: 足は地面に止まり、ももは前→後ろ (一定の速さ)
                u = ph / 0.5
                # 膝の曲げ伸ばしは踏み替えの瞬間に速さ 0 (sin²)。腰の動きがカクッとしない
                th, ka, stance = A * (1 - 2 * u), Ks * math.sin(math.pi * u) ** 2, True
            else:                           # 遊脚: ももは後ろ→前、膝を曲げて足を持ち上げる
                # 人の歩き方に合わせ、ももは早めに前へ出し、膝はももが体の真下を過ぎるころに最も曲がる。
                # すねは常に下向き (後ろへ水平に跳ね上げない)。蹴り出し直後は膝をゆっくり曲げる
                u = (ph - 0.5) / 0.5
                th, ka, stance = -A * math.cos(math.pi * u ** 0.75), Kw * math.sin(math.pi * u) ** 1.5, False
            legs[s] = (th, ka, stance)
        st = "L" if legs["L"][2] else "R"
        th, ka, _ = legs[st]
        ay, az = leg_ankle(th, ka, lt, ls)
        k = math.floor(t / half)
        hip_y = k * S + S / 2 - ay          # 足首 (地面に固定) から腰を逆算
        hip_z = ANKLE_H - az
        return {"legs": legs, "hip": (hip_y, hip_z), "stance": st}

    return at, S


def build(spec):
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.armatures, bpy.data.actions, bpy.data.materials):
        for item in list(coll):
            coll.remove(item)
    r = spec["robot"]
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, FRAMES
    scene.render.fps = 24

    hip, hw, knee = r["leg_len"], r["hip_w"], r["leg_len"] / 2
    th, tw, td = r["torso_h"], r["torso_w"], r["torso_d"]
    shoulder = hip + th - 0.05
    arm_x = tw / 2 + r["arm_th"] / 2 + 0.01
    hd_w, hd_d, hd_h = r["head_w"], r["head_w"] * 0.9, r["head_w"] * 0.85
    head_z = hip + th + 0.03 + hd_h / 2

    # --- メッシュ -----------------------------------------------------------
    mats = []
    for name, rgba in (("Body", (0.55, 0.6, 0.68, 1.0)), ("Front", (1.0, 0.45, 0.1, 1.0))):
        m = bpy.data.materials.new(name)
        m.diffuse_color = rgba
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = rgba
        mats.append(m)
    bm = bmesh.new()
    parts = {"torso": add_box(bm, (0, 0, hip + th / 2), (tw, td, th), 2)}
    # 頭と、前面の目印 (顔のバイザー + 胸のプレート、オレンジ)。どちらも torso に追従する
    parts["head"] = add_box(bm, (0, 0, head_z), (hd_w, hd_d, hd_h))
    parts["visor"] = add_box(bm, (0, hd_d / 2 + 0.012, head_z + hd_h * 0.08), (hd_w * 0.75, 0.024, hd_h * 0.3), mat=1)
    parts["chest"] = add_box(bm, (0, td / 2 + 0.01, hip + th * 0.68), (tw * 0.5, 0.02, th * 0.28), mat=1)
    for s, sg in SIDES:
        parts[f"leg.{s}"] = add_box(bm, (sg * hw, 0, (hip + ANKLE_H) / 2), (r["leg_th"], r["leg_th"], hip - ANKLE_H), 8)
        # 足はつま先が前 (+Y)。足首より前に 68%、後ろに 32%
        parts[f"foot.{s}"] = add_box(bm, (sg * hw, r["foot_len"] * 0.18, ANKLE_H / 2), (r["foot_w"], r["foot_len"], ANKLE_H))
        parts[f"arm.{s}"] = add_box(bm, (sg * arm_x, 0, shoulder - r["arm_len"] / 2), (r["arm_th"], r["arm_th"], r["arm_len"]), 4)
    bm.verts.index_update()
    idx = {k: [v.index for v in vs] for k, vs in parts.items()}
    zs = {v.index: v.co.z for v in bm.verts}
    me = bpy.data.meshes.new("Robot")
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    robot = bpy.data.objects.new("Robot", me)
    scene.collection.objects.link(robot)

    # --- アーマチュア -------------------------------------------------------
    arm_data = bpy.data.armatures.new("Rig")
    rig = bpy.data.objects.new("Rig", arm_data)
    scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones
    heads_tails = {
        "root": ((0, 0, hip), (0, 0, hip + 0.2)),
        "torso": ((0, 0, hip), (0, 0, hip + th)),
    }
    for s, sg in SIDES:
        x = sg * hw
        heads_tails[f"thigh.{s}"] = ((x, 0, hip), (x, 0, knee))
        heads_tails[f"shin.{s}"] = ((x, 0, knee), (x, 0, ANKLE_H))
        heads_tails[f"foot.{s}"] = ((x, 0, ANKLE_H), (x, 0.2, ANKLE_H))
        heads_tails[f"arm.{s}"] = ((sg * arm_x, 0, shoulder), (sg * arm_x, 0, shoulder - r["arm_len"]))
    for name, (h, t) in heads_tails.items():
        b = eb.new(name)
        b.head, b.tail = h, t
    for name, parent in BONE_PARENTS.items():
        if parent:
            eb[name].parent = eb[parent]
    for s in "LR":
        eb[f"shin.{s}"].use_connect = True
    bpy.ops.object.mode_set(mode="OBJECT")

    # --- ウェイト -----------------------------------------------------------
    vg = {n: robot.vertex_groups.new(name=n) for n in BONE_PARENTS}
    owner: dict[int, str] = {}          # 頂点 → 支配ボーン (ウェイト 1.0 のもの)
    upper = idx["torso"] + idx["head"] + idx["visor"] + idx["chest"]
    vg["torso"].add(upper, 1.0, "REPLACE")
    owner.update({i: "torso" for i in upper})
    seg = (hip - ANKLE_H) / 8
    for s in "LR":
        for i in idx[f"leg.{s}"]:
            z = zs[i]
            if z > knee + seg * 0.5:
                vg[f"thigh.{s}"].add([i], 1.0, "REPLACE"); owner[i] = f"thigh.{s}"
            elif z < knee - seg * 0.5:
                vg[f"shin.{s}"].add([i], 1.0, "REPLACE"); owner[i] = f"shin.{s}"
            else:
                vg[f"thigh.{s}"].add([i], 0.5, "REPLACE")
                vg[f"shin.{s}"].add([i], 0.5, "REPLACE")
        vg[f"foot.{s}"].add(idx[f"foot.{s}"], 1.0, "REPLACE")
        owner.update({i: f"foot.{s}" for i in idx[f"foot.{s}"]})
        vg[f"arm.{s}"].add(idx[f"arm.{s}"], 1.0, "REPLACE")
        owner.update({i: f"arm.{s}" for i in idx[f"arm.{s}"]})
    mod = robot.modifiers.new("Armature", "ARMATURE")
    mod.object = rig

    # --- リグ仕様 (ハーネスが読む) -------------------------------------------
    pb = rig.pose.bones
    for b in pb:
        b.rotation_mode = "XYZ"
    for s in "LR":
        pb[f"shin.{s}"]["gr_rot_limit_deg"] = [0, *SHIN_LIMIT]
        pb[f"thigh.{s}"]["gr_rot_limit_deg"] = [0, *THIGH_LIMIT]
    pb["torso"]["gr_rot_limit_deg"] = [2, *TORSO_TILT_LIMIT]
    rig["gr_min_stride_deg"] = round(math.degrees(r["amp"]), 2)      # 振り幅 2A の半分
    rig["gr_forward_axis"] = [0.0, 1.0, 0.0]

    # ボーンのローカル軸 (レスト)。world の変化量をローカルのチャンネル値に直すのに使う
    rest3 = {b.name: b.matrix_local.to_3x3() for b in arm_data.bones}

    def world_loc(bone, v):
        return tuple(rest3[bone].inverted() @ Vector(v))

    def world_rot(bone, R):
        e = (rest3[bone].inverted() @ R @ rest3[bone]).to_euler("XYZ")
        return tuple(e)

    # --- 歩行アニメーション -------------------------------------------------
    gait_at, step_len = gait_fn(r, hip, knee)
    P, arm_amp, sway = r["period"], r["arm_amp"], r["sway"]

    def walk(f, t=None):
        """フレーム f の姿勢。t を渡すと、その時刻 (フレーム - 1) の歩行姿勢を返す。"""
        t = f - 1 if t is None else t
        g = gait_at(t)
        hy, hz = g["hip"]
        out = {"root": (world_loc("root", (0, hy, hz - hip)), (0, 0, 0))}
        # 上半身を立脚側へ少し寄せる (左右の重心移動)。L 立脚のとき -X
        sx = -sway * math.sin(2 * math.pi * t / P)
        out["torso"] = (world_loc("torso", (sx, 0, 0)), (0, 0, 0))
        for s, _ in SIDES:
            th_, ka, _st = g["legs"][s]
            out[f"thigh.{s}"] = ((0, 0, 0), (th_, 0, 0))     # 正 = 前に振る
            out[f"shin.{s}"] = ((0, 0, 0), (-ka, 0, 0))      # 負 = 膝を後ろへ曲げる
            out[f"foot.{s}"] = ((0, 0, 0), (ka - th_, 0, 0))  # 足裏を地面と平行に保つ
            other = "R" if s == "L" else "L"
            out[f"arm.{s}"] = ((0, 0, 0), (arm_amp * g["legs"][other][0] / r["amp"], 0, 0))
        return out

    poses = {f: walk(f) for f in range(1, FRAMES + 1)}

    # --- 不具合を仕込む -----------------------------------------------------
    truth, oracle = [], ["import bpy", "robot = bpy.data.objects['Robot']", "rig = bpy.data.objects['Rig']", ""]
    for d in spec["defects"]:
        if d["type"] == "leak":
            cand = sorted(i for i, o in owner.items() if o == d["part"])
            if d["region"] == "lower":
                zmed = sorted(zs[i] for i in cand)[len(cand) // 2]
                cand = [i for i in cand if zs[i] <= zmed]
            vg[d["leak_to"]].add(cand, d["weight"], "REPLACE")
            truth.append({**d, "vertices": cand})
            oracle += [f"# leak {d['part']} <- {d['leak_to']}",
                       f"robot.vertex_groups[{d['leak_to']!r}].remove({cand})", ""]
        elif d["type"] == "balance":
            # 上半身が左右に倒れ込む (lean) か、腰の上から横へずれる (slide)。
            # 重心が足の支えの外に出て、立っていられなくなる
            s0, ramp = d["start"], d["ramp"]
            side = -1 if d["direction"] == "left" else 1       # 左 = -X
            clean = {f: poses[f]["torso"] for f in range(s0, FRAMES + 1)}
            for f in range(s0, FRAMES + 1):
                t = min(1.0, (f - s0) / ramp)
                e = 3 * t * t - 2 * t * t * t
                loc, rot = poses[f]["torso"]
                if d["mode"] == "lean":
                    R = Matrix.Rotation(math.radians(d["angle"]) * e * side, 3, "Y")   # +Y 軸回り = +X 側へ傾く
                    poses[f]["torso"] = (loc, world_rot("torso", R))
                else:
                    w = Vector(rest3["torso"] @ Vector(loc)) + Vector((side * d["offset"] * e, 0, 0))
                    poses[f]["torso"] = (world_loc("torso", w), rot)
            truth.append(d)
            if d["mode"] == "lean":
                oracle += [f"# balance loss from frame {s0}: torso leans {d['direction']}; restore its rotation keys",
                           "pb = rig.pose.bones['torso']",
                           f"for f in range({s0}, {FRAMES + 1}):",
                           "    pb.rotation_euler = (0.0, 0.0, 0.0)",
                           "    pb.keyframe_insert('rotation_euler', frame=f)", ""]
            else:
                vals = {f: [round(x, 6) for x in clean[f][0]] for f in clean}
                oracle += [f"# balance loss from frame {s0}: torso slides {d['direction']}; restore its location keys",
                           "pb = rig.pose.bones['torso']",
                           f"clean = {vals}",
                           "for f, v in clean.items():",
                           "    pb.location = v",
                           "    pb.keyframe_insert('location', frame=f)", ""]
        elif d["type"] == "knee":
            # 遊脚の途中で、すねが前に反る (過伸展)
            bone = f"shin.{d['side']}"
            f0, f1 = d["start"], min(FRAMES, d["start"] + d["length"] - 1)
            clean = {f: [round(x, 6) for x in poses[f][bone][1]] for f in range(f0, f1 + 1)}
            for f in range(f0, f1 + 1):
                loc, _ = poses[f][bone]
                poses[f][bone] = (loc, (math.radians(d["angle"]), 0, 0))
            truth.append({**d, "frames": [f0, f1]})
            # 模範解答: 限界値に切り詰めるのではなく、いつもの歩行の値 (膝を後ろへ曲げた値) に戻す
            oracle += [f"# knee hyperextension on {bone}, frames {f0}-{f1}: restore the normal walk-cycle rotation",
                       f"pb = rig.pose.bones[{bone!r}]",
                       f"clean = {clean}",
                       "for f, r in clean.items():",
                       "    pb.rotation_euler = r",
                       "    pb.keyframe_insert('rotation_euler', frame=f)", ""]

    for f, pose in poses.items():
        for name, (loc, rot) in pose.items():
            pb[name].location = loc
            pb[name].rotation_euler = rot
            pb[name].keyframe_insert("location", frame=f)
            pb[name].keyframe_insert("rotation_euler", frame=f)
    scene.frame_set(1)
    if any("math." in line for line in oracle):
        oracle.insert(0, "import math")
    meta = {"step_length": round(step_len, 4), "parts": {k: v for k, v in idx.items()}}
    return truth, "\n".join(oracle) + "\n", meta


if __name__ == "__main__":
    spec_p, out_blend, out_json = sys.argv[sys.argv.index("--") + 1:][:3]
    with open(spec_p, encoding="utf-8") as fh:
        spec = json.load(fh)
    truth, oracle, meta = build(spec)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({"truth": truth, "oracle_code": oracle, "meta": meta}, fh, indent=1)
    print(f"[gen_episode] {out_blend} defects={[d['type'] for d in truth]}")
