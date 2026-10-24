"""GaitReflect 検証ハーネス v2 (Blender 4.x / 5.x, bpy + bmesh)

アニメーション区間を決定論的に再生し、以下を数値で検出する:
  1. メッシュ歪み   : 各フェースの面積をレストポーズと比較し、異常な伸縮(ウェイト漏れ)を特定
  2. ウェイト監査   : 親子でない 2 本のボーンに同時に載っている頂点
  3. バランス       : 重心の XY 投影が、接地している頂点の凸包 (support polygon) から
                      一定フレーム以上続けて外れたら「立っていられない」= error
                      (v1 の「重心の高さが 35% 落ちたら転倒」は廃止。キーで沈めた転倒は物理的でないため)
  4. 関節可動域     : リグ仕様 (gr_rot_limit_deg) の範囲外の回転。膝の過伸展はここで捕まる
  5. 歩行の維持     : 太ももの振り幅と、前 (gr_forward_axis) に進んでいるか / 足先が前を向いているか
結果は JSON 辞書と、Reflection Agent にそのまま渡せるプロンプト文字列として出力する。

使い方 (バックグラウンド):
  python3 blender_run.py scene.blend harness.py -- \
      --mesh Robot --armature Rig --start 1 --end 60 \
      --report-out report.json --prompt-out prompt.md

Blender 内から:
  from harness import HarnessConfig, run_harness, build_reflection_prompt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field

import bmesh
import bpy
from mathutils import Vector


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

@dataclass
class HarnessConfig:
    mesh_name: str
    armature_name: str
    frame_start: int = 1
    frame_end: int = 60
    frame_step: int = 1

    # メッシュ歪み
    stretch_ratio: float = 1.5        # 面積がレスト比でこれ以上 → 伸び
    shrink_ratio: float = 0.4         # これ以下 → 潰れ/反転の疑い
    min_rest_area: float = 1e-6       # これ未満の極小フェースは無視(ゼロ割回避)
    weight_epsilon: float = 0.02      # これ未満のウェイトは「載っていない」扱い
    max_faces_listed: int = 8         # クラスタごとにレポートへ載せるフェース数

    # 支持基底面 (バランス)。v2 ではこれが「立っていられない」= error の判定
    ground_z: float = 0.0
    contact_epsilon: float = 0.03     # 地面からこの高さ以内の頂点を接地とみなす
    support_margin: float = 0.05      # 凸包外でもこの距離以内なら許容
    imbalance_min_frames: int = 6     # この連続フレーム数以上、重心が足の外 → error (歩行中の一瞬の逸脱は許す)
    imbalance_warn_frames: int = 4    # これ以上 (error 未満) は warning
    penetration_depth: float = 0.02   # 床にこれ以上めり込んだ頂点があれば error (足を床に沈めるだけの修正を防ぐ)
    # 動きのなめらかさ: ボーン自身のチャンネルの 2 階差分 |x(f+1) - 2x(f) + x(f-1)| (= 1 フレームでの急な加速)
    pop_rot_deg: float = 15.0         # 回転がこれ以上 → error (指摘されたフレームだけ直す「つぎはぎ」修正を防ぐ)
    pop_loc: float = 0.04             # 位置 (m) がこれ以上 → error

    # 関節可動域
    joint_tolerance_deg: float = 0.5


# ---------------------------------------------------------------------------
# 低レベルユーティリティ
# ---------------------------------------------------------------------------

def _world_bmesh(obj: bpy.types.Object, depsgraph) -> bmesh.types.BMesh:
    """評価済み(モディファイア適用後)メッシュをワールド座標の BMesh として取得。"""
    obj_eval = obj.evaluated_get(depsgraph)
    me = obj_eval.to_mesh()
    try:
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.transform(obj_eval.matrix_world)
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        return bm
    finally:
        obj_eval.to_mesh_clear()


def _face_areas(bm: bmesh.types.BMesh) -> list[float]:
    return [f.calc_area() for f in bm.faces]


def _center_of_mass(bm: bmesh.types.BMesh) -> Vector:
    """表面の面積加重重心。中身の詰まった剛体の近似としてはプロトタイプで十分。"""
    total = 0.0
    acc = Vector((0.0, 0.0, 0.0))
    for f in bm.faces:
        a = f.calc_area()
        acc += f.calc_center_median() * a
        total += a
    return acc / total if total > 0 else acc


def _convex_hull_2d(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain。反時計回りの凸包頂点を返す。"""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _outside_hull(p: tuple[float, float], hull: list[tuple[float, float]]) -> tuple[float, tuple | None]:
    """点が凸包の内側なら (0, None)、外側なら (最近接辺までの距離, 最近接点)。"""
    if not hull:
        return math.inf, None
    if len(hull) == 1:
        return math.dist(p, hull[0]), hull[0]

    def seg_near(a, b):
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
        q = (ax + t * dx, ay + t * dy)
        return math.dist(p, q), q

    edges = list(zip(hull, hull[1:] + hull[:1])) if len(hull) > 2 else [(hull[0], hull[1])]
    if len(hull) > 2:
        inside = all(
            (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0
            for a, b in edges
        )
        if inside:
            return 0.0, None
    return min((seg_near(a, b) for a, b in edges), key=lambda x: x[0])


def _vertex_weights(obj: bpy.types.Object, eps: float) -> list[dict[str, float]]:
    """頂点 index → {頂点グループ名: ウェイト}。"""
    names = {vg.index: vg.name for vg in obj.vertex_groups}
    out = []
    for v in obj.data.vertices:
        out.append({names[g.group]: round(g.weight, 3) for g in v.groups if g.weight >= eps and g.group in names})
    return out


def _bone_neighbors(arm: bpy.types.Object) -> dict[str, set[str]]:
    """ボーン名 → 自身・親・子の集合。これ以外への影響は「漏れ」の疑いが強い。"""
    nb: dict[str, set[str]] = {}
    for b in arm.data.bones:
        s = {b.name}
        if b.parent:
            s.add(b.parent.name)
        s.update(c.name for c in b.children)
        nb[b.name] = s
    return nb


def _dominant(weights: dict[str, float]) -> str | None:
    return max(weights, key=weights.get) if weights else None


def _seg_dist(p: Vector, a: Vector, b: Vector) -> float:
    ab = b - a
    L2 = ab.length_squared
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, (p - a).dot(ab) / L2))
    return (p - (a + ab * t)).length


def _compact_ranges(ids: list[int]) -> str:
    """[1,2,3,7,9,10] → '1-3, 7, 9-10'"""
    ids = sorted(ids)
    out, s = [], None
    for i, v in enumerate(ids):
        if s is None:
            s = v
        if i + 1 == len(ids) or ids[i + 1] != v + 1:
            out.append(f"{s}" if s == v else f"{s}-{v}")
            s = None
    return ", ".join(out)


def _weight_audit(obj, arm, weights, neighbors) -> list[dict]:
    """全頂点を走査し、互いに隣接しないボーンの両方にウェイトが載る頂点を組ごとに集計する。

    どちらが「漏れ」側かは、レストポーズでの頂点–ボーン距離の平均で判定する(遠い方が漏れ)。
    """
    segs = {b.name: (arm.matrix_world @ b.head_local, arm.matrix_world @ b.tail_local) for b in arm.data.bones}
    pairs: dict[tuple[str, str], list[int]] = {}
    for i, w in enumerate(weights):
        names = sorted(g for g in w if g in segs)
        for x in range(len(names)):
            for y in range(x + 1, len(names)):
                a, b = names[x], names[y]
                if b not in neighbors.get(a, ()):
                    pairs.setdefault((a, b), []).append(i)

    out = []
    for (a, b), vids in sorted(pairs.items(), key=lambda kv: -len(kv[1])):
        pos = [obj.matrix_world @ obj.data.vertices[i].co for i in vids]
        dist = {g: sum(_seg_dist(p, *segs[g]) for p in pos) / len(pos) for g in (a, b)}
        leak = max(dist, key=dist.get)
        keep = a if leak == b else b
        ws = [weights[i][leak] for i in vids]
        out.append({
            "groups": [a, b],
            "vertex_count": len(vids),
            "vertices": _compact_ranges(vids),
            "vertex_ids": vids,
            "mean_distance_to_bone": {g: round(d, 3) for g, d in dist.items()},
            "likely_leak_group": leak,
            "keep_group": keep,
            "leak_weight_range": [round(min(ws), 3), round(max(ws), 3)],
        })
    for n, e in enumerate(out, 1):
        e["id"] = f"W{n}"
    return out


def _keyed_channels(arm) -> dict[str, dict]:
    """ボーンごとのキー付きチャンネル。Blender 4.4+ のスロット付きアクションにも対応。"""
    ad = arm.animation_data
    action = ad.action if ad else None
    if action is None:
        return {}
    if hasattr(action, "fcurves"):           # 〜4.x (レガシー API)
        fcurves = list(action.fcurves)
    else:                                    # 5.x
        from bpy_extras import anim_utils
        cb = anim_utils.action_get_channelbag_for_slot(action, ad.action_slot)
        fcurves = list(cb.fcurves) if cb else []
    out: dict[str, dict] = {}
    for fc in fcurves:
        dp = fc.data_path
        if not dp.startswith('pose.bones["'):
            continue
        bone = dp.split('"')[1]
        prop = dp.rsplit(".", 1)[-1]
        frames = [k.co[0] for k in fc.keyframe_points]
        if not frames:
            continue
        e = out.setdefault(bone, {}).setdefault(prop, {"indices": [], "frames": [int(min(frames)), int(max(frames))], "keys": len(frames)})
        e["indices"].append(fc.array_index)
    return out


def _pose_snapshot(arm) -> dict[str, dict]:
    """各ポーズボーンのワールド行列と、ボーン自身のチャンネル値 (ボーンローカル空間)。"""
    snap = {}
    for pb in arm.pose.bones:
        if pb.rotation_mode == "QUATERNION":
            q = pb.rotation_quaternion.copy()
            rot = {"rotation_quaternion": [round(x, 4) for x in q]}
        elif pb.rotation_mode == "AXIS_ANGLE":
            ang, *ax = pb.rotation_axis_angle
            q = mathutils_quat_from_axis_angle(ax, ang)
            rot = {"rotation_axis_angle": [round(x, 4) for x in pb.rotation_axis_angle]}
        else:
            q = pb.rotation_euler.to_quaternion()
            rot = {"rotation_euler_deg": [round(math.degrees(x), 2) for x in pb.rotation_euler],
                   "rotation_mode": pb.rotation_mode}
        snap[pb.name] = {
            "world": (arm.matrix_world @ pb.matrix).copy(),
            "own_q": q,
            "own_loc": pb.location.copy(),
            "channels": {"location": [round(x, 4) for x in pb.location], **rot},
        }
    return snap


def mathutils_quat_from_axis_angle(axis, angle):
    from mathutils import Quaternion
    return Quaternion(Vector(axis).normalized() if Vector(axis).length else Vector((0, 0, 1)), angle)


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------

def run_harness(cfg: HarnessConfig) -> dict:
    scene = bpy.context.scene
    obj = bpy.data.objects[cfg.mesh_name]
    arm = bpy.data.objects[cfg.armature_name]
    if obj.type != "MESH" or arm.type != "ARMATURE":
        raise TypeError("mesh_name must be a MESH and armature_name an ARMATURE")

    frame_orig = scene.frame_current
    pose_orig = arm.data.pose_position

    # --- レストポーズを基準にする -------------------------------------------
    arm.data.pose_position = "REST"
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bm = _world_bmesh(obj, depsgraph)
    rest_areas = _face_areas(bm)
    face_verts = [[v.index for v in f.verts] for f in bm.faces]
    bm.free()
    arm.data.pose_position = "POSE"

    weights = _vertex_weights(obj, cfg.weight_epsilon)
    neighbors = _bone_neighbors(arm)
    bone_names = set(neighbors)

    # --- フレーム走査 ---------------------------------------------------------
    face_hits: dict[int, dict] = {}     # face index → 集計
    com_track: list[dict] = []
    pose_track: dict[int, dict] = {}    # frame → _pose_snapshot
    topology_warning = None

    frames = list(range(cfg.frame_start, cfg.frame_end + 1, cfg.frame_step))
    for frame in frames:
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        pose_track[frame] = _pose_snapshot(arm)
        bm = _world_bmesh(obj, depsgraph)

        if len(bm.faces) != len(rest_areas):
            # トポロジを変えるモディファイアがあると対応が取れない
            topology_warning = f"face count changed at frame {frame} ({len(rest_areas)} -> {len(bm.faces)})"
            bm.free()
            break

        # 1) 面積変化率
        for i, f in enumerate(bm.faces):
            a0 = rest_areas[i]
            if a0 < cfg.min_rest_area:
                continue
            r = f.calc_area() / a0
            if r >= cfg.stretch_ratio or r <= cfg.shrink_ratio:
                h = face_hits.setdefault(i, {
                    "face": i, "kind": "stretch" if r >= 1 else "shrink",
                    "first_frame": frame, "last_frame": frame,
                    "worst_ratio": r, "worst_frame": frame, "frames_flagged": 0,
                })
                h["last_frame"] = frame
                h["frames_flagged"] += 1
                if abs(math.log(r)) > abs(math.log(h["worst_ratio"])):
                    h["worst_ratio"], h["worst_frame"] = r, frame

        # 2,3) 重心と接地
        com = _center_of_mass(bm)
        contacts = [(v.co.x, v.co.y) for v in bm.verts if v.co.z <= cfg.ground_z + cfg.contact_epsilon]
        low = min(bm.verts, key=lambda v: v.co.z)
        hull = _convex_hull_2d(contacts)
        outside, near = _outside_hull((com.x, com.y), hull) if hull else (None, None)
        com_track.append({
            "frame": frame,
            "com": [round(com.x, 4), round(com.y, 4), round(com.z, 4)],
            "contacts": len(contacts),
            "outside_support": None if outside is None else round(outside, 4),
            # 重心が外に出ているとき、支えの範囲の最寄り点から重心へのベクトル (どちらへ外れたか)
            "outside_vector": None if not outside else [round(com.x - near[0], 4), round(com.y - near[1], 4)],
            "lowest": [low.index, round(low.co.z - cfg.ground_z, 4)],
        })
        bm.free()

    scene.frame_set(frame_orig)
    arm.data.pose_position = pose_orig

    # --- 集計 -----------------------------------------------------------------
    distortion = _summarize_distortion(face_hits, face_verts, weights, neighbors, bone_names, cfg)
    audit = _weight_audit(obj, arm, weights, neighbors)
    for c in distortion["clusters"]:
        c["related_audit"] = [w["id"] for w in audit if set(c["leak_groups"]) & set(w["groups"])]
    unweighted = [i for i, w in enumerate(weights) if not any(g in bone_names for g in w)]
    period = _estimate_period(pose_track)
    balance = _detect_imbalance(com_track, cfg)
    for e in balance["events"]:
        if e["severity"] == "error":
            e["bone_motion"] = _balance_bone_motion(pose_track, e, period, arm)
    joints = _joint_limits(arm, pose_track, cfg)
    ground = _ground_penetration(com_track, weights, cfg)
    pops = _motion_pops(pose_track, cfg)
    gait = _gait_check(arm, pose_track)
    gait_errors = [k for k in ("stride_ok", "direction_ok") if gait and not gait[k]]

    errors = (sum(1 for c in distortion["clusters"] if c["severity"] == "error")
              + len(audit) + sum(1 for e in balance["events"] if e["severity"] == "error")
              + len(joints) + len(ground) + len(pops) + (1 if unweighted else 0) + len(gait_errors)
              + (1 if topology_warning else 0))
    warnings = (sum(1 for c in distortion["clusters"] if c["severity"] == "warning")
                + sum(1 for e in balance["events"] if e["severity"] == "warning"))

    return {
        "schema": "gaitreflect.report/v2",
        "blender_version": bpy.app.version_string,
        "target": {"mesh": obj.name, "armature": arm.name,
                   "faces": len(rest_areas), "vertices": len(obj.data.vertices),
                   "bones": sorted(bone_names),
                   "bone_parents": {b.name: (b.parent.name if b.parent else None) for b in arm.data.bones}},
        "frames": {"start": cfg.frame_start, "end": cfg.frame_end, "step": cfg.frame_step,
                   "evaluated": len(com_track)},
        "config": asdict(cfg),
        "status": "fail" if errors else ("warn" if warnings else "pass"),
        "counts": {"errors": errors, "warnings": warnings},
        "distortion": distortion,
        "weight_audit": audit,
        "unweighted_vertices": unweighted,
        "gait_period_frames": period,
        "joint_limits": joints,
        "gait": gait,
        "keyed_channels": _keyed_channels(arm),
        "balance": balance,
        "ground_penetration": ground,
        "motion_pops": pops,
        "com_track": com_track,
        "notes": [topology_warning] if topology_warning else [],
    }


def _summarize_distortion(face_hits, face_verts, weights, neighbors, bone_names, cfg) -> dict:
    """フェースを「影響ボーンの組み合わせ」でクラスタ化し、漏れの疑いがある頂点を添える。"""
    clusters: dict[tuple, dict] = {}
    for i, h in face_hits.items():
        vids = face_verts[i]
        groups = sorted({g for v in vids for g in weights[v] if g in bone_names})
        key = (h["kind"], tuple(groups))
        c = clusters.setdefault(key, {
            "kind": h["kind"], "vertex_groups": list(groups),
            "face_count": 0, "worst_ratio": h["worst_ratio"], "worst_frame": h["worst_frame"],
            "frame_range": [h["first_frame"], h["last_frame"]],
            "faces": [], "suspect_vertices": {},
        })
        c["face_count"] += 1
        c["frame_range"][0] = min(c["frame_range"][0], h["first_frame"])
        c["frame_range"][1] = max(c["frame_range"][1], h["last_frame"])
        if abs(math.log(h["worst_ratio"])) > abs(math.log(c["worst_ratio"])):
            c["worst_ratio"], c["worst_frame"] = h["worst_ratio"], h["worst_frame"]
        c["faces"].append({**h, "worst_ratio": round(h["worst_ratio"], 3), "vertices": vids})

        # 漏れ判定: 頂点の支配ボーンと隣接していないボーンへのウェイト
        for v in vids:
            w = {g: x for g, x in weights[v].items() if g in bone_names}
            dom = _dominant(w)
            if dom is None:
                c["suspect_vertices"][v] = {"weights": {}, "reason": "unweighted"}
                continue
            leaks = {g: x for g, x in w.items() if g not in neighbors.get(dom, {dom})}
            if leaks:
                c["suspect_vertices"][v] = {"weights": w, "dominant": dom, "leaks_to": leaks,
                                            "reason": "weight on non-adjacent bone"}

    out = []
    for c in clusters.values():
        c["faces"].sort(key=lambda f: -abs(math.log(f["worst_ratio"])))
        c["faces"] = c["faces"][: cfg.max_faces_listed]
        c["worst_ratio"] = round(c["worst_ratio"], 3)
        c["suspect_vertices"] = [{"vertex": k, **v} for k, v in sorted(c["suspect_vertices"].items())]
        leak_groups = sorted({g for s in c["suspect_vertices"] for g in s.get("leaks_to", {})})
        c["leak_groups"] = leak_groups
        # 漏れ頂点があるか、比率が閾値の 1.5 倍を超えたら error
        hard = c["worst_ratio"] >= cfg.stretch_ratio * 1.5 or c["worst_ratio"] <= cfg.shrink_ratio * 0.5
        c["severity"] = "error" if (leak_groups or hard) else "warning"
        out.append(c)
    out.sort(key=lambda c: (c["severity"] != "error", -c["face_count"]))
    for n, c in enumerate(out, 1):
        c["id"] = f"D{n}"

    return {
        "flagged_faces": len(face_hits),
        "clusters": out,
    }


def _estimate_period(pose_track: dict) -> int | None:
    """歩行周期 (フレーム数) を thigh.L の X 回転の自己相関から推定する。見つからなければ None。"""
    frames = sorted(pose_track)
    xs = []
    for f in frames:
        ch = pose_track[f].get("thigh.L", {}).get("channels", {})
        xs.append(ch.get("rotation_euler_deg", [0, 0, 0])[0])
    if len(xs) < 20 or max(xs) - min(xs) < 2:
        return None
    best, best_k = None, None
    for k in range(8, min(48, len(xs) - 8)):
        d = sum(abs(xs[i] - xs[i + k]) for i in range(len(xs) - k)) / (len(xs) - k)
        if best is None or d < best - 1e-9:
            best, best_k = d, k
    return best_k


def _channel_vec(snap_b: dict) -> tuple[list[float], list[float]]:
    ch = snap_b["channels"]
    rot = ch.get("rotation_euler_deg")
    if rot is None:
        e = snap_b["own_q"].to_euler("XYZ")
        rot = [math.degrees(x) for x in e]
    return list(ch["location"]), list(rot)


def _balance_bone_motion(pose_track: dict, event: dict, period: int | None, arm) -> dict:
    """支えから外れた区間で、各ボーン自身のチャンネルが「いつもの歩行」からどれだけずれたか。

    歩行中は脚が大きく動くので、単純な前後差では原因のボーンが埋もれる。
    そこで、最も外れたフレームの値を「1 周期前 (同じ位相) の値」と比べ、
    さらに周期ごとの定常的な変化 (root が前へ進む分) を差し引いた量を「ずれ」とする。
    """
    frames = sorted(pose_track)
    f1 = event["worst_frame"]
    ref = None
    if period:
        cand = [g for g in range(f1 - period, frames[0] - 1, -period) if g < event["frame_range"][0]]
        ref = cand[0] if cand else None
    method = "same phase one gait cycle earlier"
    if ref is None:
        ref = max(frames[0], event["frame_range"][0] - 1)
        method = "last frame before the event"
    bones = []
    for name in pose_track[f1]:
        loc1, rot1 = _channel_vec(pose_track[f1][name])
        loc0, rot0 = _channel_vec(pose_track[ref][name])
        dloc = [a - b for a, b in zip(loc1, loc0)]
        drot = [a - b for a, b in zip(rot1, rot0)]
        if period and method.startswith("same"):
            # 周期ごとの定常変化 (中央値) を差し引く。周期的なチャンネルではほぼ 0
            pairs = [(f, f - period) for f in frames if f - period in pose_track]
            def med(xs):
                xs = sorted(xs)
                return xs[len(xs) // 2] if xs else 0.0
            steady_loc = [med([_channel_vec(pose_track[a][name])[0][k] - _channel_vec(pose_track[b][name])[0][k]
                               for a, b in pairs]) for k in range(3)]
            dloc = [d - s for d, s in zip(dloc, steady_loc)]
        bones.append({
            "bone": name,
            "rot_deviation_deg": round(math.sqrt(sum(x * x for x in drot)), 1),
            "loc_deviation": round(math.sqrt(sum(x * x for x in dloc)), 4),
            "channels_reference": pose_track[ref][name]["channels"],
            "channels_at_worst": pose_track[f1][name]["channels"],
        })
    bones.sort(key=lambda x: -(x["rot_deviation_deg"] + 100 * x["loc_deviation"]))
    axes = {}
    for x in bones[:3]:
        m = (arm.matrix_world.to_3x3() @ arm.data.bones[x["bone"]].matrix_local.to_3x3())
        axes[x["bone"]] = {ax: [round(v, 3) for v in m.col[k]] for k, ax in enumerate("XYZ")}
    return {"reference_frame": ref, "worst_frame": f1, "method": method, "period": period,
            "bones": bones, "rest_local_axes_in_world": axes}


def _ground_penetration(track: list[dict], weights, cfg: HarnessConfig) -> list[dict]:
    """床より cfg.penetration_depth 以上下にある頂点。めり込んでいる部位 (支配ボーン) ごとに 1 件。"""
    by_bone: dict[str, list] = {}
    for t in track:
        vid, z = t["lowest"]
        if z < -cfg.penetration_depth:
            dom = _dominant(weights[vid]) or "?"
            by_bone.setdefault(dom, []).append((t["frame"], z, vid))
    out = []
    for bone, hits in sorted(by_bone.items()):
        worst = min(hits, key=lambda h: h[1])
        out.append({"bone": bone, "frames": [hits[0][0], hits[-1][0]], "frames_below": len(hits),
                    "worst_depth": round(-worst[1], 4), "worst_frame": worst[0], "worst_vertex": worst[2]})
    for n, g in enumerate(out, 1):
        g["id"] = f"P{n}"
    return out


def _motion_pops(pose_track: dict, cfg: HarnessConfig) -> list[dict]:
    """ボーン自身のチャンネルが 1 フレームで不自然に跳ぶところ (2 階差分が閾値以上)。ボーンごとに 1 件。

    歩行や、なめらかに傾いていく動きは 2 階差分が小さい。数フレームだけ値を差し替えると、その前後で大きく跳ぶ。
    """
    frames = sorted(pose_track)
    out = []
    names = list(pose_track[frames[0]])
    for name in names:
        hits = []
        for a, b, c in zip(frames, frames[1:], frames[2:]):
            la, ra = _channel_vec(pose_track[a][name])
            lb, rb = _channel_vec(pose_track[b][name])
            lc, rc = _channel_vec(pose_track[c][name])
            dr = max(abs(x - 2 * y + z) for x, y, z in zip(ra, rb, rc))
            dl = max(abs(x - 2 * y + z) for x, y, z in zip(la, lb, lc))
            if dr >= cfg.pop_rot_deg or dl >= cfg.pop_loc:
                hits.append((b, round(dr, 1), round(dl, 4)))
        if hits:
            worst = max(hits, key=lambda h: h[1] / cfg.pop_rot_deg + h[2] / cfg.pop_loc)
            out.append({"bone": name, "frames": [h[0] for h in hits][:12], "count": len(hits),
                        "worst_frame": worst[0], "worst_rot_jump_deg": worst[1], "worst_loc_jump": worst[2]})
    for n, x in enumerate(out, 1):
        x["id"] = f"M{n}"
    return out


def _joint_limits(arm, pose_track: dict, cfg: HarnessConfig) -> list[dict]:
    """リグ仕様 (pose bone のカスタムプロパティ gr_rot_limit_deg = [軸, 最小, 最大]) に対する可動域違反。"""
    out = []
    for pb in arm.pose.bones:
        spec = pb.get("gr_rot_limit_deg")
        if spec is None:
            continue
        ax, lo, hi = int(spec[0]), float(spec[1]), float(spec[2])
        bad = []
        for f, snap in sorted(pose_track.items()):
            ch = snap[pb.name]["channels"]
            if "rotation_euler_deg" not in ch:
                continue
            v = ch["rotation_euler_deg"][ax]
            if v < lo - cfg.joint_tolerance_deg or v > hi + cfg.joint_tolerance_deg:
                bad.append((f, v))
        if bad:
            worst = max(bad, key=lambda x: max(lo - x[1], x[1] - hi))
            out.append({
                "bone": pb.name, "axis": "XYZ"[ax], "limit_deg": [lo, hi],
                "frames": [bad[0][0], bad[-1][0]], "frames_violating": len(bad),
                "worst_value_deg": round(worst[1], 1), "worst_frame": worst[0],
            })
    for n, j in enumerate(out, 1):
        j["id"] = f"J{n}"
    return out


def _gait_check(arm, pose_track: dict) -> dict | None:
    """歩行が消されていないか、前に歩いているか。

    - rig["gr_min_stride_deg"]: 太ももの振り幅 (X 回転の範囲) の下限
    - rig["gr_forward_axis"]: 体の前方向 (world)。root の移動方向と足先 (foot ボーン) の向きがこれと一致するか
    """
    need = arm.get("gr_min_stride_deg")
    if need is None:
        return None
    ranges = {}
    for name in ("thigh.L", "thigh.R"):
        vals = [s[name]["channels"].get("rotation_euler_deg", [0, 0, 0])[0] for s in pose_track.values() if name in s]
        ranges[name] = round(max(vals) - min(vals), 1) if vals else 0.0
    fwd = Vector(arm.get("gr_forward_axis", [0.0, 1.0, 0.0])).normalized()
    frames = sorted(pose_track)
    r0 = pose_track[frames[0]]["root"]["world"].translation
    r1 = pose_track[frames[-1]]["root"]["world"].translation
    travel = (r1 - r0)
    along = travel.dot(fwd)
    toe = []
    for s in ("foot.L", "foot.R"):
        if s in pose_track[frames[0]]:
            for f in frames:
                m = pose_track[f][s]["world"]
                d = m.to_3x3().col[1]                  # ボーンの向き (かかと → つま先)
                toe.append(Vector((d.x, d.y, 0.0)).normalized().dot(fwd) if Vector((d.x, d.y)).length > 1e-6 else 0.0)
    toe_mean = sum(toe) / len(toe) if toe else 1.0
    direction_ok = along > 0.2 and toe_mean > 0.5
    return {"min_stride_deg": float(need), "thigh_swing_deg": ranges,
            "stride_ok": all(r >= float(need) for r in ranges.values()),
            "forward_axis": [round(x, 3) for x in fwd],
            "root_travel": [round(x, 3) for x in travel], "travel_along_forward": round(along, 3),
            "toe_alignment": round(toe_mean, 3), "direction_ok": direction_ok,
            "ok": all(r >= float(need) for r in ranges.values()) and direction_ok}


def _detect_imbalance(track: list[dict], cfg: HarnessConfig) -> dict:
    """重心投影が支持基底面の外(または接地なし)にある区間を列挙。

    歩行では重心が一瞬だけ足の外に出るのは普通なので、imbalance_min_frames 以上続いたら
    「立っていられない」(error)、imbalance_warn_frames 以上なら warning とする。
    """
    events, run = [], []

    def label(v):
        if not v:
            return None
        x, y = v
        if abs(y) >= abs(x):
            return "forward (+Y)" if y > 0 else "backward (-Y)"
        return "right (+X)" if x > 0 else "left (-X)"

    def flush():
        if len(run) >= cfg.imbalance_warn_frames:
            dists = [t["outside_support"] for t in run if t["outside_support"] is not None]
            worst = max(run, key=lambda t: t["outside_support"] if t["outside_support"] is not None else 1e9)
            vecs = [t["outside_vector"] for t in run if t.get("outside_vector")]
            mean = [sum(v[0] for v in vecs) / len(vecs), sum(v[1] for v in vecs) / len(vecs)] if vecs else None
            events.append({
                "frame_range": [run[0]["frame"], run[-1]["frame"]],
                "frames": len(run),
                "severity": "error" if len(run) >= cfg.imbalance_min_frames else "warning",
                "max_outside_distance": round(max(dists), 4) if dists else None,
                "worst_frame": worst["frame"],
                "direction": label(mean),
                "mean_outside_vector": [round(x, 3) for x in mean] if mean else None,
                "airborne_frames": sum(1 for t in run if t["contacts"] == 0),
                "com_at_worst": worst["com"],
            })
        run.clear()

    for t in track:
        bad = t["contacts"] == 0 or (t["outside_support"] or 0) > cfg.support_margin
        if bad:
            run.append(t)
        else:
            flush()
    flush()
    for n, e in enumerate(events, 1):
        e["id"] = f"B{n}"
    return {"events": events}


# ---------------------------------------------------------------------------
# Reflection Agent 向けプロンプト
# ---------------------------------------------------------------------------

EVIDENCE_LEVELS = ("L1", "L2", "L3")


def build_reflection_prompt(report: dict, iteration: int = 1, max_iterations: int = 3,
                            history: list[str] | None = None, structured: bool = False,
                            evidence: str = "L3") -> str:
    """検証レポートを LLM にそのまま渡せる構造化テキストへ変換する。

    evidence でヒントの量を変える (ベンチマークのアブレーション用)。
      L1: 測定値のみ (どのチェックが何件・どの値で落ちたか)
      L2: + 全頂点の頂点番号・ボーン距離、転倒中のボーン運動とローカル軸
      L3: + 「漏れ側はどちらか」の判定 (修正方針は system prompt 側で追加)
    数値はすべてハーネスが計測したもの。LLM には「原因の推定」と「修正 bpy コード」だけを求める。
    """
    if evidence not in EVIDENCE_LEVELS:
        raise ValueError(f"evidence must be one of {EVIDENCE_LEVELS}")
    lv = EVIDENCE_LEVELS.index(evidence) + 1
    t = report["target"]
    fr = report["frames"]
    L: list[str] = []
    L.append("# GaitReflect Validation Report")
    L.append(f"iteration: {iteration}/{max_iterations}")
    L.append(f"status: {report['status'].upper()}  (errors={report['counts']['errors']}, warnings={report['counts']['warnings']})")
    L.append("")
    L.append("## Target")
    L.append(f"- Blender version: {report['blender_version']}")
    L.append(f"- mesh object: `{t['mesh']}` ({t['vertices']} verts, {t['faces']} faces)")
    L.append(f"- armature object: `{t['armature']}`")
    parents = t.get("bone_parents") or {}
    if parents:
        L.append("- bones (parent): " + ", ".join(f"{b} ({p or '-'})" for b, p in sorted(parents.items())))
    else:
        L.append(f"- bones: {', '.join(t['bones'])}")
    L.append(f"- frames evaluated: {fr['start']}–{fr['end']} (step {fr['step']}, n={fr['evaluated']})")
    g = report.get("gait") or {}
    if g.get("forward_axis"):
        L.append(f"- world orientation: the robot faces and walks toward {g['forward_axis']} (+Y); its left side is -X; up is +Z")
    L.append("")

    # 歪み
    L.append("## Issues: Mesh distortion (face area vs. rest pose)")
    clusters = report["distortion"]["clusters"]
    if not clusters:
        L.append("- none")
    for c in clusters:
        L.append(f"### {c['id']} [{c['severity']}] {c['kind']} — {c['face_count']} faces")
        L.append(f"- vertex groups on these faces: {', '.join(c['vertex_groups']) or '(none)'}")
        L.append(f"- worst area ratio: {c['worst_ratio']}x at frame {c['worst_frame']}; flagged frames {c['frame_range'][0]}–{c['frame_range'][1]}")
        if lv >= 2 and c["leak_groups"]:
            L.append(f"- weight leaks detected toward non-adjacent bones: {', '.join(c['leak_groups'])}")
        if lv >= 2:
            sv = c["suspect_vertices"][:10]
            if sv:
                L.append("- suspect vertices (index: weights):")
                for s in sv:
                    L.append(f"  - v{s['vertex']}: {json.dumps(s['weights'])} — {s['reason']}")
                if len(c["suspect_vertices"]) > len(sv):
                    L.append(f"  - … {len(c['suspect_vertices']) - len(sv)} more")
            if c.get("related_audit"):
                L.append(f"- these faces are only the visible symptom; the full leak region is {', '.join(c['related_audit'])} below")
    L.append("")

    # ウェイト監査 (全頂点)
    L.append("## Issues: Weight audit (ALL vertices, rest pose)")
    audit = report.get("weight_audit", [])
    if not audit:
        L.append("- none")
    for w in audit:
        a, b = w["groups"]
        L.append(f"### {w['id']} [error] {w['vertex_count']} vertices weighted to both `{a}` and `{b}`, which are not parent/child")
        if lv >= 2:
            d = w["mean_distance_to_bone"]
            L.append(f"- vertex indices: {w['vertices']}")
            L.append(f"- mean distance from these vertices to bone: {a}={d[a]} m, {b}={d[b]} m")
        if lv >= 3:
            L.append(f"- likely leak: `{w['likely_leak_group']}` (weights {w['leak_weight_range'][0]}–{w['leak_weight_range'][1]}); keep: `{w['keep_group']}`")
    uw = report.get("unweighted_vertices") or []
    if uw:
        L.append(f"### U1 [error] {len(uw)} vertices have no bone weight at all")
        if lv >= 2:
            L.append(f"- vertex indices: {_compact_ranges(uw)}")
    L.append("")

    # バランス (支持基底面)。v2 ではここが「立っていられない」の判定
    L.append("## Issues: Balance (center-of-mass projection vs. the feet's support polygon)")
    ev = report["balance"]["events"]
    cfg_ = report.get("config", {})
    L.append(f"- rule: the CoM's ground projection may leave the polygon spanned by ground-contact vertices "
             f"(within {cfg_.get('contact_epsilon', 0.03)} m of the floor) by at most {cfg_.get('support_margin', 0.05)} m; "
             f"{cfg_.get('imbalance_min_frames', 8)}+ consecutive frames outside = the robot cannot stay upright (error). "
             "Brief excursions during normal walking are allowed.")
    if not ev:
        L.append("- none")
    for e in ev:
        L.append(f"### {e['id']} [{e['severity']}] CoM outside the support polygon for {e['frames']} consecutive frames "
                 f"({e['frame_range'][0]}–{e['frame_range'][1]}), toward {e['direction']}")
        L.append(f"- max distance outside: {e['max_outside_distance']} m at frame {e['worst_frame']} "
                 f"(CoM {e['com_at_worst']}); airborne frames: {e['airborne_frames']}")
        bm_ = e.get("bone_motion")
        if bm_ and lv >= 2:
            ref, f1 = bm_["reference_frame"], bm_["worst_frame"]
            L.append(f"- gait period (estimated from thigh swing): {bm_['period']} frames")
            L.append(f"- bone channels at frame {f1} (worst) compared with frame {ref} ({bm_['method']}); "
                     "deviation = change that is NOT part of the normal walk cycle (root's steady forward travel removed):")
            L.append("  | bone | own rot deviation (deg) | own loc deviation |")
            L.append("  |---|---|---|")
            for x in bm_["bones"]:
                L.append(f"  | {x['bone']} | {x['rot_deviation_deg']} | {x['loc_deviation']} |")
            L.append("- pose channel values of the top bones (bone-local space):")
            for x in bm_["bones"][:3]:
                L.append(f"  - {x['bone']}: frame {ref} {json.dumps(x['channels_reference'])} → frame {f1} {json.dumps(x['channels_at_worst'])}")
            L.append("- rest-pose local axes of those bones expressed in world space "
                     "(pose channels are in this local space, NOT world space; world up is +Z, forward is +Y):")
            for name, ax in bm_["rest_local_axes_in_world"].items():
                L.append(f"  - {name}: local X={ax['X']}, local Y={ax['Y']}, local Z={ax['Z']}")
    L.append("")

    # 床へのめり込み
    L.append("## Issues: Ground penetration (mesh below the floor, z < 0)")
    gp = report.get("ground_penetration") or []
    if not gp:
        L.append("- none")
    for g_ in gp:
        L.append(f"### {g_['id']} [error] vertices of `{g_['bone']}` go {g_['worst_depth']} m below the floor "
                 f"(worst at frame {g_['worst_frame']}); frames {g_['frames'][0]}–{g_['frames'][1]} ({g_['frames_below']} frames)")
        if lv >= 2:
            L.append(f"- deepest vertex: v{g_['worst_vertex']}")
    L.append("")

    # 動きのなめらかさ
    L.append("## Issues: Motion continuity (sudden jumps in a bone's own channels)")
    cfg_ = report.get("config", {})
    L.append(f"- rule: the frame-to-frame change of a bone's own rotation may not jump by {cfg_.get('pop_rot_deg', 15)}° or more "
             f"(second difference), nor its location by {cfg_.get('pop_loc', 0.04)} m or more")
    mp = report.get("motion_pops") or []
    if not mp:
        L.append("- none")
    for x in mp:
        L.append(f"### {x['id']} [error] `{x['bone']}` jumps at {x['count']} frame(s); worst at frame {x['worst_frame']} "
                 f"(rotation jump {x['worst_rot_jump_deg']}°, location jump {x['worst_loc_jump']} m)")
        if lv >= 2:
            L.append(f"- frames: {', '.join(str(f) for f in x['frames'])}")
    L.append("")

    # 関節可動域
    L.append("## Issues: Joint limits (rig spec)")
    joints = report.get("joint_limits") or []
    if not joints:
        L.append("- none")
    for j in joints:
        L.append(f"### {j['id']} [error] `{j['bone']}` own rotation {j['axis']} outside its allowed range {j['limit_deg'][0]}°–{j['limit_deg'][1]}°")
        L.append(f"- worst {j['worst_value_deg']}° at frame {j['worst_frame']}; violating frames {j['frames'][0]}–{j['frames'][1]} ({j['frames_violating']} frames)")
    L.append("")

    # 歩行が保たれているか
    g = report.get("gait")
    if g:
        L.append("## Constraint: Walk cycle must be preserved")
        sw = ", ".join(f"{k} {v}°" for k, v in g["thigh_swing_deg"].items())
        L.append(f"- each thigh must swing at least {g['min_stride_deg']}° (own X rotation range); now: {sw}")
        L.append(f"- the root must travel forward and the feet must point forward: root moved {g['root_travel']} "
                 f"({g['travel_along_forward']} m along forward), toe alignment {g['toe_alignment']} (1 = straight ahead)")
        if not g["stride_ok"]:
            L.append("### G1 [error] walk cycle was removed or flattened")
        if not g["direction_ok"]:
            L.append("### G2 [error] the robot no longer walks forward (root travel or toe direction is wrong)")
        L.append("")

    kc = report.get("keyed_channels", {})
    if kc:
        L.append("## Animation: keyed pose channels")
        for bone, props in sorted(kc.items()):
            parts = [f"{p} idx{v['indices']} frames {v['frames'][0]}–{v['frames'][1]} ({v['keys']} keys)" for p, v in props.items()]
            L.append(f"- {bone}: " + "; ".join(parts))
        L.append("")

    for n in report.get("notes", []):
        L.append(f"> note: {n}")

    if history:
        L.append("## Previous attempts (do not repeat failed fixes)")
        for h in history:
            L.append(f"- {h}")
        L.append("")

    L.append("## Your task")
    L.append("1. For each issue ID, state the most likely root cause in one line, citing the numbers above.")
    L.append("2. Write ONE Python script (bpy, Blender 4.x/5.x) that fixes the errors. Allowed edits:")
    L.append("   - vertex group weights on the mesh (remove leaks, normalize, split rigid parts to a single bone)")
    L.append("   - pose-bone IK / limit-rotation constraints and their angle limits")
    L.append("   - pose-bone keyframes (foot roll, hip height, stride length) within the evaluated frame range")
    L.append("3. Do NOT add/remove vertices, delete objects, rename bones, change the frame range, remove the walk cycle,")
    L.append("   or edit custom properties on the rig or bones (they hold the rig spec).")
    L.append("4. Reference objects only by the names given above. The script must be idempotent.")
    L.append("")
    if structured:
        # JSON スキーマで出力形式を強制する場合 (reflect_loop.py)
        L.append("Respond with JSON matching the provided schema: `analysis` (one entry per issue id),")
        L.append("`fix_code` (the complete bpy script, no markdown fences), and `expected`.")
        return "\n".join(L)
    L.append("## Output format")
    L.append("```")
    L.append("ANALYSIS:")
    L.append("- <issue id>: <root cause>")
    L.append("FIX:")
    L.append("```python")
    L.append("# bpy code")
    L.append("```")
    L.append("EXPECTED: <which metrics should change and to what>")
    L.append("```")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GaitReflect validation harness")
    p.add_argument("--mesh", required=True)
    p.add_argument("--armature", required=True)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--stretch", type=float, default=1.5)
    p.add_argument("--report-out", default=None)
    p.add_argument("--prompt-out", default=None)
    p.add_argument("--iteration", type=int, default=1)
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--history-json", default=None, help="過去の試行の要約 (JSON 文字列リストのファイル)")
    p.add_argument("--structured", action="store_true")
    p.add_argument("--evidence", default="L3", choices=["L1", "L2", "L3"])
    return p.parse_args(argv)


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    a = _parse_args(argv)
    scene = bpy.context.scene
    cfg = HarnessConfig(
        mesh_name=a.mesh, armature_name=a.armature,
        frame_start=a.start if a.start is not None else scene.frame_start,
        frame_end=a.end if a.end is not None else scene.frame_end,
        frame_step=a.step, stretch_ratio=a.stretch,
    )
    report = run_harness(cfg)
    history = None
    if a.history_json:
        with open(a.history_json, encoding="utf-8") as fh:
            history = json.load(fh)
    prompt = build_reflection_prompt(report, iteration=a.iteration, max_iterations=a.max_iterations,
                                     history=history, structured=a.structured, evidence=a.evidence)

    if a.report_out:
        with open(a.report_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
    if a.prompt_out:
        with open(a.prompt_out, "w", encoding="utf-8") as fh:
            fh.write(prompt)
    if not a.prompt_out:
        print(prompt)
    print(f"[GaitReflect] status={report['status']} errors={report['counts']['errors']} warnings={report['counts']['warnings']}")
    return 0


if __name__ == "__main__":
    main()
