"""表示用の焼き込み: 検証 → (立っていられなければ) Bullet で倒す → 各フレームのメッシュと .glb を書き出す。

  python3 blender_run.py rig.blend bake.py -- out_prefix [--frames 84]

出力:
  out_prefix.npz   verts[frame, vertex, 3], faces, face_mat, stretch[frame, face], com[frame, 3],
                   outside[frame], fall_onset (0 = 倒れない)
  out_prefix.glb   スキン付きメッシュ + 歩行アニメーション (+ 倒れる区間は Rig オブジェクトの剛体運動)

倒れる動きはキーフレームではなく物理で作る:
  バランスの error (重心が足の外に一定フレーム以上) が始まったフレームで姿勢を固定し、
  メッシュの部品ごと (箱ごと) の凸包を剛体として Bullet に渡して、重力で倒す。
  初速は、そのフレームの重心速度。.blend 自体は変更しない (検証は常に元のアニメーションで行う)。
"""

import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402

FPS = 24


def evaluated_world(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    oe = obj.evaluated_get(dg)
    me = oe.to_mesh()
    mw = oe.matrix_world
    co = np.array([tuple(mw @ v.co) for v in me.vertices], dtype=np.float64)
    oe.to_mesh_clear()
    return co


def islands(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    seen, out = set(), []
    for v in bm.verts:
        if v.index in seen:
            continue
        stack, comp = [v], []
        seen.add(v.index)
        while stack:
            x = stack.pop()
            comp.append(x.index)
            for e in x.link_edges:
                y = e.other_vert(x)
                if y.index not in seen:
                    seen.add(y.index)
                    stack.append(y)
        out.append(sorted(comp))
    bm.free()
    return out


def face_areas(co, faces):
    out = np.zeros(len(faces))
    for i, f in enumerate(faces):
        p = co[f]
        a = 0.0
        for k in range(1, len(f) - 1):
            a += 0.5 * np.linalg.norm(np.cross(p[k] - p[0], p[k + 1] - p[0]))
        out[i] = a
    return out


def simulate_fall(co0, parts, faces, com_vel, n_frames):
    """固定した姿勢 co0 を剛体として倒す。各フレームの (回転行列, 並進) を返す (co = R @ (co0 - c0) + c)。"""
    import pybullet as p

    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.setTimeStep(1 / 240, physicsClientId=cid)
    plane = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=cid)
    ground = p.createMultiBody(0, plane, physicsClientId=cid)
    p.changeDynamics(ground, -1, lateralFriction=0.9, restitution=0.0, physicsClientId=cid)

    # 質量は表面積に比例 (ハーネスの重心と同じ定義)
    areas = face_areas(co0, faces)
    vert_area = np.zeros(len(co0))
    for f, a in zip(faces, areas):
        vert_area[f] += a / len(f)
    total = vert_area.sum()
    c0 = (vert_area[:, None] * co0).sum(0) / total
    lift = max(0.0, -co0[:, 2].min()) + 0.002
    shapes, masses, centers = [], [], []
    for part in parts:
        pts = co0[part]
        m = max(1e-3, 60.0 * vert_area[part].sum() / total)       # 全体 60 kg
        ctr = pts.mean(0)
        sid = p.createCollisionShape(p.GEOM_MESH, vertices=(pts - ctr).tolist(), physicsClientId=cid)
        shapes.append(sid); masses.append(m); centers.append(ctr)
    # 1 つ目の部品を base、残りを固定ジョイントの link にする (= 1 つの剛体)
    base_pos = centers[0] + np.array([0, 0, lift])
    body = p.createMultiBody(
        baseMass=masses[0], baseCollisionShapeIndex=shapes[0], basePosition=base_pos.tolist(),
        linkMasses=masses[1:], linkCollisionShapeIndices=shapes[1:], linkVisualShapeIndices=[-1] * (len(shapes) - 1),
        linkPositions=[(c - centers[0]).tolist() for c in centers[1:]],
        linkOrientations=[[0, 0, 0, 1]] * (len(shapes) - 1),
        linkInertialFramePositions=[[0, 0, 0]] * (len(shapes) - 1),
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * (len(shapes) - 1),
        linkParentIndices=[0] * (len(shapes) - 1),
        linkJointTypes=[p.JOINT_FIXED] * (len(shapes) - 1),
        linkJointAxis=[[0, 0, 1]] * (len(shapes) - 1),
        physicsClientId=cid)
    for j in range(-1, len(shapes) - 1):
        p.changeDynamics(body, j, lateralFriction=0.9, restitution=0.0, physicsClientId=cid)
    p.resetBaseVelocity(body, linearVelocity=list(com_vel), angularVelocity=[0, 0, 0], physicsClientId=cid)

    out = []
    for _ in range(n_frames):
        for _ in range(240 // FPS):
            p.stepSimulation(physicsClientId=cid)
        pos, orn = p.getBasePositionAndOrientation(body, physicsClientId=cid)
        R = Quaternion((orn[3], orn[0], orn[1], orn[2])).to_matrix()
        out.append((np.array(R), np.array(pos)))
    p.disconnect(cid)
    # base 部品の中心 (centers[0]) を基準にした剛体変換に直す
    return out, centers[0], lift


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    prefix = Path(argv[0])
    n_total = int(argv[argv.index("--frames") + 1]) if "--frames" in argv else 84
    scene = bpy.context.scene
    robot, rig = bpy.data.objects["Robot"], bpy.data.objects["Rig"]
    f_start, f_end = scene.frame_start, scene.frame_end

    rep = harness.run_harness(harness.HarnessConfig("Robot", "Rig", f_start, f_end))
    errs = [e for e in rep["balance"]["events"] if e["severity"] == "error"]
    onset = errs[0]["frame_range"][0] if errs else 0
    # 足が床から浮いている (接地なし) のが主な理由なら "airborne"。物理では落ちて着地する
    fall_kind = ("airborne" if errs[0]["airborne_frames"] * 2 >= errs[0]["frames"] else "tip") if errs else ""

    me = robot.data
    faces = [list(pl.vertices) for pl in me.polygons]
    face_mat = [pl.material_index for pl in me.polygons]
    parts = islands(me)

    rig.data.pose_position = "REST"
    bpy.context.view_layer.update()
    rest_area = face_areas(evaluated_world(robot), faces)
    rig.data.pose_position = "POSE"

    verts = []
    last_anim = onset if onset else f_end
    for f in range(f_start, last_anim + 1):
        scene.frame_set(f)
        verts.append(evaluated_world(robot))

    track = {t["frame"]: t for t in rep["com_track"]}
    if onset:
        com = np.array(track[onset]["com"])
        prev = np.array(track[max(f_start, onset - 1)]["com"])
        vel = (com - prev) * FPS
        co0 = verts[-1]
        sim, base_c, lift = simulate_fall(co0, parts, faces, vel, n_total - onset)
        # Bullet の base 姿勢 (R, pos) → メッシュ全体の剛体変換 x_t = R @ x0 + t
        # (初期は向き単位・位置 p0 = base_c + lift。浮かせた lift の分は表示では戻す)
        L = np.array([0, 0, lift])
        p0 = base_c + L
        rigid = []
        for R, pos in sim:
            t = R @ (L - p0) + pos - L
            rigid.append((R, t))
            verts.append((R @ co0.T).T + t)
    else:
        # 倒れない: 歩行の最後の姿勢で止まる (表示の長さをそろえる)
        for _ in range(n_total - f_end):
            verts.append(verts[-1].copy())
        rigid = []

    V = np.stack(verts)
    stretch = np.stack([face_areas(v, faces) / np.maximum(rest_area, 1e-9) for v in V])
    out_list = []
    for k in range(len(V)):
        f = f_start + k
        out_list.append(track[f]["outside_support"] if f in track and track[f]["outside_support"] is not None else 0.0)
    fa = np.zeros(max(len(fc) for fc in faces), dtype=np.int32)
    faces_pad = np.array([fc + [-1] * (len(fa) - len(fc)) for fc in faces], dtype=np.int32)
    vert_area = np.zeros(V.shape[1])
    for fc, a in zip(faces, rest_area):
        vert_area[fc] += a / len(fc)
    np.savez_compressed(prefix.with_suffix(".npz"), verts=V.astype(np.float32), faces=faces_pad,
                        face_mat=np.array(face_mat, dtype=np.int8), stretch=stretch.astype(np.float32),
                        outside=np.array(out_list, dtype=np.float32), fall_onset=np.int32(onset),
                        fall_kind=np.array(fall_kind),
                        status=np.array(rep["status"]), errors=np.int32(rep["counts"]["errors"]))

    # --- glTF: 倒れる区間は姿勢を固定し、Rig オブジェクトに剛体運動のキーを打つ -------------------
    if onset:
        ad = rig.animation_data
        scene.frame_set(onset)
        held = {pb.name: (pb.location.copy(), pb.rotation_euler.copy()) for pb in rig.pose.bones}
        for k, (Rr, t) in enumerate(rigid, 1):
            f = onset + k
            for pb in rig.pose.bones:
                pb.location, pb.rotation_euler = held[pb.name]
                pb.keyframe_insert("location", frame=f)
                pb.keyframe_insert("rotation_euler", frame=f)
            M = Matrix([list(Rr[i]) + [t[i]] for i in range(3)] + [[0, 0, 0, 1]])
            rig.matrix_world = M
            rig.keyframe_insert("location", frame=f)
            rig.keyframe_insert("rotation_euler", frame=f)
        rig.matrix_world = Matrix.Identity(4)
        rig.keyframe_insert("location", frame=onset)
        rig.keyframe_insert("rotation_euler", frame=onset)
        rig.keyframe_insert("location", frame=f_start)
        rig.keyframe_insert("rotation_euler", frame=f_start)
        scene.frame_end = n_total
    else:
        scene.frame_end = n_total
    for o in bpy.data.objects:
        o.select_set(o.name in ("Robot", "Rig"))
    bpy.ops.export_scene.gltf(filepath=str(prefix.with_suffix(".glb")), export_format="GLB", use_selection=True,
                              export_animations=True, export_frame_range=True, export_force_sampling=True,
                              export_def_bones=False, export_apply=False, export_yup=True,
                              export_animation_mode="SCENE", export_materials="EXPORT")
    print(f"[bake] {prefix.name}: status={rep['status']} fall_onset={onset} frames={len(V)}")


main()
