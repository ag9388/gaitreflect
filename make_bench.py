"""GaitReflect-Bench v2: 不具合入りロボット 30 問を生成し、全問が「解ける」ことを検証する。

  python3 make_bench.py --out episodes

各問について次を確認し、満たさなければシードを変えて作り直す:
  1. 何もしないと FAIL し、仕込んだ不具合の種類がすべて検出される
  2. 仕込んでいない種類のエラーは出ない (例: ウェイト漏れのせいでバランスエラーが出る、は不可)
  3. 模範解答 (oracle) を当てると 1 回で PASS する
  4. 前に歩いている: root の移動方向と足先の向きが、どちらも前 (+Y)
API は使わない。
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent

BONE_PARENTS = {
    "root": None, "torso": "root",
    "thigh.L": "root", "shin.L": "thigh.L", "foot.L": "shin.L",
    "thigh.R": "root", "shin.R": "thigh.R", "foot.R": "shin.R",
    "arm.L": "torso", "arm.R": "torso",
}
WEIGHTED_PARTS = ["torso", "thigh.L", "shin.L", "foot.L", "thigh.R", "shin.R", "foot.R", "arm.L", "arm.R"]
FRAMES = 60

# 30 問の構成: 単独 10 / 2 種 12 / 3 種 8 (v1 と同じ。fall → balance)
PLAN = (
    [["leak"]] * 4 + [["balance"]] * 3 + [["knee"]] * 3
    + [["leak", "balance"]] * 4 + [["leak", "knee"]] * 4 + [["balance", "knee"]] * 4
    + [["leak", "balance", "knee"]] * 6 + [["leak", "leak", "balance"]] * 2
)


def neighbors(b: str) -> set[str]:
    s = {b}
    if BONE_PARENTS[b]:
        s.add(BONE_PARENTS[b])
    s.update(c for c, p in BONE_PARENTS.items() if p == b)
    return s


def swing_windows(side: str, period: int) -> list[tuple[int, int]]:
    """その脚が地面から離れている (遊脚) フレーム区間の一覧。gen_episode.gait_tables と同じ位相。"""
    out, off = [], 0.0 if side == "L" else 0.5
    cur = None
    for f in range(1, FRAMES + 1):
        ph = ((f - 1) / period + off) % 1.0
        sw = ph >= 0.5
        if sw and cur is None:
            cur = f
        if not sw and cur is not None:
            out.append((cur, f - 1)); cur = None
    if cur is not None:
        out.append((cur, FRAMES))
    return out


def make_spec(kinds: list[str], rng: random.Random) -> dict:
    robot = {
        "leg_len": round(rng.uniform(0.85, 1.15), 3),
        "hip_w": round(rng.uniform(0.13, 0.19), 3),
        "leg_th": round(rng.uniform(0.12, 0.16), 3),
        "torso_w": round(rng.uniform(0.5, 0.7), 3),
        "torso_d": round(rng.uniform(0.26, 0.34), 3),
        "torso_h": round(rng.uniform(0.42, 0.6), 3),
        "arm_len": round(rng.uniform(0.4, 0.55), 3),
        "arm_th": round(rng.uniform(0.08, 0.11), 3),
        "foot_len": round(rng.uniform(0.28, 0.36), 3),
        "foot_w": round(rng.uniform(0.22, 0.28), 3),
        "head_w": round(rng.uniform(0.22, 0.28), 3),
        "period": rng.choice([24, 30, 36]),
        "amp": round(rng.uniform(0.24, 0.34), 3),
        "knee_stance": round(rng.uniform(5, 12), 1),
        "knee_swing": round(rng.uniform(40, 52), 1),
        "arm_amp": round(rng.uniform(0.2, 0.45), 3),
        "sway": round(rng.uniform(0.05, 0.07), 3),
    }
    defects, used_parts, used_sides = [], set(), set()
    for k in kinds:
        if k == "leak":
            part = rng.choice([p for p in WEIGHTED_PARTS if p not in used_parts])
            used_parts.add(part)
            to = rng.choice([b for b in BONE_PARENTS if b not in neighbors(part)])
            defects.append({"type": "leak", "part": part, "leak_to": to,
                            "region": rng.choice(["lower", "all"]), "weight": round(rng.uniform(0.4, 1.0), 2)})
        elif k == "balance":
            mode = rng.choice(["lean", "slide"])
            d = {"type": "balance", "mode": mode, "direction": rng.choice(["left", "right"]),
                 "start": rng.randint(22, 34), "ramp": rng.randint(8, 12)}
            if mode == "lean":
                d["angle"] = rng.randint(38, 50)
            else:
                d["offset"] = round(rng.uniform(0.18, 0.26), 3)
            defects.append(d)
        elif k == "knee":
            side = rng.choice([s for s in "LR" if s not in used_sides])
            used_sides.add(side)
            P = robot["period"]
            wins = [w for w in swing_windows(side, P) if w[1] - w[0] + 1 >= 8]
            w0, w1 = rng.choice(wins)
            length = rng.randint(6, min(12, w1 - w0 + 1 - 2))
            start = rng.randint(w0 + 1, w1 - length + 1)
            defects.append({"type": "knee", "side": side, "start": start, "length": length,
                            "angle": rng.randint(15, 35)})
    return {"robot": robot, "defects": defects}


def run(cmd: list[str], log: Path) -> int:
    with open(log, "w", encoding="utf-8") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=900).returncode


def check_detection(report: dict, truth: list[dict]) -> list[str]:
    """仕込んだ不具合が検出されたか、仕込んでいない種類のエラーが出ていないか。問題点の一覧を返す。"""
    problems = []
    kinds = {d["type"] for d in truth}
    for d in truth:
        if d["type"] == "leak":
            if not any(set(w["groups"]) == {d["part"], d["leak_to"]} for w in report["weight_audit"]):
                problems.append(f"missed leak {d['part']}<-{d['leak_to']}")
        elif d["type"] == "balance":
            if not any(e["severity"] == "error" for e in report["balance"]["events"]):
                problems.append(f"missed balance {d['mode']} {d['direction']}")
        elif d["type"] == "knee":
            if not any(j["bone"] == f"shin.{d['side']}" for j in report["joint_limits"]):
                problems.append(f"missed knee {d['side']}")
    if "balance" not in kinds and any(e["severity"] == "error" for e in report["balance"]["events"]):
        problems.append("unplanned balance error")
    knee_bones = {f"shin.{d['side']}" for d in truth if d["type"] == "knee"}
    if any(d["type"] == "balance" and d["mode"] == "lean" for d in truth):
        knee_bones.add("torso")          # 倒れ込み (lean) は上半身の傾き上限にもかかる
    for j in report["joint_limits"]:
        if j["bone"] not in knee_bones:
            problems.append(f"unplanned joint error {j['bone']}")
    leak_pairs = [{d["part"], d["leak_to"]} for d in truth if d["type"] == "leak"]
    for w in report["weight_audit"]:
        if set(w["groups"]) not in leak_pairs:
            problems.append(f"unplanned audit {w['groups']}")
    if "knee" not in kinds and report.get("ground_penetration"):
        problems.append("unplanned ground penetration")
    for x in report.get("motion_pops") or []:      # 膝の過伸展は前後で角度が跳ぶので、すねだけは許す
        if x["bone"] not in {f"shin.{d['side']}" for d in truth if d["type"] == "knee"}:
            problems.append(f"unplanned motion pop {x['bone']}")
    g = report.get("gait") or {}
    if not g.get("ok", True):
        problems.append("gait check failed")
    if report["unweighted_vertices"]:
        problems.append("unweighted vertices")
    return problems


def build_episode(n: int, kinds: list[str], out: Path) -> dict:
    work = out / "_work" / f"ep{n:02d}"
    work.mkdir(parents=True, exist_ok=True)
    log = []
    for attempt in range(10):
        seed = 1000 * attempt + n
        spec = make_spec(kinds, random.Random(seed))
        spec_p, blend_p, res_p = work / "spec.json", out / f"ep{n:02d}.blend", work / "result.json"
        spec_p.write_text(json.dumps(spec, indent=1))
        rc = run([sys.executable, str(HERE / "blender_run.py"), "-", str(HERE / "gen_episode.py"),
                  "--", str(spec_p), str(blend_p), str(res_p)], work / "gen.log")
        if rc != 0:
            raise RuntimeError(f"ep{n:02d}: generator crashed, see {work / 'gen.log'}")
        res = json.loads(res_p.read_text())
        oracle_p = out / f"ep{n:02d}.oracle.py"
        oracle_p.write_text(res["oracle_code"])

        vdir = work / f"check{attempt}"
        rc = run([sys.executable, str(HERE / "reflect_loop.py"), str(blend_p), "--mesh", "Robot", "--armature", "Rig",
                  "--mock", "--mock-fix", str(oracle_p), "--max-iter", "1", "--out-dir", str(vdir)], work / "check.log")
        tl = json.loads((vdir / "timeline.json").read_text()) if (vdir / "timeline.json").exists() else []
        if len(tl) < 2:
            log.append(f"seed {seed}: initial check did not fail or loop crashed (rc={rc})")
            continue
        report1 = json.loads((vdir / "report1.json").read_text())
        problems = check_detection(report1, res["truth"])
        if problems:
            log.append(f"seed {seed}: {problems}")
            continue
        if tl[-1]["status"] == "fail":
            log.append(f"seed {seed}: oracle did not pass (errors={tl[-1]['errors']})")
            continue
        final = json.loads((vdir / "report2.json").read_text())
        g = final["gait"]
        return {
            "id": f"ep{n:02d}", "seed": seed, "kinds": kinds,
            "blend": blend_p.name, "oracle": oracle_p.name,
            "initial_errors": tl[0]["errors"], "robot": spec["robot"], "truth": res["truth"],
            "oracle_final_status": final["status"],
            "walk_check": {"root_travel_forward_m": g["travel_along_forward"], "toe_alignment": g["toe_alignment"]},
            "rejected_seeds": log,
        }
    raise RuntimeError(f"ep{n:02d}: no valid seed after 10 attempts: {log}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "episodes"))
    ap.add_argument("--only", type=int, nargs="*", help="この番号だけ作る (デバッグ用)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(n, kinds) for n, kinds in enumerate(PLAN, 1) if not a.only or n in a.only]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(lambda j: build_episode(j[0], j[1], out), jobs))
    for ep in results:
        print(f"{ep['id']} OK  {'+'.join(ep['kinds']):22s} seed={ep['seed']:5d} initial_errors={ep['initial_errors']} "
              f"oracle={ep['oracle_final_status']} rejected={len(ep['rejected_seeds'])}")
    if not a.only:
        (out / "manifest.json").write_text(json.dumps(results, indent=1))
        print(f"wrote {len(results)} episodes → {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
