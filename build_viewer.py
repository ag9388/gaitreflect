"""ベンチマーク結果から、Space 用のデータ (3D モデル・要約・GIF) を作る。

  python3 build_viewer.py --results results --out site

各エピソードについて「壊れた入力」と「L1/L2/L3 で Beam が最後に出した修正」の 4 体を bake.py で焼き込み
(立っていられないものは Bullet で倒す)、次を書き出す:
  site/bench_models.bin    .glb を連結したもの (120 体)
  site/bench_viewer.json   エピソードごとの不具合タグ・結果・Beam の最終診断・モデルの位置
  site/bench.json          要約 (レベル別・不具合の数別・種類別)
  site/bench_runs.json     各ラウンドの診断・スクリプト・エラー (ページの「全ラウンド」表示用)
  site/bake/*.npz          GIF 用の各フレームのメッシュ
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEVELS = ["L1", "L2", "L3"]
FRAMES_SHOWN = 96


def truth_tags(truth: list[dict]) -> list[str]:
    out = []
    for d in truth:
        if d["type"] == "leak":
            out.append(f"weight leak: {d['part']} ← {d['leak_to']} ({len(d['vertices'])} verts)")
        elif d["type"] == "balance":
            if d["mode"] == "lean":
                out.append(f"balance loss: torso leans {d['direction']} {d['angle']}° from frame {d['start']} → CoM leaves the feet")
            else:
                out.append(f"balance loss: torso slides {d['direction']} {d['offset']:.2f} m from frame {d['start']} → CoM leaves the feet")
        elif d["type"] == "knee":
            f0, f1 = d["frames"]
            out.append(f"knee hyperextension: shin.{d['side']} bends forward past straight (+{d['angle']}°), frames {f0}–{f1}")
    return out


def bake(blend: Path, prefix: Path) -> dict:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    log = prefix.with_suffix(".log")
    with open(log, "w") as fh:
        rc = subprocess.run([sys.executable, str(HERE / "blender_run.py"), str(blend), str(HERE / "bake.py"), "--",
                             str(prefix), "--frames", str(FRAMES_SHOWN)], stdout=fh, stderr=subprocess.STDOUT,
                            timeout=900).returncode
    if rc != 0 or not prefix.with_suffix(".glb").exists():
        raise RuntimeError(f"bake failed for {blend}: see {log}")
    import numpy as np
    d = np.load(prefix.with_suffix(".npz"))
    return {"fall_onset": int(d["fall_onset"]), "fall_kind": str(d["fall_kind"]),
            "status": str(d["status"]), "errors": int(d["errors"])}


def final_blend(run_dir: Path) -> Path:
    tl = json.loads((run_dir / "timeline.json").read_text())
    return run_dir / tl[-1]["blend"]


def rounds_of(run_dir: Path) -> list[dict]:
    tl = json.loads((run_dir / "timeline.json").read_text())
    out = []
    for step in tl:
        i = step["iteration"]
        r = {"round": i, "status": step["status"], "errors": step["errors"], "warnings": step["warnings"]}
        resp_p = run_dir / f"response{i}.json"
        if resp_p.exists():
            resp = json.loads(resp_p.read_text())
            r.update({"analysis": resp["analysis"], "fix_code": resp["fix_code"], "expected": resp["expected"]})
            if step.get("usage"):
                r["tokens"] = step["usage"].get("total_tokens")
            if step.get("llm_seconds"):
                r["seconds"] = step["llm_seconds"]
            if step.get("apply_error"):
                r["apply_error"] = step["apply_error"].replace(str(run_dir), "<run>")
        out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--episodes", default=str(HERE / "episodes"))
    ap.add_argument("--out", default=str(HERE / "site"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    res, eps, out = Path(a.results), Path(a.episodes), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((eps / "manifest.json").read_text())
    runs = {(r["episode"], r["level"]): r for r in json.loads((res / "runs.json").read_text())}
    summary = json.loads((res / "summary.json").read_text())

    jobs = []
    for ep in manifest:
        jobs.append((ep["id"], "broken", eps / ep["blend"]))
        for lv in LEVELS:
            jobs.append((ep["id"], lv, final_blend(res / lv / ep["id"])))
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        baked = dict(zip([(j[0], j[1]) for j in jobs],
                         ex.map(lambda j: bake(j[2], out / "bake" / f"{j[0]}_{j[1]}"), jobs)))

    blob, episodes, all_runs = bytearray(), [], {}
    for ep in manifest:
        models = {}
        for key in ["broken", *LEVELS]:
            g = (out / "bake" / f"{ep['id']}_{key}.glb").read_bytes()
            models[key] = [len(blob), len(g)]
            blob += g
            while len(blob) % 4:
                blob += b"\0"
        levels = {}
        for lv in LEVELS:
            r = runs[(ep["id"], lv)]
            rd = rounds_of(res / lv / ep["id"])
            all_runs[f"{ep['id']}/{lv}"] = rd
            last = [x for x in rd if "analysis" in x]
            diag = [f"{x['issue_id']}: {x['root_cause']}" for x in (last[-1]["analysis"] if last else [])]
            levels[lv] = {"solved": r["solved"], "rounds": r["rounds"], "final_errors": r["final_errors"],
                          "tokens": r["tokens"], "crashes": r["crashes"], "diagnosis": diag,
                          "fall_onset": baked[(ep["id"], lv)]["fall_onset"],
                          "fall_kind": baked[(ep["id"], lv)]["fall_kind"]}
        episodes.append({
            "id": ep["id"], "defects": ep["kinds"], "truth": truth_tags(ep["truth"]),
            "initial_errors": ep["initial_errors"], "models": models, "levels": levels,
            "fall_onset": baked[(ep["id"], "broken")]["fall_onset"],
            "fall_kind": baked[(ep["id"], "broken")]["fall_kind"],
            "walk_check": ep["walk_check"],
        })
    (out / "bench_models.bin").write_bytes(bytes(blob))
    (out / "bench_viewer.json").write_text(json.dumps({"frames": FRAMES_SHOWN, "walk_frames": 60,
                                                        "episodes": episodes}, ensure_ascii=False))
    (out / "bench_runs.json").write_text(json.dumps(all_runs, ensure_ascii=False))
    notes_p = HERE / "bench_notes.json"          # 失敗パターンなど、結果を読んで手で書いた注釈
    notes = json.loads(notes_p.read_text()) if notes_p.exists() else {}
    (out / "bench.json").write_text(json.dumps({**summary, **notes}, indent=1, ensure_ascii=False))
    print(f"wrote {len(episodes)} episodes, {len(blob) / 1e6:.1f} MB of models → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
