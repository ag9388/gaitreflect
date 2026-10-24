"""ベンチマークの問題と結果を、Hugging Face データセット用のフォルダにまとめる (v2)。

  python3 build_dataset.py --out dataset [--gifs site/gifs]

ローカルパスと Beam の推論過程 (reasoning_content) は含めない。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE
QUOTED_PATH = re.compile(r'"([^"]*/)([^"/]+)"')
BARE_PATH = re.compile(r"(/(?:Users|private|tmp|var|home|root)/[^\s\"']*/)([^/\s\"']+)")


def scrub(text: str) -> str:
    text = QUOTED_PATH.sub(lambda m: f'"{m.group(2)}"', text)
    return BARE_PATH.sub(lambda m: m.group(2), text)


def traceback_tail(log: str) -> str:
    lines = [l for l in log.splitlines() if l.strip()]
    start = next((i for i, l in enumerate(lines) if l.startswith("Traceback") or "REJECTED" in l), max(0, len(lines) - 8))
    keep = [l for l in lines[start:] if not l.startswith(("Blender ", "[apply_fix] FIX RAISED"))]
    return scrub("\n".join(keep[-10:]))


def run_record(ep: dict, level: str, d: Path) -> dict:
    tl = json.loads((d / "timeline.json").read_text())
    rounds = []
    for s in tl:
        i = s["iteration"]
        resp_p = d / f"response{i}.json"
        if not resp_p.exists():
            continue
        resp = json.loads(resp_p.read_text())
        after = tl[i] if i < len(tl) else None          # 次の検証結果 (適用に失敗した回は同じ blend の再検証)
        rounds.append({
            "round": i,
            "prompt": scrub((d / f"prompt{i}.md").read_text()),
            "analysis": resp["analysis"],
            "fix_code": (d / f"fix{i}.py").read_text(),
            "expected": resp["expected"],
            "script_error": traceback_tail(s["apply_error"]) if "apply_error" in s else None,
            "errors_before": s["errors"],
            "errors_after": after["errors"] if after else None,
            "status_after": after["status"] if after else None,
            "llm_seconds": s.get("llm_seconds"),
            "usage": s.get("usage"),
            "provider": (s.get("request_mode") or {}).get("provider"),
        })
    final = tl[-1]
    return {
        "episode": ep["id"], "evidence_level": level, "defects": ep["kinds"],
        "num_defects": len(ep["kinds"]),
        "solved": final["status"] != "fail", "fix_rounds": len(tl) - 1,
        "solved_in_one_round": final["status"] != "fail" and len(tl) == 2,
        "script_crashes": sum(1 for s in tl if "apply_error" in s),
        "initial_errors": tl[0]["errors"], "final_errors": final["errors"],
        "total_tokens": sum((s.get("usage") or {}).get("total_tokens", 0) for s in tl),
        "provider": sorted({r["provider"] for r in rounds}),
        "rounds": rounds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=str(HERE / "episodes"))
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=str(ROOT / "dataset"))
    ap.add_argument("--gifs", default=None, help="エピソードごとの GIF (epXX.gif) があるフォルダ")
    a = ap.parse_args()
    eps_dir, res_dir, out = Path(a.episodes), Path(a.results), Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    manifest = json.loads((eps_dir / "manifest.json").read_text())

    with open(out / "episodes.jsonl", "w", encoding="utf-8") as fh:
        for ep in manifest:
            fh.write(json.dumps({
                "episode": ep["id"], "defects": ep["kinds"], "num_defects": len(ep["kinds"]),
                "initial_errors": ep["initial_errors"], "seed": ep["seed"], "robot": ep["robot"],
                "ground_truth": ep["truth"], "oracle_fix": (eps_dir / ep["oracle"]).read_text(),
                "blend_file": f"episodes/{ep['blend']}",
                "walk_check": ep["walk_check"],
                "gif": f"gifs/{ep['id']}.gif" if a.gifs else None,
            }, ensure_ascii=False) + "\n")

    with open(out / "runs.jsonl", "w", encoding="utf-8") as fh:
        for level in ("L1", "L2", "L3"):
            for ep in manifest:
                fh.write(json.dumps(run_record(ep, level, res_dir / level / ep["id"]), ensure_ascii=False) + "\n")

    with zipfile.ZipFile(out / "episodes.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for ep in manifest:
            z.write(eps_dir / ep["blend"], f"episodes/{ep['blend']}")
            z.write(eps_dir / ep["oracle"], f"episodes/{ep['oracle']}")
        z.writestr("episodes/manifest.json", json.dumps(manifest, indent=1))

    for f in ("summary.md", "summary.json"):
        shutil.copy(res_dir / f, out / f)
    for f in ("harness.py", "reflect_loop.py", "apply_fix.py", "blender_run.py", "gen_episode.py", "make_bench.py",
              "run_bench.py", "bake.py", "render_gif.py", "build_viewer.py", "build_dataset.py"):
        shutil.copy(ROOT / f, out / f)
    if a.gifs:
        (out / "gifs").mkdir()
        for ep in manifest:
            shutil.copy(Path(a.gifs) / f"{ep['id']}.gif", out / "gifs" / f"{ep['id']}.gif")

    # 公開前の安全確認: ローカルパス・鍵らしき文字列が残っていないこと
    for p in out.iterdir():
        if p.is_dir() or p.suffix in (".zip", ".gif", ".png"):
            continue
        s = p.read_text(encoding="utf-8")
        home = str(Path.home()) + "/"      # "/root" だけだと文中の "foot/root" に一致してしまう
        if p.suffix == ".py":       # スクリプト自身は検出用の文字列 "/Users/" を含むので、実際のホームだけ見る
            assert home not in s, f"local path in {p.name}"
        else:
            assert "/Users/" not in s and "/private/" not in s and "/home/user" not in s and "/tmp/claude" not in s \
                and home not in s, f"local path in {p.name}"
        jwt_head = "eyJhbG" + "ciOi"    # 分割して書く (このファイル自身に一致しないように)
        assert not re.search(r"\bhf_[A-Za-z0-9]{20,}|" + jwt_head, s), f"secret-like string in {p.name}"
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
