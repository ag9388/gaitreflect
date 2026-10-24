"""GaitReflect-Bench v2 を Beam で解く (30 問 × 証拠レベル L1/L2/L3)。

  export HF_TOKEN=...                             # Hugging Face Inference Providers 経由 (既定)
  python3 run_bench.py                            # 全 90 本
  python3 run_bench.py --limit 3                  # 試運転: 3 問 × 3 レベル
  python3 run_bench.py --mock-oracle              # API なしでパイプラインだけ確認
  python3 run_bench.py --token-budget 2200000     # 合計トークンがこれを超えたら、新しいジョブを始めない

途中で止めても、もう一度実行すれば終わった分は飛ばして続きから再開する。
最後に bench/results/summary.md と summary.json を書く。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE
LEVELS = ["L1", "L2", "L3"]
LEVEL_NAMES = {"L1": "numbers only", "L2": "+ indices & bone motion", "L3": "+ near-answer hints"}


QUOTA_HIT = threading.Event()   # 日次上限 / トークン予算に達したら、以降のジョブは始めない
TOKENS_USED = [0]
TOKENS_LOCK = threading.Lock()


def job_done(d: Path) -> bool:
    """結果があり、API エラーで中断していないもの。"""
    p = d / "timeline.json"
    return p.exists() and not any("llm_error" in s for s in json.loads(p.read_text()))


def run_job(ep: dict, level: str, a, eps_dir: Path, out_dir: Path) -> dict:
    d = out_dir / level / ep["id"]
    if not job_done(d):
        if QUOTA_HIT.is_set() or (a.token_budget and TOKENS_USED[0] >= a.token_budget):
            return {"episode": ep["id"], "level": level, "kinds": ep["kinds"], "complete": False, "skipped": True}
        shutil.rmtree(d, ignore_errors=True)      # 中断した回の残骸を消してからやり直す
        d.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "reflect_loop.py"), str(eps_dir / ep["blend"]),
               "--mesh", "Robot", "--armature", "Rig", "--evidence", level,
               "--max-iter", str(a.max_iter), "--effort", a.effort, "--out-dir", str(d), "--provider", a.provider]
        if a.mock_oracle:
            cmd += ["--mock", "--mock-fix", str(eps_dir / ep["oracle"])]
        t0 = time.time()
        with open(d / "run.log", "w", encoding="utf-8") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=3600)
        (d / "wall_seconds.txt").write_text(f"{time.time() - t0:.1f}")
        tl_p = d / "timeline.json"
        if tl_p.exists() and "daily_allowance_exhausted" in tl_p.read_text():
            QUOTA_HIT.set()
    r = score(ep, level, d)
    with TOKENS_LOCK:
        TOKENS_USED[0] += r["tokens"]
    return r


def score(ep: dict, level: str, d: Path) -> dict:
    tl = json.loads((d / "timeline.json").read_text()) if (d / "timeline.json").exists() else []
    api_error = any("llm_error" in s for s in tl)
    final = tl[-1] if tl else {}
    solved = bool(tl) and final.get("status") != "fail"
    rounds = len(tl) - 1 if tl else None           # Beam に修正させた回数
    crashes = sum(1 for s in tl if "apply_error" in s)
    tokens = sum((s.get("usage") or {}).get("total_tokens", 0) for s in tl)
    secs = sum(s.get("llm_seconds", 0) for s in tl)
    return {
        "episode": ep["id"], "level": level, "kinds": ep["kinds"],
        "solved": solved, "rounds": rounds, "crashes": crashes, "api_error": api_error,
        "initial_errors": tl[0]["errors"] if tl else None, "final_errors": final.get("errors"),
        "tokens": tokens, "llm_seconds": round(secs, 1), "complete": bool(tl) and not api_error,
        # どの経路で Beam を呼んだか (古い結果には記録がないので reflection 直接とみなす)
        "provider": sorted({(s.get("request_mode") or {}).get("provider", "?") for s in tl if "usage" in s}),
        "request_modes": sorted({(s.get("request_mode") or {}).get("response_format", "json_schema") for s in tl if "usage" in s}),
    }


def summarize(rows: list[dict], max_iter: int) -> tuple[dict, str]:
    by_level = defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(r)
    summ = {"levels": {}, "by_kind": {}, "by_count": {}}
    md = ["| Evidence | Solved in 1 round | Solved within 3 | Avg rounds (solved) | Scripts crashed | Errors left | Tokens / episode |",
          "|---|---|---|---|---|---|---|"]
    for lv in LEVELS:
        rs = [r for r in by_level.get(lv, []) if r["complete"]]
        if not rs:
            continue
        n = len(rs)
        s1 = sum(1 for r in rs if r["solved"] and r["rounds"] == 1)
        s3 = sum(1 for r in rs if r["solved"])
        avg = sum(r["rounds"] for r in rs if r["solved"]) / max(1, s3)
        scripts = sum(r["rounds"] for r in rs)
        crashed = sum(r["crashes"] for r in rs)
        left = sum(r["final_errors"] or 0 for r in rs) / max(1, sum(r["initial_errors"] or 0 for r in rs))
        tok = sum(r["tokens"] for r in rs) / n
        summ["levels"][lv] = {"n": n, "solved_1": s1, "solved_any": s3, "avg_rounds": round(avg, 2),
                              "scripts": scripts, "crashed": crashed, "errors_left_ratio": round(left, 3),
                              "tokens_per_episode": round(tok)}
        md.append(f"| {lv} · {LEVEL_NAMES[lv]} | {s1}/{n} ({s1/n:.0%}) | {s3}/{n} ({s3/n:.0%}) | {avg:.2f} | "
                  f"{crashed}/{scripts} | {left:.0%} | {tok:,.0f} |")
        for kind in ("leak", "balance", "knee"):
            sub = [r for r in rs if kind in r["kinds"]]
            summ["by_kind"].setdefault(kind, {})[lv] = [sum(r["solved"] for r in sub), len(sub)]
        for c in (1, 2, 3):
            sub = [r for r in rs if len(r["kinds"]) == c]
            summ["by_count"].setdefault(str(c), {})[lv] = [sum(r["solved"] for r in sub), len(sub)]
    md += ["", "Solved within 3 rounds, by defect type / number of defects:", "",
           "| Subset | " + " | ".join(lv for lv in LEVELS if lv in summ["levels"]) + " |",
           "|---|" + "---|" * len(summ["levels"])]
    for name, table in [(f"contains {k}", summ["by_kind"].get(k, {})) for k in ("leak", "balance", "knee")] + \
                       [(f"{c} defect{'s' if c != '1' else ''}", summ["by_count"].get(c, {})) for c in ("1", "2", "3")]:
        cells = [f"{table[lv][0]}/{table[lv][1]}" if lv in table else "–" for lv in LEVELS if lv in summ["levels"]]
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    incomplete = [f"{r['level']}/{r['episode']}" for r in rows if not r["complete"]]
    if incomplete:
        md += ["", f"Not counted (API error or not run yet): {', '.join(incomplete)}"]
    return summ, "\n".join(md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=str(HERE / "episodes"))
    ap.add_argument("--token-budget", type=int, default=0, help="合計トークンの上限 (0 = なし)")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--levels", nargs="+", default=LEVELS, choices=LEVELS)
    ap.add_argument("--limit", type=int, default=None, help="先頭 N 問だけ")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--provider", default="hf", choices=["reflection", "hf"],
                    help="hf = Hugging Face 経由 (HF_TOKEN が必要、HF クレジットで課金)")
    ap.add_argument("--mock-oracle", action="store_true", help="API を使わず模範解答で流す (配線確認)")
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from reflect_loop import PROVIDERS, get_api_key
    key_env = PROVIDERS[a.provider]["key_env"]
    has_key = get_api_key(a.provider)
    if not a.mock_oracle and not has_key:
        print(f"{key_env} が未設定です。")
        return 2
    eps_dir, out_dir = Path(a.episodes), Path(a.out)
    if a.mock_oracle:
        out_dir = out_dir.with_name(out_dir.name + "_mock")
    manifest = json.loads((eps_dir / "manifest.json").read_text())[: a.limit]
    jobs = [(ep, lv) for ep in manifest for lv in a.levels]
    print(f"{len(jobs)} runs ({len(manifest)} episodes × {len(a.levels)} levels), {a.workers} in parallel, "
          f"provider={a.provider} → {out_dir}")

    rows, lock, done = [], threading.Lock(), 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_job, ep, lv, a, eps_dir, out_dir): (ep, lv) for ep, lv in jobs}
        for fut in as_completed(futs):
            ep, lv = futs[fut]
            try:
                r = fut.result()
            except Exception as e:      # 1 本の失敗で全体を止めない
                r = {"episode": ep["id"], "level": lv, "kinds": ep["kinds"], "complete": False, "error": str(e)}
            with lock:
                rows.append(r)
                done += 1
                if r.get("complete"):
                    msg = (f"PASS in {r['rounds']} round{'s' if r['rounds'] != 1 else ''}" if r["solved"]
                           else f"FAIL ({r['initial_errors']}→{r['final_errors']} errors)")
                elif r.get("skipped"):
                    msg = "skipped (daily allowance or token budget used up)"
                else:
                    msg = "API ERROR / incomplete"
                print(f"[{done:3d}/{len(jobs)}] {ep['id']} {lv}  {'+'.join(ep['kinds']):16s} {msg}", flush=True)

    rows.sort(key=lambda r: (r["level"], r["episode"]))
    summ, md = summarize([r for r in rows if "solved" in r], a.max_iter)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs.json").write_text(json.dumps(rows, indent=1))
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1))
    (out_dir / "summary.md").write_text(md + "\n")
    print("\n" + md)
    if QUOTA_HIT.is_set():
        print("\nReflection の 1 日の出力トークン上限に達しました。00:00 UTC (日本時間 9:00) 以降に、"
              "同じコマンドで残りを再開できます。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
