"""GaitReflect Reflection ループ v2 (Beam を Reflection Agent として使う)

  export HF_TOKEN=...                  # Hugging Face Inference Providers 経由 (既定)
  python3 reflect_loop.py bad.blend --mesh Robot --armature Rig --evidence L2

  # API を使わずにパイプラインだけ確認
  python3 reflect_loop.py bad.blend --mesh Robot --armature Rig --mock

流れ (最大 --max-iter 回):
  検証(harness.py) → FAIL なら Beam に構造化レポートを渡す → 修正 bpy コードを受け取る
  → Blender で適用(apply_fix.py) → 再検証 …  エラー 0 で終了。
Blender は pip の bpy を blender_run.py 経由で別プロセスとして動かす (v1 は Blender.app)。
それ以外は標準ライブラリのみ。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Beam の呼び先。Hugging Face 経由 (Inference Providers) は HF クレジットで課金され、Reflection の日次上限とは別枠
PROVIDERS = {
    "reflection": {"base": "https://api.reflection.ai/openai/v1", "model": "Beam-1.0-501B-A23B",
                   "key_env": "REFLECTION_API_KEY"},
    "hf": {"base": "https://router.huggingface.co/v1", "model": "reflection/Beam-1.0-501B-A23B",
           "key_env": "HF_TOKEN"},
}
API_BASE = PROVIDERS["reflection"]["base"]
DEFAULT_MODEL = PROVIDERS["reflection"]["model"]

SYSTEM_PROMPT = """You are GaitReflect, a rigging & locomotion repair agent for Blender.
You receive a deterministic validation report measured by bpy/bmesh (face-area ratios vs. rest pose,
vertex-group weights, center-of-mass trajectory, support polygon). Trust these numbers; do not invent others.
Diagnose each issue id, then write one minimal, idempotent bpy script (Blender 4.x/5.x) that fixes the errors.
Prefer the smallest change that makes the metrics pass. The script runs headless: no bpy.ops.wm, no file or OS access,
no imports other than bpy, bmesh, mathutils, math, bpy_extras. Objects are already in the scene; reference them by name.

Blender API notes (the report states the exact version; these apply to 4.4+ and 5.x):
- `Action.fcurves` DOES NOT EXIST (removed with slotted/layered actions). Do not touch fcurves directly.
  Preferred way to change animation: for each frame in the range, set the pose-bone channel values and call
  `pose_bone.keyframe_insert("location", frame=f)` / `keyframe_insert("rotation_euler", frame=f)`; this overwrites existing keys.
  If you must read F-Curves: `from bpy_extras import anim_utils;
  cb = anim_utils.action_get_channelbag_for_slot(rig.animation_data.action, rig.animation_data.action_slot); cb.fcurves`.
- Vertex groups live on the OBJECT: `obj.vertex_groups[name]` (never `obj.data.vertex_groups`).
  `vg.remove([indices])` removes vertices from a group; `vg.add([indices], weight, "REPLACE")` sets weights.
  Read a weight via `{obj.vertex_groups[g.group].name: g.weight for g in obj.data.vertices[i].groups}`.
- Pose-bone `location` / `rotation_*` are in the bone's LOCAL rest space, relative to the rest pose (zero = rest).
  Do not assume local Z is world up; if the report lists a bone's local axes in world space, use them.
  Check `pose_bone.rotation_mode` before writing `rotation_euler` vs `rotation_quaternion`.
- Rig-spec joint limits, the minimum stride and the forward axis are stored as custom properties; reading them is fine,
  editing is not."""

# 証拠レベル L3 のみ: 修正方針そのもののヒント
STRATEGY_HINTS = """

Repair strategies:
- A balance loss caused by a bone's own keyed rotation/location is fixed by returning those keys to their normal
  walk-cycle values (the same phase one gait cycle earlier, or the pre-onset values), not by offsetting other bones.
- When the weight audit lists a likely leak group, remove that group from ALL listed vertex indices, not only the
  vertices on the stretched faces.
- A joint-limit violation is fixed by returning that bone's keyed rotation to its normal walk-cycle values on the
  violating frames (e.g. the same phase one gait cycle earlier or later). Merely clamping to the limit can leave the
  leg straight and push the foot into the floor.
- Fix the whole defective window, not only the frames a check flagged: values must stay continuous from frame to frame
  (the verifier rejects sudden jumps), so restore every frame of the affected range."""


def system_prompt(evidence: str) -> str:
    return SYSTEM_PROMPT + (STRATEGY_HINTS if evidence == "L3" else "")

FIX_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue_id": {"type": "string"},
                    "root_cause": {"type": "string"},
                },
                "required": ["issue_id", "root_cause"],
                "additionalProperties": False,
            },
        },
        "fix_code": {"type": "string"},
        "expected": {"type": "string"},
    },
    "required": ["analysis", "fix_code", "expected"],
    "additionalProperties": False,
}



# ---------------------------------------------------------------------------

def run_blender(blend: Path, script: Path, args: list[str], log: Path, timeout: int = 600) -> int:
    cmd = [sys.executable, str(HERE / "blender_run.py"), str(blend), str(script), "--", *args]
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode


def _post(base: str, api_key: str, body: dict) -> dict:
    """chat/completions を呼ぶ。レート制限・一時的な障害は待って再試行する。"""
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:800]
            if "daily_allowance_exhausted" in msg:      # 日次上限は待っても戻らない
                raise RuntimeError(f"API {e.code}: {msg}") from None
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(15 * (attempt + 1))
                continue
            raise ApiHTTPError(e.code, msg) from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 3:
                time.sleep(15 * (attempt + 1))
                continue
            raise RuntimeError(f"API connection error: {e}") from None
    raise RuntimeError("unreachable")


class ApiHTTPError(RuntimeError):
    def __init__(self, code: int, msg: str):
        super().__init__(f"API {code}: {msg}")
        self.code, self.msg = code, msg


def call_beam(prompt: str, model: str, effort: str, api_key: str, system: str = SYSTEM_PROMPT,
              max_tokens: int = 16000, provider: str = "reflection") -> dict:
    """Beam に修正を依頼し、スキーマどおりの JSON を返す。

    プロバイダが json_schema や reasoning_effort を受け付けない場合は、
    json_object + プロンプト内のスキーマ指定 → 指定なし、の順に自動で落とす。
    """
    base = PROVIDERS[provider]["base"]
    schema_note = ("\n\nReturn ONLY a JSON object with exactly these keys: "
                   "analysis (array of {issue_id: string, root_cause: string}), fix_code (string), expected (string).")
    modes = [("json_schema", True), ("json_object", True), ("json_object", False), ("none", False)]
    last_err = None
    for fmt, with_effort in modes:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system + ("" if fmt == "json_schema" else schema_note)},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": max_tokens,
        }
        if with_effort:
            body["reasoning_effort"] = effort
        if fmt == "json_schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "gaitreflect_fix", "schema": FIX_SCHEMA, "strict": True}}
        elif fmt == "json_object":
            body["response_format"] = {"type": "json_object"}
        try:
            data = _post(base, api_key, body)
        except ApiHTTPError as e:
            # パラメータ非対応 (400/422) のときだけ、次の簡易モードで再挑戦する
            if e.code in (400, 422):
                last_err = e
                continue
            raise
        choice = data["choices"][0]
        msg = choice["message"]
        if choice.get("finish_reason") not in ("stop", None) or msg.get("refusal") or not msg.get("content"):
            raise RuntimeError(f"no structured output (finish_reason={choice.get('finish_reason')}, refusal={msg.get('refusal')})")
        out = json.loads(strip_fences(msg["content"]))
        for k in ("analysis", "fix_code", "expected"):
            if k not in out:
                raise RuntimeError(f"response JSON is missing '{k}'")
        out["_usage"] = data.get("usage")
        out["_reasoning"] = msg.get("reasoning_content")
        out["_request_mode"] = {"provider": provider, "response_format": fmt, "reasoning_effort": with_effort}
        return out
    raise RuntimeError(f"provider rejected every request mode; last error: {last_err}")


KEY_FILE = HERE / "reflection_key.txt"


def read_key_file() -> str | None:
    """reflection_key.txt の、# で始まらない最初の行をキーとして読む。"""
    if not KEY_FILE.exists():
        return None
    for line in KEY_FILE.read_text(encoding="utf-8").splitlines():
        s = line.strip().strip("'\"")
        if s.startswith("export REFLECTION_API_KEY="):
            s = s.split("=", 1)[1].strip("'\"")
        if s and not s.startswith("#"):
            return s
    return None


def get_api_key(provider: str) -> str | None:
    """環境変数 → (reflection) reflection_key.txt / (hf) Hugging Face 標準のトークンファイル の順に探す。"""
    if provider == "reflection":
        return os.environ.get("REFLECTION_API_KEY") or read_key_file()
    key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if key:
        return key
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    tok = hf_home / "token"
    return tok.read_text().strip() or None if tok.exists() else None


def strip_fences(code: str) -> str:
    s = code.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        s = s.rsplit("```", 1)[0]
    return s


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="GaitReflect reflection loop on Reflection API")
    ap.add_argument("blend")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--armature", required=True)
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--provider", default=os.environ.get("BEAM_PROVIDER", "hf"), choices=list(PROVIDERS),
                    help="reflection = Reflection API 直接 / hf = Hugging Face Inference Providers 経由")
    ap.add_argument("--model", default=os.environ.get("REFLECTION_MODEL"))
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--mock", action="store_true", help="API を呼ばず --mock-fix のスクリプトを使う (模範解答の検証用)")
    ap.add_argument("--mock-fix", default=None, help="--mock 時に使う修正スクリプト (模範解答の検証用)")
    ap.add_argument("--max-tokens", type=int, default=16000, help="1 回の応答の出力トークン上限")
    ap.add_argument("--evidence", default="L3", choices=["L1", "L2", "L3"], help="レポートのヒント量")
    a = ap.parse_args()

    prov = PROVIDERS[a.provider]
    a.model = a.model or prov["model"]
    api_key = get_api_key(a.provider)
    if a.mock and not a.mock_fix:
        print("--mock には --mock-fix が必要です")
        return 2
    if not a.mock and not api_key:
        print(f"API キーが見つかりません。{prov['key_env']} を設定してください。--mock で疎通だけ確認できます。")
        return 2

    run_dir = Path(a.out_dir or HERE / "runs" / time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    current = run_dir / "iter0.blend"
    shutil.copy(a.blend, current)

    history: list[str] = []
    timeline: list[dict] = []
    hist_path = run_dir / "history.json"

    for i in range(1, a.max_iter + 2):  # 最後の 1 回は修正後の最終検証
        report_p, prompt_p = run_dir / f"report{i}.json", run_dir / f"prompt{i}.md"
        hist_path.write_text(json.dumps(history, ensure_ascii=False))
        h_args = ["--mesh", a.mesh, "--armature", a.armature,
                  "--report-out", str(report_p), "--prompt-out", str(prompt_p),
                  "--iteration", str(i), "--max-iterations", str(a.max_iter),
                  "--history-json", str(hist_path), "--structured", "--evidence", a.evidence]
        if a.start is not None:
            h_args += ["--start", str(a.start)]
        if a.end is not None:
            h_args += ["--end", str(a.end)]
        rc = run_blender(current, HERE / "harness.py", h_args, run_dir / f"harness{i}.log")
        if rc != 0:
            print(f"harness failed (see {run_dir / f'harness{i}.log'})")
            return 1

        report = json.loads(report_p.read_text())
        c = report["counts"]
        step = {"iteration": i, "blend": current.name, "status": report["status"], **c}
        timeline.append(step)
        print(f"[iter {i}] {report['status'].upper():4}  errors={c['errors']} warnings={c['warnings']}  ({current.name})")
        if history:
            history[-1] += f" → result: {report['status']} (errors={c['errors']}, warnings={c['warnings']})"

        if report["status"] != "fail" or i > a.max_iter:
            break

        # --- Reflection --------------------------------------------------------
        if a.mock:
            code = Path(a.mock_fix).read_text()
            resp = {"analysis": [{"issue_id": "*", "root_cause": "mock"}], "fix_code": code, "expected": "mock"}
        else:
            t0 = time.time()
            try:
                resp = call_beam(prompt_p.read_text(), a.model, a.effort, api_key, system_prompt(a.evidence), a.max_tokens, a.provider)
            except Exception as e:
                print(f"[iter {i}] Reflection API error: {e}")
                step["llm_error"] = str(e)
                break
            step["llm_seconds"] = round(time.time() - t0, 1)
            step["usage"] = resp.get("_usage")
            step["request_mode"] = resp.get("_request_mode")
        (run_dir / f"response{i}.json").write_text(json.dumps(resp, indent=2, ensure_ascii=False))
        for an in resp["analysis"]:
            print(f"    {an['issue_id']}: {an['root_cause']}")

        fix_p = run_dir / f"fix{i}.py"
        fix_p.write_text(strip_fences(resp["fix_code"]))
        nxt = run_dir / f"iter{i}.blend"
        rc = run_blender(current, HERE / "apply_fix.py", [str(fix_p), str(nxt)], run_dir / f"apply{i}.log")
        summary = "; ".join(f"{x['issue_id']}: {x['root_cause']}" for x in resp["analysis"])
        if rc != 0:
            err = (run_dir / f"apply{i}.log").read_text()[-1200:]
            print(f"[iter {i}] fix failed to apply (rc={rc})")
            step["apply_error"] = err
            history.append(f"iteration {i}: diagnosis [{summary}] — script FAILED to run:\n{err}")
            continue  # 同じ blend のまま、エラーを履歴に載せて再試行
        history.append(f"iteration {i}: diagnosis [{summary}]; expected: {resp['expected']}")
        current = nxt

    (run_dir / "timeline.json").write_text(json.dumps(timeline, indent=2, ensure_ascii=False))
    final = timeline[-1]
    print(f"\nfinal: {final['status'].upper()} after {len(timeline) - 1} fix iteration(s) → {run_dir}")
    return 0 if final["status"] != "fail" else 1


if __name__ == "__main__":
    sys.exit(main())
