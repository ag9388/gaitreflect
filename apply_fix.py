"""LLM が生成した修正 bpy コードを Blender 内で実行し、結果を別名保存する。

  blender -b in.blend --python-exit-code 1 --python apply_fix.py -- fix.py out.blend
"""

import math
import re
import sys
import traceback

import bmesh
import bpy
import bpy_extras
import mathutils

# LLM 生成コードに許さない操作 (ファイル/プロセス/ネットワーク/保存の乗っ取り)
BANNED = [
    r"\bimport\s+(os|sys|subprocess|shutil|socket|urllib|requests|pathlib|importlib)\b",
    r"\bfrom\s+(os|sys|subprocess|shutil|socket|urllib|requests|pathlib|importlib)\b",
    r"__import__", r"\beval\s*\(", r"\bexec\s*\(", r"\bopen\s*\(", r"\bcompile\s*\(",
    r"bpy\.ops\.wm\.", r"bpy\.data\.(objects|meshes|armatures)\.remove",
    # リグ仕様 (可動域・最小歩幅) の書き換え・削除は不正
    r"""\[\s*["']gr_\w+["']\s*\]\s*=""", r"""\bdel\b[^\n]*["']gr_""", r"""\.pop\(\s*["']gr_""",
]


def check(code: str) -> list[str]:
    return [p for p in BANNED if re.search(p, code)]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    fix_path, out_path = argv[0], argv[1]
    with open(fix_path, encoding="utf-8") as fh:
        code = fh.read()

    hits = check(code)
    if hits:
        print(f"[apply_fix] REJECTED: banned patterns {hits}")
        raise SystemExit(2)

    ns = {"bpy": bpy, "bmesh": bmesh, "mathutils": mathutils, "bpy_extras": bpy_extras, "math": math,
          "__name__": "__gaitreflect_fix__"}
    try:
        exec(compile(code, fix_path, "exec"), ns)
    except Exception:
        print("[apply_fix] FIX RAISED:")
        traceback.print_exc()
        raise SystemExit(1)

    bpy.context.scene.frame_set(bpy.context.scene.frame_start)
    bpy.ops.wm.save_as_mainfile(filepath=out_path, copy=True)
    print(f"[apply_fix] OK -> {out_path}")


main()
