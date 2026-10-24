"""pip 版 bpy で「blender -b file.blend --python script.py -- args」と同じことをする。

  python3 blender_run.py file.blend script.py -- args...
  python3 blender_run.py - script.py -- args...      # 空のシーンから

v1 は Blender.app を subprocess で呼んでいた。この環境には Blender 本体がないので、
.blend を開いてからスクリプトを __main__ として実行する。スクリプトの例外は終了コード 1 になる。
"""

import runpy
import sys

import bpy


def main() -> int:
    argv = sys.argv[1:]
    blend, script = argv[0], argv[1]
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    if blend != "-":
        bpy.ops.wm.open_mainfile(filepath=blend, load_ui=False)
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    sys.argv = ["blender", "--", *rest]
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0) if not isinstance(e.code, str) else 1
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # bpy モジュールの後始末で落ちることがあるので、そのまま終了する
    import os
    os._exit(code)
