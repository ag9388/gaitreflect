"""ローカルでページを開いてスクリーンショットを撮る (Playwright + プリインストールの Chromium)。

  python3 tools/shoot.py site_dir out_dir [--width 1280] [--episode ep23] [--frame 40]

CDN (cdn.jsdelivr.net) に届かない環境向けに、three.js (同じ 0.160.0) を npm から取ってきた
ローカルのファイルに差し替える (THREE_DIR)。公開ページの import map はそのまま (CDN) 。
"""

import argparse
import http.server
import os
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

THREE = Path(os.environ.get("THREE_DIR", "three/package"))   # npm pack three@0.160.0 を展開した場所


def serve(root: str) -> int:
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=root, **k)  # noqa: E731
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--episode", default=None)
    ap.add_argument("--frame", type=float, default=None)
    ap.add_argument("--dark", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    port = serve(a.site)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium", args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
        ctx = b.new_context(viewport={"width": a.width, "height": 900}, device_scale_factor=1,
                            color_scheme="dark" if a.dark else "light")
        page = ctx.new_page()
        logs = []
        page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))

        def three(route):
            rel = route.request.url.split("three@0.160.0/")[1]
            f = THREE / rel
            route.fulfill(path=str(f), content_type="application/javascript") if f.exists() else route.abort()
        page.route("https://cdn.jsdelivr.net/npm/three@0.160.0/**", three)
        page.goto(f"http://127.0.0.1:{port}/index.html", wait_until="networkidle")
        page.wait_for_timeout(2500)
        if a.episode:
            page.select_option("#epSelect", a.episode)
            page.wait_for_timeout(2500)
        if a.frame is not None:
            page.click("#benchBtn")          # 一時停止
            page.evaluate(f"""() => {{ const s = document.getElementById('benchSlider'); s.value = {a.frame};
                                       s.dispatchEvent(new Event('input')); }}""")
            page.wait_for_timeout(1200)
        tag = f"{a.width}{'_dark' if a.dark else ''}"
        page.screenshot(path=f"{a.out}/full_{tag}.png", full_page=True)
        page.locator("#explore").scroll_into_view_if_needed()
        page.wait_for_timeout(800)
        page.locator("#benchViewers").screenshot(path=f"{a.out}/viewers_{tag}.png")
        sw = page.evaluate("() => document.documentElement.scrollWidth")
        print(f"scrollWidth={sw} viewport={a.width}")
        for line in logs:
            print(line)
        b.close()


if __name__ == "__main__":
    main()
