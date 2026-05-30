"""Patch-and-Predict — AnchorSlicer切分 + VLM提取 + 滑动扫描"""

import cv2, os, base64, requests, sys, subprocess, time, hashlib
sys.path.insert(0, ".")
from src.core.anchor_slicer import AnchorSlicer

ADB = r"C:\Program Files\scrcpy\scrcpy-win64-v3.1\adb.exe"
DEV = "182f4e2f"
MODEL = "minicpm-v:latest"
PROMPT = "直接读出这张商品卡片上显示的商品标题。忽略价格和促销标签。只返回标题文字."


def screenshot(path="captures/now.png"):
    subprocess.run([ADB, "-s", DEV, "shell", "screencap", "-p", "/sdcard/screen.png"])
    subprocess.run([ADB, "-s", DEV, "pull", "/sdcard/screen.png", path])


def scroll_down():
    # 70%卡高 — 主动重叠，每张卡至少在一屏完整
    subprocess.run([ADB, "-s", DEV, "shell", "input", "swipe", "610", "1000", "610", "1700", "500"])


def run_scan(store_name="", max_screens=10) -> list[str]:
    """滑动扫描全店商品，去重提取标题"""
    import easyocr
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
    seen = set()
    all_titles = []

    # 先滚回顶部
    for _ in range(3):
        subprocess.run([ADB, "-s", DEV, "shell", "input", "swipe", "610", "2200", "610", "400", "300"])
        time.sleep(1)

    for screen in range(max_screens):
        print(f"\n=== Screen {screen+1} ===")
        screenshot()
        img = cv2.imread("captures/now.png")
        if img is None:
            break

        # 检查画面是否变化
        if screen > 0:
            prev = cv2.imread(f"captures/screen_{screen-1:02d}.png")
            if prev is not None and cv2.absdiff(img, prev).mean() < 1.0:
                print("  No change -> end")
                break

        cv2.imwrite(f"captures/screen_{screen:02d}.png", img)

        # 切分
        slicer = AnchorSlicer()
        slicer.extract_anchors(img, reader)
        cols, col_anchors = slicer.detect_layout()
        cards = slicer.slice(img, col_anchors)
        print(f"  {len(cards)} cards")

        # 保存每屏卡片
        screen_dir = f"captures/cards/s{screen}"
        os.makedirs(screen_dir, exist_ok=True)
        for i, card in enumerate(cards):
            cv2.imwrite(f"{screen_dir}/c{i}.png", card)

        # 提取标题
        for i, card in enumerate(cards):
            # 去重
            gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
            tiny = cv2.resize(gray, (16, 16))
            phash = hashlib.md5(tiny.tobytes()).hexdigest()[:8]
            if phash in seen:
                continue
            seen.add(phash)

            _, buf = cv2.imencode(".png", card)
            b64 = base64.b64encode(buf).decode()
            try:
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": MODEL, "prompt": PROMPT, "images": [b64],
                          "stream": False, "options": {"temperature": 0, "num_predict": 128}},
                    timeout=120,
                )
                title = resp.json()["response"].strip()
                if title and len(title) >= 4:
                    all_titles.append(title)
                    print(f"    [{len(all_titles)}] {title}")
            except Exception:
                pass

        scroll_down()
        time.sleep(3)

    # 输出
    print(f"\n=== {store_name}: {len(all_titles)} products ===")
    for i, t in enumerate(all_titles):
        print(f"  {i+1}. {t}")

    with open(f"data/{store_name}_titles.txt", "w", encoding="utf-8") as f:
        for t in all_titles:
            f.write(t + "\n")

    return all_titles


if __name__ == "__main__":
    run_scan("朱公子上岸教育")
