"""Patch-and-Predict V2 — 70%滑动 + 增强锚点 + 懒加载等待 + 标题去重"""

import cv2, os, base64, requests, sys, subprocess, time, hashlib, numpy as np
sys.path.insert(0, ".")
from src.core.anchor_slicer import AnchorSlicer

ADB = r"C:\Program Files\scrcpy\scrcpy-win64-v3.1\adb.exe"
DEV = "182f4e2f"
MODEL = "minicpm-v:latest"
PROMPT = "直接读出这张商品卡片上显示的商品标题。忽略价格和促销标签。只返回标题文字."


def screenshot(path="captures/now.png"):
    subprocess.run([ADB, "-s", DEV, "shell", "screencap", "-p", "/sdcard/screen.png"])
    subprocess.run([ADB, "-s", DEV, "pull", "/sdcard/screen.png", path])


def wait_stable(patience=3):
    """懒加载等待：连续截两张图，差值<0.5%说明渲染完毕"""
    for _ in range(patience):
        screenshot("captures/tmp1.png")
        time.sleep(0.3)
        screenshot("captures/tmp2.png")
        a = cv2.imread("captures/tmp1.png")
        b = cv2.imread("captures/tmp2.png")
        if a is not None and b is not None:
            diff = np.abs(a.astype(float) - b.astype(float)).mean() / 255 * 100
            if diff < 0.5:
                return True
        time.sleep(0.5)
    return True  # 兜底


def scroll_70pct():
    """黄金70%滑动：卡高~440px，滑~300px"""
    subprocess.run([ADB, "-s", DEV, "shell", "input", "swipe", "610", "1200", "610", "1500", "500"])


def scroll_to_top():
    """回到顶部"""
    for _ in range(4):
        subprocess.run([ADB, "-s", DEV, "shell", "input", "swipe", "610", "2200", "610", "400", "300"])
        time.sleep(1.5)


def is_marketing(text: str) -> bool:
    """过滤非商品：纯营销词、太短、VLM错误"""
    if len(text) < 6:
        return True
    if "无法" in text and "提供" in text:
        return True
    mkt = {"精选习题","学考无忧","全程通关","长期有效","高清视频",
           "配套讲义","优师教学","快速提分","学员好评","深挖出题",
           "技巧实战","面试实战","逆向思维","框架构建"}
    if any(m in text for m in mkt):
        return True
    return False


def levenshtein(s1: str, s2: str) -> int:
    """编辑距离"""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1!=c2)))
        prev = curr
    return prev[-1]


def dedup_titles(titles: list[str]) -> list[str]:
    """模糊去重：编辑距离 < 5 且提取核心词相同 → 合并"""
    # 先过滤非商品
    clean = [t for t in titles if not is_marketing(t)]

    unique = []
    for t in clean:
        merged = False
        for i, u in enumerate(unique):
            # 子串包含 → 保留长的
            if t in u or u in t:
                if len(t) > len(u):
                    unique[i] = t
                merged = True
                break
            # 编辑距离 < 6 且核心词匹配 → 合并
            if levenshtein(t[:10], u[:10]) <= 5:
                if len(t) > len(u):
                    unique[i] = t
                merged = True
                break
        if not merged:
            unique.append(t)
    return unique


def ask_model_count(img_path: str) -> int:
    """让minicpm-v数一下有几个商品"""
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": MODEL, "prompt": "这一屏有几个商品？只回答数字。",
                  "images": [b64], "stream": False,
                  "options": {"temperature": 0, "num_predict": 8}},
            timeout=30,
        )
        ans = resp.json()["response"].strip()
        return int("".join(c for c in ans if c.isdigit()) or "0")
    except Exception:
        return 0


def run_scan(store_name="", max_screens=15) -> list[str]:
    print(f"=== {store_name} ===\n")
    import easyocr
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
    all_titles = []

    # 回到顶部
    scroll_to_top()
    time.sleep(2)

    for screen in range(max_screens):
        print(f"[Screen {screen+1}]", end=" ")

        # 懒加载等待
        wait_stable()
        screenshot()
        img = cv2.imread("captures/now.png")
        if img is None:
            break

        # 画面变化检测
        if screen > 0:
            prev = cv2.imread(f"captures/screen_{screen-1:02d}.png")
            if prev is not None:
                diff = cv2.absdiff(img, prev).mean()
                if diff < 1.0:
                    print("  end (no change)")
                    break

        cv2.imwrite(f"captures/screen_{screen:02d}.png", img)

        # minicpm-v数商品（用于验证锚点数）
        expected = ask_model_count("captures/now.png")

        # 增强锚点：AnchorSlicer已在内部处理
        slicer = AnchorSlicer()
        slicer.extract_anchors(img, reader)
        cols, col_anchors = slicer.detect_layout()
        cards = slicer.slice(img, col_anchors)

        print(f"{len(cards)} cards", end="")
        if expected > 0:
            print(f" (VLM sees {expected})", end="")
        print()

        # 保存卡片
        sd = f"captures/cards/s{screen}"
        os.makedirs(sd, exist_ok=True)
        for i, card in enumerate(cards):
            cv2.imwrite(f"{sd}/c{i}.png", card)

        # 提取标题
        new_titles = []
        for card in cards:
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
                if title and len(title) >= 4 and "无法" not in title and "提供" not in title:
                    new_titles.append(title)
                    print(f"    [{len(all_titles)+len(new_titles)}] {title}")
            except Exception:
                pass

        all_titles.extend(new_titles)

        # 70%滑动
        scroll_70pct()
        time.sleep(2)

    # 去重
    unique = dedup_titles(all_titles)

    print(f"\n=== {store_name}: {len(all_titles)} raw -> {len(unique)} unique ===")
    for i, t in enumerate(unique):
        print(f"  {i+1}. {t}")

    with open(f"data/{store_name}_titles.txt", "w", encoding="utf-8") as f:
        for t in unique:
            f.write(t + "\n")

    return unique


if __name__ == "__main__":
    run_scan("朱公子上岸教育")
