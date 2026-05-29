#!/usr/bin/env python3
"""真机随动采集 — 全自动化搜索竞品店铺 + OCR 销量提取"""
import subprocess, time, cv2, sys, easyocr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.core.target_loader import TargetLoader
from src.ocr.sales_extractor import SalesExtractor

ADB = r"C:\Program Files\scrcpy\scrcpy-win64-v3.1\adb.exe"
DEVICE = "182f4e2f"


def adb(cmd: list[str]) -> str:
    r = subprocess.run([ADB, "-s", DEVICE] + cmd, capture_output=True, text=True, timeout=20)
    return (r.stdout or "") + (r.stderr or "")



def switch_ime(to_adb: bool):
    ime = "com.android.adbkeyboard/.AdbIME" if to_adb else "com.sohu.inputmethod.sogou.xiaomi/.SogouIME"
    adb(["shell", "ime", "set", ime])
    time.sleep(0.8)
def adb_pull(local_path: str) -> bool:
    """从设备拉取文件（二进制安全）"""
    r = subprocess.run([ADB, "-s", DEVICE, "pull", "/sdcard/screen.png", local_path],
                       capture_output=True, timeout=15)
    return r.returncode == 0



def search_store(name: str) -> bytes | None:
    """搜索店铺并返回截图"""
    switch_ime(True)

    # 先回桌面再打开淘宝
    adb(["shell", "input", "keyevent", "KEYCODE_HOME"])
    time.sleep(1.5)
    adb(["shell", "monkey", "-p", "com.taobao.taobao", "1"])
    time.sleep(4)

    # 点搜索框
    adb(["shell", "input", "tap", "610", "260"])
    time.sleep(2)

    # 输入中文
    adb(["shell", "am", "broadcast", "-a", "ADB_INPUT_TEXT", "--es", "msg", name])
    time.sleep(1.5)

    # 搜索
    adb(["shell", "input", "keyevent", "KEYCODE_ENTER"])
    time.sleep(4)

    # 截图
    adb(["shell", "screencap", "-p", "/sdcard/screen.png"])
    adb_pull(r"captures/live_screen.png")

    switch_ime(False)

    img = cv2.imread("captures/live_screen.png")
    if img is None:
        return None
    _, buf = cv2.imencode(".png", img)
    return buf


def extract_sales(img, reader, ext) -> list[tuple[str, int]]:
    """从截图中提取销量数据"""
    h = img.shape[0]
    mid = img[h // 3: 2 * h // 3, :]
    results = reader.readtext(mid)
    hits = []
    for r in results:
        text, conf = r[1], r[2]
        if conf < 0.4 or len(text) < 3:
            continue
        val = ext.extract(text)
        if val and val > 0:
            hits.append((text.strip(), val))
    return hits


def main():
    print("=" * 55)
    print("  真机随动采集 — 竞品店铺轮询")
    print("=" * 55)

    reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
    ext = SalesExtractor()
    loader = TargetLoader("config/targets.yaml")
    targets = loader.shuffled()

    total = len(targets)
    all_hits = []

    for i, target in enumerate(targets[:3]):  # 先跑3家测试
        name = target.store_name
        print(f"\n[{i+1}/3] {name}")

        buf = search_store(name)
        if buf is None:
            print("  [FAIL] 截图失败")
            continue

        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        hits = extract_sales(img, reader, ext)

        # 检查是否搜到店铺
        h = img.shape[0]
        top = img[:h // 4, :]
        top_texts = [r[1] for r in reader.readtext(top) if r[2] > 0.4]
        found_store = any(name[:4] in t for t in top_texts)

        if found_store or hits:
            print(f"  [OK] 找到店铺, {len(hits)} 条销量:")
            for t, v in hits[:5]:
                print(f"       '{t}' -> {v}")
                all_hits.append({"store": name, "text": t, "sales": v})
        else:
            print(f"  [WARN] 未找到店铺，可能是新店铺或搜索无结果")

    print(f"\n{'='*55}")
    print(f"完成! {total} 家店铺, 共提取 {len(all_hits)} 条销量数据")
    for h in all_hits:
        print(f"  {h['store']}: {h['text']} → {h['sales']}")


if __name__ == "__main__":
    import numpy as np
    main()
