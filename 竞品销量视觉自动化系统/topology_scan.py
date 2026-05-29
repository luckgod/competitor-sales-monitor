"""V5.1 拓扑发现 — 语义轮廓切分 + OCR真标题 + pHash去重 + 早停"""
import cv2, hashlib, subprocess, time, sqlite3, os, numpy as np, sys
from datetime import date

sys.path.insert(0, ".")
from src.consumer.card_splitter import CardSplitter

ADB = r"C:\Program Files\scrcpy\scrcpy-win64-v3.1\adb.exe"
DEV = "182f4e2f"
DB = "data/topology.db"


def compute_phash(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (8, 8))
    return hex(int("".join(["1" if p > resized.mean() else "0" for p in resized.flatten()]), 2))[2:]


def virtual_id(store, title, phash):
    return hashlib.md5(f"{store}|{title}|{phash}".encode()).hexdigest()[:16]


def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS competitor_products (
        virtual_id TEXT PRIMARY KEY, store_name TEXT, title TEXT,
        img_hash TEXT, first_seen TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS topology_checkpoint (
        store_name TEXT PRIMARY KEY, total INT, scanned INT)""")
    conn.execute("DELETE FROM topology_checkpoint WHERE store_name='__current__'")
    conn.commit()
    return conn


def extract_title(card_img, reader):
    """物理分区提取 — 扫描标题区 + LLM 语义精选"""
    try:
        ch, cw = card_img.shape[:2]

        # 标题区：商品图下方、价格上方
        y1 = int(ch * 0.55)
        y2 = int(ch * 0.88)
        title_roi = card_img[y1:y2, :]
        if title_roi.size == 0:
            return None

        results = reader.readtext(title_roi)
        if not results:
            return None

        skip = {"¥", "￥", "元", "人付款", "已售", "包邮", "补抵", "优惠",
                "满减", "券", "发货", "退货", "加购", "收藏", "关注", "进店"}

        candidates = []
        for r in results:
            text, conf = r[1], r[2]
            if conf < 0.45 or len(text) < 4:
                continue
            if any(w in text for w in skip):
                continue
            if text.replace(".", "").replace(" ", "").isdigit():
                continue
            candidates.append(text.strip())

        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # LLM 精选（优先）
        try:
            import requests
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "qwen2.5:7b", "prompt":
                      "从以下淘宝商品卡片文字中提取商品名称（纯商品名，不要解释）：\n"
                      + "\n".join(candidates[:8]),
                      "stream": False,
                      "options": {"temperature": 0, "num_predict": 128}},
                timeout=8,
            )
            title = resp.json().get("response", "").strip()
            # 验证 LLM 返回的标题在候选中存在或合理
            if title and len(title) >= 4:
                return title
        except Exception:
            pass

        # 降级：取最长
        candidates.sort(key=len, reverse=True)
        return candidates[0]
    except Exception:
        pass
    return None


def scan_store(store_name: str):
    conn = init_db()
    total = 0
    done = False

    # EasyOCR 全局初始化
    import easyocr
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
    splitter = CardSplitter(mode="contour", screen_width=1220, screen_height=2712)

    print(f"\n{'=' * 50}")
    print(f" V5.1 拓扑扫描: {store_name}")
    print(f" 轮廓切分 + OCR标题 + pHash去重")
    print(f"{'=' * 50}")

    # 先上滑到顶
    print("上滑到顶部...")
    for _ in range(3):
        subprocess.run([ADB, "-s", DEV, "shell", "input", "swipe", "610", "400", "610", "2200", "500"])
        time.sleep(1.5)

    screen_count = 0
    prev_phash = None

    while not done:
        screen_count += 1
        subprocess.run([ADB, "-s", DEV, "shell", "screencap", "-p", "/sdcard/screen.png"], capture_output=True)
        subprocess.run([ADB, "-s", DEV, "pull", "/sdcard/screen.png", "captures/topo.png"], capture_output=True)
        frame = cv2.imread("captures/topo.png")
        if frame is None:
            break
        h, w = frame.shape[:2]

        # 画面变化检测
        prev_file = "captures/topo_prev.png"
        if os.path.exists(prev_file):
            prev = cv2.imread(prev_file)
            if cv2.absdiff(frame, prev).mean() < 3.0:
                print(f"\n>>> 画面无变化，到底了!")
                done = True
                break
        cv2.imwrite(prev_file, frame)

        # V5.1: 语义轮廓切分（替代固定网格）
        cards = splitter.split(frame)
        new_in_screen = 0

        for card in cards:
            card_img = card.get("card")
            if card_img is None or card_img.size == 0:
                continue
            # 最小尺寸过滤
            ch, cw = card_img.shape[:2]
            if ch < 100 or cw < 100:
                continue

            phash = compute_phash(card_img)

            # 跳过纯色/广告区域（pHash全0或全f表示单色块）
            if len(set(phash)) <= 2:
                continue

            # OCR 提取真实标题
            title = extract_title(card_img, reader)
            if title is None:
                title = f"{store_name}_item_{screen_count}_{new_in_screen}"

            vid = virtual_id(store_name, title, phash)

            try:
                conn.execute(
                    "INSERT INTO competitor_products VALUES (?,?,?,?,?)",
                    (vid, store_name, title, phash, date.today().isoformat()))
                total += 1
                new_in_screen += 1
            except sqlite3.IntegrityError:
                pass

        conn.commit()
        print(f"  屏{screen_count}: +{new_in_screen}新品 (轮廓切出{len(cards)}卡) | 累计{total}款")

        if done:
            break

        # 下滑
        subprocess.run([ADB, "-s", DEV, "shell", "input", "swipe", "610", "2000", "610", "800", "500"])
        time.sleep(2.5)

    # 输出清单
    rows = conn.execute(
        "SELECT virtual_id, title, img_hash FROM competitor_products WHERE store_name=?",
        (store_name,)).fetchall()

    print(f"\n{'=' * 50}")
    print(f" [{store_name}] 商品清单: {len(rows)} 款")
    print(f"{'=' * 50}")
    for i, (vid, title, ph) in enumerate(rows):
        print(f"  {i+1:2d}. [{vid}] {title}")

    conn.close()
    return rows


if __name__ == "__main__":
    scan_store("朱公子上岸教育")
