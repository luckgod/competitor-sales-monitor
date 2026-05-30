"""Patch-and-Predict — AnchorSlicer切分 + VLM提取"""

import cv2, os, base64, requests, sys
sys.path.insert(0, ".")
from src.core.anchor_slicer import AnchorSlicer

MODEL = "minicpm-v:latest"
PROMPT = "直接读出这张商品卡片上显示的商品标题。忽略价格和促销标签。只返回标题文字。"


def run_pipeline(screenshot_path: str = "captures/now.png") -> list[str]:
    print("[Phase 1] 锚点中线切割")
    img = cv2.imread(screenshot_path)
    if img is None:
        print("ERR: no screenshot")
        return []

    slicer = AnchorSlicer()
    import easyocr
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
    slicer.extract_anchors(img, reader)
    cols, col_anchors = slicer.detect_layout()
    cards = slicer.slice(img, col_anchors)
    info = slicer.layout_info
    print(f"  {info['anchors']} anchors -> {info['mode']} -> {len(cards)} cards")

    os.makedirs("captures/cards", exist_ok=True)
    for i, card in enumerate(cards):
        cv2.imwrite(f"captures/cards/card_{i:02d}.png", card)

    print(f"\n[Phase 2] VLM extraction ({len(cards)} cards)")
    titles = []
    for i, card in enumerate(cards):
        _, buf = cv2.imencode(".png", card)
        b64 = base64.b64encode(buf).decode()
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": MODEL, "prompt": PROMPT, "images": [b64],
                      "stream": False, "options": {"temperature": 0, "num_predict": 64}},
                timeout=120,
            )
            title = resp.json()["response"].strip()
            skip = ["补差", "差价"]
            if title and len(title) >= 4 and not any(w in title for w in skip):
                titles.append(title)
                print(f"  [{i}] {title}")
            else:
                print(f"  [{i}] SKIP")
        except Exception:
            print(f"  [{i}] ERR")

    print(f"\n[Done] {len(titles)} titles")
    return titles


if __name__ == "__main__":
    run_pipeline()
