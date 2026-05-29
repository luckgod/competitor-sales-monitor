"""调试：截图 → 轮廓切分 → 保存每张卡片"""
import cv2, sys, subprocess, os
sys.path.insert(0, ".")
from src.consumer.card_splitter import CardSplitter

ADB = r"C:\Program Files\scrcpy\scrcpy-win64-v3.1\adb.exe"
DEV = "182f4e2f"

# 截图
subprocess.run([ADB, "-s", DEV, "shell", "screencap", "-p", "/sdcard/screen.png"])
subprocess.run([ADB, "-s", DEV, "pull", "/sdcard/screen.png", "captures/debug.png"])

img = cv2.imread("captures/debug.png")
h, w = img.shape[:2]
print(f"屏幕: {w}x{h}")

# 网格切分: 2列, 用 OCR 找第一个商品卡锚点
import easyocr
r = easyocr.Reader(['ch_sim','en'],gpu=False)

# 找"面试100题"、"央企国企"这类标题文字的位置 → 这就是商品起始行
y_start = int(h * 0.33)
for t in r.readtext(img):
    text, conf, bbox = t[1], t[2], t[0]
    if conf > 0.5 and ('面试' in text or '央企' in text or '2026' in text or '2027' in text):
        y = int((bbox[0][1] + bbox[2][1]) / 2)
        if y > h * 0.2:  # 跳过搜索栏
            y_start = y - 30  # 标题上方留边距
            print(f"锚点: '{text}' at y={y}, start={y_start}")
            break

card_h = int(h * 0.18)    # 每张卡高度
card_w = w // 2
cards = []
for row in range(6):
    for col in range(2):
        x1 = col * card_w
        y1 = y_start + row * card_h
        if y1 > h: break
        x2 = x1 + card_w
        y2 = min(h, y1 + int(card_h * 1.3))
        card_img = img[y1:y2, x1:x2]
        if card_img.size > 0:
            cards.append({"card": card_img, "x": x1, "y": y1, "w": card_w, "h": y2-y1})
print(f"卡片数: {len(cards)}")

# 保存
os.makedirs("captures/cards", exist_ok=True)
for i, card in enumerate(cards):
    card_img = card.get("card")
    if card_img is None:
        continue
    cv2.imwrite(f"captures/cards/card_{i}.png", card_img)
    ch, cw = card_img.shape[:2]
    print(f"  卡{i}: {cw}x{ch} -> captures/cards/card_{i}.png")

print(f"\n请在 captures/cards/ 查看切出的卡片是否正确")
