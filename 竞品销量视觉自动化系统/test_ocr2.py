"""真机截图 — 轮廓检测切分 + EasyOCR 验证"""
import cv2
import easyocr

reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

img = cv2.imread("captures/phone_screen.png")
h, w = img.shape[:2]
print(f"Screen: {w}x{h}")

# 轮廓检测切分
from src.consumer.card_splitter import CardSplitter
splitter = CardSplitter(mode="contour", screen_width=w, screen_height=h)
cards = splitter.split(img)
print(f"Contour mode: {len(cards)} cards")

from src.consumer.filter import LazyLoadFilter
f = LazyLoadFilter()
from src.ocr.sales_extractor import SalesExtractor
ext = SalesExtractor()

valid = 0
for i, card in enumerate(cards[:10]):
    sales_roi = card.get("sales")
    if sales_roi is None or sales_roi.size == 0:
        continue
    if f.is_invalid(card):
        continue
    valid += 1

    # OCR
    results = reader.readtext(sales_roi)
    texts = [r[1] for r in results if r[2] > 0.3]

    # 销量提取
    sales_values = []
    for t in texts:
        val = ext.extract(t)
        if val is not None:
            sales_values.append(val)

    print(f"  Card {i}: texts={texts[:5]}{'...' if len(texts)>5 else ''}")
    if sales_values:
        print(f"         sales={sales_values}")

print(f"\nValid cards: {valid}")

# 也跑全图OCR看看当前屏幕是什么
print("\n[Full screen OCR sample - top area]")
top = img[:h//4, :]
results = reader.readtext(top)
top_texts = [r[1] for r in results if r[2] > 0.4]
print(f"  Top area: {top_texts[:10]}")
