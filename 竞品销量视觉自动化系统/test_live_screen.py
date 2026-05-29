"""真机截图 → 视觉管线端到端验证"""
import cv2
from src.consumer.card_splitter import CardSplitter
from src.consumer.filter import LazyLoadFilter
from src.ocr.sales_extractor import SalesExtractor

img = cv2.imread("captures/phone_screen.png")
h, w = img.shape[:2]
print(f"截图尺寸: {w}x{h}")

# 卡片切分
splitter = CardSplitter(mode="grid", screen_width=w, screen_height=h)
cards = splitter.split(img)
print(f"切分出 {len(cards)} 张卡片")

# 过滤 + 销量提取
f = LazyLoadFilter()
ext = SalesExtractor()

valid = 0
for i, card in enumerate(cards[:8]):
    image_roi = card.get("image")
    sales_roi = card.get("sales")

    if image_roi is None:
        continue

    is_invalid = f.is_invalid(card)
    if is_invalid:
        print(f"  卡片{i}: [白块] 跳过")
        continue

    valid += 1

    # 保存子图供检查
    if sales_roi is not None and sales_roi.size > 0:
        cv2.imwrite(f"captures/card{i}_sales.png", sales_roi)
    if image_roi is not None and image_roi.size > 0:
        cv2.imwrite(f"captures/card{i}_image.png", image_roi)

    print(f"  卡片{i}: [有效] 主图={image_roi.shape}, 销量区={sales_roi.shape if sales_roi is not None else 'N/A'}")

print(f"\n有效卡片: {valid}/{min(8, len(cards))}")
print(f"子图已保存到 captures/ 目录")
print()
print("[销量提取器自检]")
for text, expected in [("月销 1.5万+", 15000), ("4200+人付款", 4200), ("已售 100件", 100)]:
    result = ext.extract(text)
    print(f"  '{text}' -> {result} {'OK' if result == expected else 'FAIL'}")
