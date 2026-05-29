"""OCR定位店铺卡片坐标"""
import cv2, easyocr
r = easyocr.Reader(['ch_sim','en'], gpu=False)
img = cv2.imread('captures/search_pos.png')
h = img.shape[0]

# Scan middle area for store cards
for t in r.readtext(img[h//3:, :]):
    text, conf, bbox = t[1], t[2], t[0]
    if conf > 0.3 and len(text) > 2:
        cx = int((bbox[0][0] + bbox[2][0]) / 2)
        cy = int((bbox[0][1] + bbox[2][1]) / 2) + h // 3
        if '店铺' in text or '旗舰' in text:
            print(f"STORE: '{text}' at ({cx}, {cy})")
            # Don't break - show all store cards
        else:
            print(f"  text: '{text}' at y={cy}")
