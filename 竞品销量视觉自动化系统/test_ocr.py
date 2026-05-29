"""真机截图 EasyOCR 文字识别验证"""
import cv2
import easyocr

print("Loading EasyOCR Chinese model...")
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
print("Model loaded.\n")

for i in range(6):
    path = f"captures/card{i}_sales.png"
    img = cv2.imread(path)
    if img is None:
        print(f"  Card {i}: no image")
        continue
    results = reader.readtext(img)
    texts = [r[1] for r in results]
    if texts:
        print(f"  Card {i} sales: {texts}")
    else:
        print(f"  Card {i} sales: (no text)")
