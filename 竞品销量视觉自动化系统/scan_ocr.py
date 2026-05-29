"""对 captures/*.png 批量 OCR 提取销量"""
import sys, cv2, easyocr, glob
sys.path.insert(0, ".")
from src.ocr.sales_extractor import SalesExtractor

reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
ext = SalesExtractor()

for path in sorted(glob.glob("captures/store_*.png")):
    img = cv2.imread(path)
    if img is None:
        continue
    h = img.shape[0]
    mid = img[h // 3: 2 * h // 3, :]
    hits = []
    for r in reader.readtext(mid):
        text, conf = r[1], r[2]
        if conf < 0.4 or len(text) < 3:
            continue
        v = ext.extract(text)
        if v and v > 0:
            hits.append((text.strip(), v))
    print(f"{path}: {hits[:5] if hits else 'no sales data'}")
