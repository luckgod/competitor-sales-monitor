import sys; sys.path.insert(0,".")
from src.core.anchor_slicer import AnchorSlicer
import cv2,easyocr
img=cv2.imread("captures/now.png")
s=AnchorSlicer()
r=easyocr.Reader(['ch_sim','en'],gpu=False)
s.extract_anchors(img,r)
xs=[a["x"] for a in s._anchors]
print(f"X spread: {max(xs)-min(xs)}")
print(f"Anchors: {len(s._anchors)}")
for a in s._anchors:
    print(f"  x={a['x']:4d} y={a['y']:4d} | {a['text']}")
cols,_=s.detect_layout()
print(f"Cols: {cols}")
