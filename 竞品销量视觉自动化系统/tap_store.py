"""极简版：OCR→找目标→点击"""
import cv2, easyocr, subprocess, time

ADB = r"C:\Program Files\scrcpy\scrcpy-win64-v3.1\adb.exe"
DEV = "182f4e2f"

# 1. 截图
subprocess.run([ADB, "-s", DEV, "shell", "screencap", "-p", "/sdcard/screen.png"])
subprocess.run([ADB, "-s", DEV, "pull", "/sdcard/screen.png", "captures/now.png"])

r = easyocr.Reader(['ch_sim', 'en'], gpu=False)
img = cv2.imread("captures/now.png")
h = img.shape[0]

# 2. 扫全屏，列出所有文字+位置
print(f"屏幕 {img.shape[1]}x{h}")
print("=" * 50)
for t in r.readtext(img):
    text, conf, bbox = t[1], t[2], t[0]
    if conf < 0.3 or len(text) < 2:
        continue
    cx = int((bbox[0][0] + bbox[2][0]) / 2)
    cy = int((bbox[0][1] + bbox[2][1]) / 2)
    y_zone = "上" if cy < h/3 else ("中" if cy < 2*h/3 else "下")
    print(f"[{y_zone}] ({cx:4d},{cy:4d}) {text}")

# 3. 用户告诉我点什么
print("\n" + "=" * 50)
print("找到想点的目标了吗？告诉我要点哪个坐标或文字")
