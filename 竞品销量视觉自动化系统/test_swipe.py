"""测试不同滑动参数"""
import subprocess, time, cv2, numpy as np

ADB = r"C:\Program Files\scrcpy\scrcpy-win64-v3.1\adb.exe"
DEV = "182f4e2f"

tests = [
    ("全屏", 2200, 400, 800),
    ("半屏", 2200, 1200, 500),
    ("小步", 2000, 1500, 400),
]

for name, y1, y2, dur in tests:
    subprocess.run([ADB, "-s", DEV, "shell", "screencap", "-p", "/sdcard/b.png"])
    subprocess.run([ADB, "-s", DEV, "pull", "/sdcard/b.png", "captures/b.png"])
    subprocess.run([ADB, "-s", DEV, "shell", "input", "swipe", "610", str(y1), "610", str(y2), str(dur)])
    time.sleep(3)
    subprocess.run([ADB, "-s", DEV, "shell", "screencap", "-p", "/sdcard/a.png"])
    subprocess.run([ADB, "-s", DEV, "pull", "/sdcard/a.png", "captures/a.png"])
    b = cv2.imread("captures/b.png")
    a = cv2.imread("captures/a.png")
    diff = np.abs(b.astype(float) - a.astype(float)).mean() / 255 * 100
    print(f"{name} ({y1}->{y2}): {diff:.1f}% change")
