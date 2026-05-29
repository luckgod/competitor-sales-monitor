"""检查手机 UI 状态"""
import subprocess, re, sys

adb = r"C:\Program Files\scrcpy\scrcpy-win64-v3.1\adb.exe"
d = "182f4e2f"

# Dump UI
subprocess.run([adb, "-s", d, "shell", "uiautomator", "dump", "/sdcard/ui.xml"], capture_output=True)
subprocess.run([adb, "-s", d, "pull", "/sdcard/ui.xml", "captures/ui.xml"], capture_output=True)

with open("captures/ui.xml", "r", encoding="utf-8") as f:
    xml = f.read()

# Find all clickable nodes with text
texts = re.findall(r'text="([^"]+)"', xml)
classes = re.findall(r'class="([^"]+)"', xml)
bounds = re.findall(r'bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', xml)
clickable = re.findall(r'clickable="([^"]+)"', xml)

print(f"Total nodes: {len(texts)}")
print(f"Non-empty text nodes:")
for i, t in enumerate(texts):
    if t and len(t) > 1:
        b = bounds[i] if i < len(bounds) else ("?","?","?","?")
        c = clickable[i] if i < len(clickable) else "?"
        print(f"  [{t}] bounds=({b[0]},{b[1]})-({b[2]},{b[3]}) clickable={c}")
