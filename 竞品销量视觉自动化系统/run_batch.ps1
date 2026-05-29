$adb = "C:\Program Files\scrcpy\scrcpy-win64-v3.1\adb.exe"
$d = "182f4e2f"
$total = 30

Write-Host "========================================"
Write-Host " 竞品店铺批量采集 - $total 家"
Write-Host "========================================"

# Clean old files
Remove-Item captures/batch_*.png -ErrorAction SilentlyContinue
$results = @()

for ($i = 1; $i -le $total; $i++) {
    # Get store name from Python
    $name = python -c "import sys; sys.path.insert(0,'.'); from src.core.target_loader import TargetLoader; t=list(TargetLoader('config/targets.yaml').shuffled()); print(t[$i-1].store_name)"

    Write-Host ""
    Write-Host ("[{0}/{1}] {2}" -f $i, $total, $name)

    # Switch IME
    & $adb -s $d shell "ime set com.android.adbkeyboard/.AdbIME" 2>$null
    Start-Sleep -Seconds 1

    # Open Taobao
    & $adb -s $d shell "monkey -p com.taobao.taobao 1" 2>$null
    Start-Sleep -Seconds 4

    # Tap search
    & $adb -s $d shell "input tap 610 260" 2>$null
    Start-Sleep -Seconds 2

    # Chinese input
    & $adb -s $d shell "am broadcast -a ADB_INPUT_TEXT --es msg '$name'" 2>$null
    Start-Sleep -Seconds 2

    # Search
    & $adb -s $d shell "input keyevent KEYCODE_ENTER" 2>$null
    Start-Sleep -Seconds 4

    # Screenshot
    $fn = "captures/batch_{0:D2}.png" -f $i
    & $adb -s $d shell "screencap -p /sdcard/screen.png" 2>$null
    & $adb -s $d pull /sdcard/screen.png $fn 2>$null

    # Restore IME
    & $adb -s $d shell "ime set com.sohu.inputmethod.sogou.xiaomi/.SogouIME" 2>$null

    # Quick OCR check
    $sales = python -c "
import cv2, sys; sys.path.insert(0,'.')
from src.ocr.sales_extractor import SalesExtractor
import easyocr
r=easyocr.Reader(['ch_sim','en'],gpu=False)
img=cv2.imread('$fn')
if img is None: sys.exit()
ext=SalesExtractor(); h=img.shape[0]
hits=[]
for t in r.readtext(img[h//3:2*h//3,:]):
    if t[2]>0.4 and len(t[1])>2:
        v=ext.extract(t[1])
        if v and v>0: hits.append(f'{t[1]}|{v}')
print(';'.join(hits[:5]))
" 2>$null

    $results += "[{0}/{1}] {2}: {3}" -f $i, $total, $name, $sales
    Write-Host "  $sales"
}

# Save results
$results | Out-File -FilePath "captures/batch_results.txt" -Encoding UTF8
Write-Host ""
Write-Host "========================================"
Write-Host " 采集完成! 结果: captures/batch_results.txt"
Write-Host " 截图: captures/batch_*.png"
Write-Host "========================================"
