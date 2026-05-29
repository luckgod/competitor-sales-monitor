$adb = "C:\Program Files\scrcpy\scrcpy-win64-v3.1\adb.exe"
$device = "182f4e2f"

# Get store names from Python (avoids PowerShell UTF-8 issues)
$names = python -c "import sys; sys.path.insert(0,'.'); from src.core.target_loader import TargetLoader; loader=TargetLoader('config/targets.yaml'); targets=loader.shuffled(); [print(t.store_name) for t in targets[:3]]"
$storeNames = $names -split "`r`n" | Where-Object { $_ -ne "" }

Write-Host "========================================"
Write-Host (" Live Scan - " + $storeNames.Count + " stores")
Write-Host "========================================"

# Clean old screenshots
Remove-Item captures/store_*.png -ErrorAction SilentlyContinue

foreach ($name in $storeNames) {
    Write-Host ("[Search] " + $name)

    # Switch to ADBKeyboard
    & $adb -s $device shell "ime set com.android.adbkeyboard/.AdbIME" 2>$null
    Start-Sleep -Seconds 1

    # Open Taobao
    & $adb -s $device shell "monkey -p com.taobao.taobao 1" 2>$null
    Start-Sleep -Seconds 4

    # Tap search bar
    & $adb -s $device shell "input tap 610 260" 2>$null
    Start-Sleep -Seconds 2

    # Input Chinese text via ADBKeyboard broadcast
    $cmd = "am broadcast -a ADB_INPUT_TEXT --es msg '$name'"
    & $adb -s $device shell $cmd 2>$null
    Start-Sleep -Seconds 2

    # Hit search
    & $adb -s $device shell "input keyevent KEYCODE_ENTER" 2>$null
    Start-Sleep -Seconds 4

    # Screenshot
    $ts = Get-Date -Format "HHmmss"
    $file = "captures/store_${ts}.png"
    & $adb -s $device shell "screencap -p /sdcard/screen.png" 2>$null
    & $adb -s $device pull /sdcard/screen.png $file 2>$null
    Write-Host ("  Screenshot: " + $file)

    # Restore IME
    & $adb -s $device shell "ime set com.sohu.inputmethod.sogou.xiaomi/.SogouIME" 2>$null
}

Write-Host ""
Write-Host "Screenshots done. Running OCR..."
python scan_ocr.py
