# CI: launch the real exe in each mode, screenshot the desktop, make sure it
# didn't crash. Runs in a copy of the exe so the build output stays clean.
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms, System.Drawing

$work = Join-Path $PWD "smoke"
$shots = Join-Path $PWD "screenshots"
New-Item -ItemType Directory -Force $work, $shots | Out-Null
Copy-Item dist\TFTCoach.exe $work
$exe = Join-Path $work "TFTCoach.exe"
python ci\make_sample_video.py (Join-Path $work "sample_game.mp4") 30

function Shot($name) {
    $b = [System.Windows.Forms.SystemInformation]::VirtualScreen
    $bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($b.Left, $b.Top, 0, 0, $bmp.Size)
    $bmp.Save((Join-Path $shots "$name.png"))
    $g.Dispose(); $bmp.Dispose()
}

$failed = $false
function Check($name, $argList, $seconds) {
    Write-Host "=== $name : TFTCoach.exe $argList"
    if ($argList) { $p = Start-Process $exe -ArgumentList $argList -WorkingDirectory $work -PassThru }
    else { $p = Start-Process $exe -WorkingDirectory $work -PassThru }
    Start-Sleep -Seconds $seconds
    Shot $name
    if ($p.HasExited) {
        Write-Host "FAIL: $name exited early with code $($p.ExitCode)"
        $script:failed = $true
    } else {
        Write-Host "OK: $name is running"
        Stop-Process -Id $p.Id -Force
        Start-Sleep -Seconds 2
    }
}

# 1. First run: no key yet -> "Connect your Google AI key" screen
Check "1_key_screen" $null 8

# 2. With a (fake) key saved -> main launcher screen
Set-Content -Path (Join-Path $work ".env") -Value "GOOGLE_API_KEY=ci-fake-key"
Check "2_main_screen" $null 8

# 3. File mode plays the video (fake key -> shows Reconnecting, must not crash)
Check "3_file_mode" @("--file", "`"$work\sample_game.mp4`"") 10

# 4. Screen mode window picker
Check "4_screen_picker" @("--screen") 8

Write-Host "---- tft_coach_output.log ----"
if (Test-Path "$work\tft_coach_output.log") { Get-Content "$work\tft_coach_output.log" -Tail 60 }
if (Test-Path "$work\tft_coach_error.log") {
    Write-Host "---- tft_coach_error.log ----"
    Get-Content "$work\tft_coach_error.log"
    $failed = $true
}
Get-ChildItem $work | Select-Object Name, Length
if ($failed) { exit 1 }
