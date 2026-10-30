# semi-signals launcher for Windows: double-click start.cmd, or run .\start.ps1 in PowerShell.
# Uses the .venv in this folder; offers to create it (same steps as docs/SEIZURE_DETECTION.md) if missing.
# Arguments are passed on to the launcher, e.g.  .\start.ps1 --viewer-args "--width 1280"
# Research use only, not diagnostic.

# Native tools (pip) write warnings to stderr; failures are checked with $LASTEXITCODE instead.
$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath $PSScriptRoot
$Py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

function Show-SetupHelp {
    Write-Host ''
    Write-Host 'Manual setup (from this folder):' -ForegroundColor Cyan
    Write-Host '  python -m venv .venv'
    Write-Host '  .venv\Scripts\python.exe -m pip install -r requirements-demo.txt'
    Write-Host '  .venv\Scripts\python.exe -m pip uninstall -y opencv-contrib-python opencv-python-headless'
    Write-Host '  .venv\Scripts\python.exe -m pip install --force-reinstall --no-deps opencv-python'
    Write-Host 'Behind a proxy, set it first:  $env:HTTPS_PROXY = "http://proxy:port"'
    Write-Host ''
}

function Find-BasePython {
    # 1) SEMI_PYTHON, 2) pyenv-win (active version, else the newest installed 3.10-3.12), 3) the py launcher,
    # 4) python on PATH (not the Microsoft Store stub)
    if ($env:SEMI_PYTHON -and (Test-Path -LiteralPath $env:SEMI_PYTHON)) { return $env:SEMI_PYTHON }
    if (Get-Command pyenv -ErrorAction SilentlyContinue) {
        $p = (& pyenv which python 2>$null | Select-Object -First 1)
        if ($LASTEXITCODE -eq 0 -and $p -and (Test-Path -LiteralPath $p.Trim())) { return $p.Trim() }
    }
    # pyenv-win installs without a usable global/local version: newest installed 3.10-3.12
    $roots = @($env:PYENV, $env:PYENV_ROOT, (Join-Path $env:USERPROFILE '.pyenv\pyenv-win')) | Where-Object { $_ }
    foreach ($r in $roots) {
        $vd = Join-Path $r 'versions'
        if (-not (Test-Path -LiteralPath $vd)) { continue }
        $v = Get-ChildItem -LiteralPath $vd -Directory | Where-Object { $_.Name -match '^3\.(1[0-2])\.\d+$' } |
            Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
        if ($v) {
            $exe = Join-Path $v.FullName 'python.exe'
            if (Test-Path -LiteralPath $exe) { return $exe }
        }
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $p = (& py -3 -c 'import sys; print(sys.executable)' 2>$null | Select-Object -First 1)
        if ($p -and (Test-Path -LiteralPath $p.Trim())) { return $p.Trim() }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notlike '*WindowsApps*') {
        $p = (& $cmd.Source -c 'import sys; print(sys.executable)' 2>$null | Select-Object -First 1)
        if ($LASTEXITCODE -eq 0 -and $p -and (Test-Path -LiteralPath $p.Trim())) { return $p.Trim() }
    }
    return $null
}

function Invoke-Step([string]$What, [scriptblock]$Block) {
    Write-Host "==> $What" -ForegroundColor Cyan
    & $Block
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

function Exit-WithPause([int]$Code) {
    if ($Code -ne 0) { Read-Host 'Press Enter to close' | Out-Null }
    exit $Code
}

if (-not (Test-Path -LiteralPath $Py)) {
    Write-Host 'No .venv found in this folder.' -ForegroundColor Yellow
    $base = Find-BasePython
    if (-not $base) {
        Write-Host 'Could not find Python 3 (pyenv-win, py launcher or python on PATH).' -ForegroundColor Red
        Write-Host 'Point SEMI_PYTHON at a python.exe, e.g.  $env:SEMI_PYTHON = "C:\Python312\python.exe"' -ForegroundColor Yellow
        Show-SetupHelp
        Exit-WithPause 1
    }
    Write-Host "Python found: $base"
    if ($env:HTTPS_PROXY) { Write-Host "Using HTTPS_PROXY=$env:HTTPS_PROXY" }
    $answer = Read-Host 'Create .venv now and install the requirements? [Y/n]'
    if ($answer -and $answer.Trim().ToLower() -notin @('y', 'yes')) {
        Show-SetupHelp
        Exit-WithPause 1
    }
    try {
        Invoke-Step 'create .venv' { & $base -m venv .venv }
        Invoke-Step 'install requirements-demo.txt' { & $Py -m pip install -r requirements-demo.txt }
        # rtmlib pulls in opencv-contrib-python; two OpenCV wheels in one venv can break cv2
        Invoke-Step 'remove extra OpenCV wheels' { & $Py -m pip uninstall -y opencv-contrib-python opencv-python-headless }
        Invoke-Step 'reinstall opencv-python' { & $Py -m pip install --force-reinstall --no-deps opencv-python }
    } catch {
        Write-Host $_.Exception.Message -ForegroundColor Red
        Write-Host 'Fix the error above (often the proxy), then run start.cmd again. The .venv is kept.' -ForegroundColor Yellow
        Show-SetupHelp
        Exit-WithPause 1
    }
    Write-Host 'Setup done. Next: click "Fetch models & sample data" in the launcher.' -ForegroundColor Green
}

& $Py -c 'import tkinter' 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'This Python has no tkinter. Reinstall Python with "tcl/tk and IDLE" ticked, then recreate .venv.' -ForegroundColor Red
    Exit-WithPause 1
}

# Pillow draws the on-screen text (added after some .venv folders were made); without it the text is plainer
& $Py -c 'import PIL' 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host '==> install Pillow (sharper on-screen text)' -ForegroundColor Cyan
    & $Py -m pip install 'pillow>=9.2'
    if ($LASTEXITCODE -ne 0) { Write-Host 'Pillow could not be installed (proxy?); the viewer uses the OpenCV font instead.' -ForegroundColor Yellow }
}

& $Py -m src.launcher @args
Exit-WithPause $LASTEXITCODE
