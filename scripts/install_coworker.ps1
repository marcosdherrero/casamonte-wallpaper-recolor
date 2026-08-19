# Download a private CPython 3.12 into runtime\ (no admin, no PATH),
# install Pillow / numpy / matplotlib, then write launchers.
#
# Uses the official nuget.org python package (stdlib, no Tk).
# Tkinter is grafted from python.org tcltk.msi into runtime\ (no admin).
# Last resort: per-user installer under %LOCALAPPDATA%\WallpaperRecolor\python.

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$PyVersion = "3.12.10"
$RuntimeDir = Join-Path $Root "runtime"
$PythonExe = Join-Path $RuntimeDir "python.exe"
$CacheDir = Join-Path $Root ".cache"
$NugetUrl = "https://www.nuget.org/api/v2/package/python/$PyVersion"
$GetPipUrl = "https://bootstrap.pypa.io/get-pip.py"
$TclTkMsiUrl = "https://www.python.org/ftp/python/$PyVersion/amd64/tcltk.msi"
$OfficialExeUrl = "https://www.python.org/ftp/python/$PyVersion/python-$PyVersion-amd64.exe"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message"
}

function Get-File([string]$Url, [string]$Dest) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Dest) | Out-Null
    if ((Test-Path $Dest) -and ((Get-Item $Dest).Length -gt 1000)) {
        Write-Host "Using cached $Dest"
        return
    }
    Write-Host "Downloading $Url"
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        & curl.exe -L --fail --retry 3 --retry-delay 2 -o $Dest $Url
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $Url" }
    } else {
        Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
    }
}

function Invoke-PythonCheck {
    param([string]$Exe, [string]$Code)
    & $Exe -c $Code
    return ($LASTEXITCODE -eq 0)
}

function Install-FromNuget {
    Write-Step "Installing CPython $PyVersion next to the app (nuget, no admin)"
    New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
    $nupkg = Join-Path $CacheDir "python-$PyVersion.nupkg"
    Get-File $NugetUrl $nupkg

    $extract = Join-Path $CacheDir "python-$PyVersion-nupkg"
    if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
    New-Item -ItemType Directory -Force -Path $extract | Out-Null
    & tar.exe -xf $nupkg -C $extract
    if ($LASTEXITCODE -ne 0) { throw "Could not extract $nupkg" }

    $tools = Join-Path $extract "tools"
    if (-not (Test-Path (Join-Path $tools "python.exe"))) {
        throw "nuget package missing tools\python.exe"
    }
    if (Test-Path $RuntimeDir) { Remove-Item -Recurse -Force $RuntimeDir }
    New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
    Copy-Item -Path (Join-Path $tools "*") -Destination $RuntimeDir -Recurse -Force

    Get-ChildItem -Path $RuntimeDir -Filter "python*._pth" -ErrorAction SilentlyContinue | ForEach-Object {
        $text = Get-Content -Raw $_.FullName
        if ($text -notmatch '(?m)^import site\s*$') {
            $text = $text -replace '(?m)^#\s*import site\s*$', 'import site'
            if ($text -notmatch '(?m)^import site\s*$') {
                $text = $text.TrimEnd() + "`r`nimport site`r`n"
            }
            Set-Content -Path $_.FullName -Value $text -NoNewline -Encoding ascii
        }
    }
}

function Install-TkinterFromMsi {
    Write-Step "Adding Tcl/Tk into runtime\ (python.org tcltk.msi, no admin)"
    New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
    $msi = Join-Path $CacheDir "tcltk.msi"
    Get-File $TclTkMsiUrl $msi
    $extract = Join-Path $CacheDir "tcltk-extract"
    if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
    New-Item -ItemType Directory -Force -Path $extract | Out-Null
    $p = Start-Process -FilePath "msiexec.exe" -ArgumentList @("/a", $msi, "/qn", "TARGETDIR=$extract") -Wait -PassThru
    if ($p.ExitCode -ne 0) {
        throw "msiexec extract of tcltk.msi exited $($p.ExitCode)"
    }
    $dllsSrc = Join-Path $extract "DLLs"
    $dllsDst = Join-Path $RuntimeDir "DLLs"
    if (-not (Test-Path $dllsSrc)) { throw "tcltk.msi extract missing DLLs\" }
    New-Item -ItemType Directory -Force -Path $dllsDst | Out-Null
    Copy-Item -Path (Join-Path $dllsSrc "*") -Destination $dllsDst -Force
    foreach ($dll in @("tcl86t.dll", "tk86t.dll")) {
        $from = Join-Path $dllsSrc $dll
        if (Test-Path $from) {
            Copy-Item $from (Join-Path $RuntimeDir $dll) -Force
        }
    }
    $tkSrc = Join-Path $extract "Lib\tkinter"
    $tkDst = Join-Path $RuntimeDir "Lib\tkinter"
    if (-not (Test-Path $tkSrc)) { throw "tcltk.msi extract missing Lib\tkinter" }
    if (Test-Path $tkDst) { Remove-Item -Recurse -Force $tkDst }
    Copy-Item -Path $tkSrc -Destination $tkDst -Recurse -Force
    $tclSrc = Join-Path $extract "tcl"
    $tclDst = Join-Path $RuntimeDir "tcl"
    if (-not (Test-Path $tclSrc)) { throw "tcltk.msi extract missing tcl\" }
    if (Test-Path $tclDst) { Remove-Item -Recurse -Force $tclDst }
    Copy-Item -Path $tclSrc -Destination $tclDst -Recurse -Force
}

function Install-FromOfficialInstaller {
    Write-Step "Per-user CPython into %LOCALAPPDATA%\WallpaperRecolor\python (no all-users)"
    $localRuntime = Join-Path $env:LOCALAPPDATA "WallpaperRecolor\python"
    $setup = Join-Path $CacheDir "python-$PyVersion-amd64.exe"
    Get-File $OfficialExeUrl $setup
    $installArgs = @(
        "/quiet",
        "InstallAllUsers=0",
        "PrependPath=0",
        "Include_launcher=0",
        "Include_test=0",
        "Include_doc=0",
        "Include_dev=0",
        "Include_debug=0",
        "Include_symbols=0",
        "AssociateFiles=0",
        "Shortcuts=0",
        "CompileAll=0",
        "Include_pip=1",
        "Include_tcltk=1",
        "TargetDir=$localRuntime"
    )
    $p = Start-Process -FilePath $setup -ArgumentList $installArgs -Wait -PassThru
    if ($p.ExitCode -ne 0) {
        throw "Python installer exited $($p.ExitCode). Try Install.bat again, or ask IT to allow python.org."
    }
    $script:PythonExe = Join-Path $localRuntime "python.exe"
    $script:RuntimeDir = $localRuntime
    if (-not (Test-Path $script:PythonExe)) {
        throw "Python installer did not create $localRuntime\python.exe"
    }
}

function Ensure-Pip {
    param([string]$Exe)
    & $Exe -m pip --version
    if ($LASTEXITCODE -eq 0) { return }
    Write-Step "Installing pip"
    $getPip = Join-Path $CacheDir "get-pip.py"
    Get-File $GetPipUrl $getPip
    & $Exe $getPip --no-warn-script-location
    if ($LASTEXITCODE -ne 0) { throw "get-pip.py failed" }
}

function Install-AppDeps {
    param([string]$Exe)
    Write-Step "Installing app packages (Pillow, numpy, matplotlib; no EasyOCR)"
    $req = Join-Path $Root "requirements.txt"
    $plot = Join-Path $Root "requirements-plot.txt"
    & $Exe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
    & $Exe -m pip install -r $req -r $plot
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
}

function Test-AppImport {
    param([string]$Exe)
    Write-Step "Checking Tk, Pillow, numpy, matplotlib, and the app"
    & $Exe -c "import tkinter, PIL, numpy, matplotlib; from wallpaper_recolor.ui.app import WallpaperRecolorApp; print(WallpaperRecolorApp.__name__)"
    if ($LASTEXITCODE -ne 0) { throw "Import check failed" }
}

function Write-Shortcuts {
    $wscript = New-Object -ComObject WScript.Shell
    $pyw = Join-Path $RuntimeDir "pythonw.exe"
    $target = if (Test-Path $pyw) { $pyw } else { $PythonExe }

    foreach ($dir in @(
        [Environment]::GetFolderPath("Desktop"),
        (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs")
    )) {
        if (-not $dir) { continue }
        try {
            $lnkPath = Join-Path $dir "Wallpaper Recolor.lnk"
            $sc = $wscript.CreateShortcut($lnkPath)
            $sc.TargetPath = $target
            $sc.Arguments = "`"$Root\run_app.py`""
            $sc.WorkingDirectory = $Root
            $sc.WindowStyle = 1
            $sc.Description = "Wallpaper Recolor"
            $sc.Save()
            Write-Host "Shortcut: $lnkPath"
        } catch {
            Write-Host "Could not write shortcut in $dir ($($_.Exception.Message))"
        }
    }
}

Write-Host "Wallpaper Recolor - local Python install"
Write-Host "Folder: $Root"
Write-Host "No admin, no system Python, no EasyOCR/LaMa."

$haveRuntime = (Test-Path $PythonExe) -and (Invoke-PythonCheck $PythonExe "import sys; assert sys.version_info[:2] == (3, 12)")
if ($haveRuntime) {
    Write-Host "Found existing runtime\python.exe"
} else {
    Install-FromNuget
    if (-not (Test-Path $PythonExe)) { throw "runtime\python.exe missing after nuget extract" }
}

if (-not (Invoke-PythonCheck $PythonExe "import tkinter")) {
    Install-TkinterFromMsi
    if (-not (Invoke-PythonCheck $PythonExe "import tkinter")) {
        Write-Host "Folder-local Tkinter still missing; trying per-user installer under LocalAppData."
        Install-FromOfficialInstaller
        $PythonExe = $script:PythonExe
        $RuntimeDir = $script:RuntimeDir
        if (-not (Invoke-PythonCheck $PythonExe "import tkinter")) {
            throw "Tkinter is still missing. This PC may block Python installs."
        }
    }
}

Ensure-Pip $PythonExe
Install-AppDeps $PythonExe
Test-AppImport $PythonExe
if ($env:WALLPAPER_RECOLOR_SKIP_SHORTCUTS -ne "1") {
    Write-Shortcuts
}

Write-Host ""
Write-Host "Install finished. Double-click WallpaperRecolor.bat (offline after this)."
Write-Host "Labels Detect/Remove stay unavailable unless you later pip-install OCR extras."
