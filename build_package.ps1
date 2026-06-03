param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $AppDir

$VenvPython = Join-Path $AppDir ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating local Python environment..."
    python -m venv .venv
}

if (-not $SkipInstall) {
    Write-Host "Installing runtime engines..."
    & $VenvPython -m pip install --upgrade -r .\requirements.txt

    Write-Host "Installing GPU acceleration libraries (NVIDIA)..."
    & $VenvPython -m pip install --upgrade nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12

    Write-Host "Installing PyInstaller..."
    & $VenvPython -m pip install --upgrade pyinstaller
}

Write-Host "Building PapaStop package (single EXE)..."
& $VenvPython -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onedir `
    --name PapaStop `
    --collect-all faster_whisper `
    --collect-all argostranslate `
    --collect-all imageio_ffmpeg `
    --collect-all soundcard `
    --collect-all ctranslate2 `
    --collect-all tokenizers `
    --collect-all onnxruntime `
    --collect-all huggingface_hub `
    --collect-all av `
    --collect-all sentencepiece `
    --collect-all transformers `
    --collect-all torch `
    --collect-all llama_cpp `
    --exclude-module tests `
    --exclude-module stanza.tests `
    --exclude-module torch.utils.tensorboard `
    --exclude-module numpy.f2py.tests `
    --add-data "README.md;." `
    .\papastop.py

Write-Host "Assembling external data folder (.papastop_data)..."
$DataDir = Join-Path $AppDir "dist\PapaStop\.papastop_data"
$NvidiaSrc = Join-Path $AppDir ".venv\Lib\site-packages\nvidia"
$WhisperSrc = Join-Path $AppDir ".papastop_data\huggingface"

New-Item -ItemType Directory -Force $DataDir | Out-Null

Write-Host "  - Copying CUDA libraries (nvidia)..."
Copy-Item $NvidiaSrc (Join-Path $DataDir "nvidia") -Recurse -Force

if (Test-Path $WhisperSrc) {
    Write-Host "  - Copying downloaded Whisper models..."
    Copy-Item $WhisperSrc (Join-Path $DataDir "huggingface") -Recurse -Force
}

Write-Host "  - Copying NLLB CT2 translation models..."
foreach ($ct2 in @("nllb-200-distilled-600M-ct2", "nllb-200-distilled-1.3B-ct2", "nllb-200-3.3B-ct2")) {
    $src = Join-Path $AppDir ".papastop_data\$ct2"
    if (Test-Path $src) {
        Write-Host "      $ct2"
        Copy-Item $src (Join-Path $DataDir $ct2) -Recurse -Force
    }
}

Write-Host ""
Write-Host "Package created:"
Write-Host "  EXE   : dist\PapaStop\PapaStop.exe"
Write-Host "  Data  : dist\PapaStop\.papastop_data\  (keep this next to the EXE)"
