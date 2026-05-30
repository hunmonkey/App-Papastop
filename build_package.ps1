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

Write-Host "Building PapaStop package..."
& $VenvPython -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onedir `
    --name PapaStop `
    --collect-all faster_whisper `
    --collect-all argostranslate `
    --collect-all imageio_ffmpeg `
    --collect-all ctranslate2 `
    --collect-all tokenizers `
    --collect-all onnxruntime `
    --collect-all huggingface_hub `
    --collect-all av `
    --collect-all sentencepiece `
    --collect-all nvidia.cublas `
    --collect-all nvidia.cudnn `
    --collect-all nvidia.cuda_runtime `
    --collect-all nvidia.cuda_nvrtc `
    --collect-all transformers `
    --exclude-module tests `
    --exclude-module stanza.tests `
    --exclude-module torch.utils.tensorboard `
    --exclude-module numpy.f2py.tests `
    --add-data "README.md;." `
    --add-data ".papastop_cache/nllb-200-distilled-1.3B-ct2;nllb-200-distilled-1.3B-ct2" `
    .\papastop.py

Write-Host "Package created: dist\PapaStop\PapaStop.exe"
