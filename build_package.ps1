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
    --onefile `
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
    --collect-all transformers `
    --exclude-module tests `
    --exclude-module stanza.tests `
    --exclude-module torch.utils.tensorboard `
    --exclude-module numpy.f2py.tests `
    --add-data "README.md;." `
    .\papastop.py

# === EXE 옆 비압축 데이터 폴더(.papastop_data) 구성 ===
# 무겁고 자주 바뀌는 CUDA 라이브러리와 번역 모델은 EXE에 넣지 않고 옆 폴더에 둡니다.
# 배포할 때 dist\PapaStop.exe 와 dist\.papastop_data 를 같은 위치(또는 USB)에 함께 두면
# 설치 없이 포터블로 실행됩니다. 모델만 교체하거나 EXE만 교체하는 업그레이드가 가능합니다.
$DataDir = Join-Path $AppDir "dist\.papastop_data"
$NvidiaSrc = Join-Path $AppDir ".venv\Lib\site-packages\nvidia"
$NllbSrc = Join-Path $AppDir ".papastop_data\nllb-200-3.3B-ct2"
$WhisperSrc = Join-Path $AppDir ".papastop_data\huggingface"

Write-Host "Assembling external data folder (.papastop_data)..."
New-Item -ItemType Directory -Force $DataDir | Out-Null

Write-Host "  - Copying CUDA libraries (nvidia)..."
Copy-Item $NvidiaSrc (Join-Path $DataDir "nvidia") -Recurse -Force

if (Test-Path $NllbSrc) {
    Write-Host "  - Copying NLLB translation model (3.3B)..."
    Copy-Item $NllbSrc (Join-Path $DataDir "nllb-200-3.3B-ct2") -Recurse -Force
} else {
    Write-Host "  - [skip] NLLB 3.3B model not prepared yet (run the app's 'NLLB 모델 준비' first to bundle it)."
}

if (Test-Path $WhisperSrc) {
    Write-Host "  - Copying downloaded Whisper models..."
    Copy-Item $WhisperSrc (Join-Path $DataDir "huggingface") -Recurse -Force
}

Write-Host ""
Write-Host "Package created:"
Write-Host "  EXE   : dist\PapaStop.exe"
Write-Host "  Data  : dist\.papastop_data\  (keep this next to the EXE)"
