param(
    [switch]$SkipInstall
)

# PapaStop 최소(minimum) 배포 빌드
# - 실행 엔진(라이브러리)은 포함합니다.
# - 음성인식/번역 모델은 포함하지 않습니다.
# - 모델은 사용자가 처음 기능을 실행할 때 자동으로 내려받습니다.
# - CUDA 라이브러리는 GPU 가속에 필요하므로 포함합니다.
# - 일반 빌드(dist\PapaStop)와 섞이지 않도록 dist_min/build_min을 사용합니다.

$ErrorActionPreference = "Stop"

$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $AppDir

$VenvPython = Join-Path $AppDir ".venv\Scripts\python.exe"
$DistDir = Join-Path $AppDir "dist_min"
$WorkDir = Join-Path $AppDir "build_min"

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

Write-Host "Building PapaStop MINIMUM package (single EXE, models excluded)..."
& $VenvPython -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onedir `
    --name PapaStop `
    --distpath $DistDir `
    --workpath $WorkDir `
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

Write-Host "Assembling minimum data folder (.papastop_data, CUDA only)..."
$DataDir = Join-Path $DistDir "PapaStop\.papastop_data"
$NvidiaSrc = Join-Path $AppDir ".venv\Lib\site-packages\nvidia"

New-Item -ItemType Directory -Force $DataDir | Out-Null

Write-Host "  - Copying CUDA libraries (nvidia)..."
Copy-Item $NvidiaSrc (Join-Path $DataDir "nvidia") -Recurse -Force

# 모델(Whisper / NLLB CT2)은 일부러 복사하지 않습니다.
# papastop.py가 모델이 없으면 첫 사용 때 자동으로 내려받도록 이미 구현되어 있습니다.

$Guide = @'
PapaStop 최소(minimum) 배포본 안내
============================================================

이 배포본은 용량을 줄이기 위해 음성인식/번역 "모델"을 포함하지 않습니다.

- 처음 실행한 뒤 음성인식이나 번역을 처음 사용할 때 필요한 모델을 인터넷에서 자동으로 내려받습니다.
- 한 번 내려받은 모델은 PapaStop.exe 옆의 .papastop_data 폴더 안에 저장됩니다.
- .papastop_data 폴더를 지우면 모델을 다시 내려받아야 합니다.
- 처음 사용하는 PC에서는 인터넷 연결이 필요하고, 모델 크기에 따라 첫 사용 시간이 오래 걸릴 수 있습니다.

CUDA(NVIDIA 그래픽카드 가속용 라이브러리)는 포함되어 있습니다.
NVIDIA 그래픽카드가 있으면 자동으로 GPU 가속을 사용할 수 있습니다.
'@
Set-Content -Path (Join-Path $DistDir "PapaStop\README_MINIMUM.txt") -Value $Guide -Encoding UTF8

Write-Host ""
Write-Host "Minimum package created:"
Write-Host "  EXE   : dist_min\PapaStop\PapaStop.exe"
Write-Host "  Data  : dist_min\PapaStop\.papastop_data\nvidia  (CUDA only)"
Write-Host "  Guide : dist_min\PapaStop\README_MINIMUM.txt"
Write-Host "  NOTE  : Speech/translation models download automatically on first use (internet required)."
