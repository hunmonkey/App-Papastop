param(
    [switch]$SkipInstall
)

# PapaStop 최소(minimum) 배포 빌드
# - EXE와 실행 엔진(라이브러리)은 그대로 포함한다.
# - 음성인식/번역 "모델"은 동봉하지 않는다. 사용할 PC가 첫 실행 때 자동으로 내려받는다.
#   (Whisper 약 37GB, NLLB CT2 약 8GB → 미동봉으로 약 45GB 절감)
# - CUDA 라이브러리(nvidia)는 모델이 아니고 GPU 가속에 필요하므로 유지한다.
# 결과물은 일반 빌드(dist\PapaStop)와 섞이지 않도록 별도 폴더(dist_min)에 만든다.

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

# 모델(Whisper / NLLB CT2)은 의도적으로 복사하지 않는다.
# papastop.py가 모델이 없으면 첫 사용 시 자동으로 내려받도록 이미 구현되어 있다.

$Guide = @'
PapaStop 최소(minimum) 배포본 안내
============================================================

이 배포본은 용량을 줄이기 위해 음성인식/번역 "모델"을 포함하지 않았습니다.

- 첫 실행 후 음성인식 또는 번역을 처음 사용할 때, 필요한 모델을
  자동으로 인터넷에서 내려받습니다. (한 번 받으면 다음부터는 다시 받지 않습니다.)
- 따라서 처음 사용할 때는 인터넷 연결이 필요하며, 모델 크기에 따라
  첫 사용이 다소 느릴 수 있습니다.
- 받은 모델은 EXE 옆 .papastop_data 폴더 안에 저장됩니다.
  이 폴더를 지우지 마세요. (지우면 다시 내려받습니다.)

CUDA(그래픽카드 가속) 라이브러리는 포함되어 있습니다.
NVIDIA 그래픽카드가 있으면 자동으로 GPU 가속을 사용합니다.
'@
Set-Content -Path (Join-Path $DistDir "PapaStop\README_MINIMUM.txt") -Value $Guide -Encoding UTF8

Write-Host ""
Write-Host "Minimum package created:"
Write-Host "  EXE   : dist_min\PapaStop\PapaStop.exe"
Write-Host "  Data  : dist_min\PapaStop\.papastop_data\nvidia  (CUDA only)"
Write-Host "  Guide : dist_min\PapaStop\README_MINIMUM.txt"
Write-Host "  NOTE  : Speech/translation models download automatically on first use (internet required)."
