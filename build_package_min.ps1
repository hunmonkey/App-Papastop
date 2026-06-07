param(
    [switch]$SkipInstall
)

# PapaStop 최소(minimum) 배포 빌드
# - 실행 엔진(라이브러리)은 포함합니다.
# - 음성인식/번역 모델은 포함하지 않습니다.
# - 모델은 사용자가 처음 기능을 실행할 때 자동으로 내려받습니다.
# - CUDA 라이브러리는 GPU 가속에 필요하므로 포함합니다.
# - 일반 빌드(dist\PapaStop)와 섞이지 않도록 dist_min/build_min을 사용합니다.
# - 권장 실행 명령:
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\build_package_min.ps1 -SkipInstall
# - 빌드 뒤 `dist_min\PapaStop\.papastop_data`에 Hugging Face 모델 캐시가 있으면 최소 배포본이 아닙니다.
# - PyInstaller가 onnxruntime, torch tensorboard, llama_cpp, CUDA DLL 경고를 보여도 항상 실패는 아닙니다.
#   마지막 `Package created:` 문구와 `dist_min\PapaStop\PapaStop.exe`를 함께 확인합니다.

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

# README_MINIMUM 안내문은 UTF-8 base64로 보관합니다.
# Windows PowerShell 5에서 한글 here-string이 깨져 저장되는 실수를 막기 위한 장치입니다.
$GuideBase64 = (
    "UGFwYVN0b3Ag7LWc7IaMKG1pbmltdW0pIOuwsO2PrOuzuCDslYjrgrQKPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CgrsnbQg67Cw7Y+s67O47J2AIOyaqeufieydhCDspITsnbTquLAg7JyE7ZW0IOydjOyEseyduOyLnS/rsojsl60gIuuqqOuNuCLsnYQg7Y+s7ZWo7ZWY7KeAIOyViuyKteuLiOuLpC4KCi0g7LKY7J2MIOyLpO2Wie2VnCDrkqQg7J2M7ISx7J247Iud7J2064KYIOuyiOyXreydhCDsspjsnYwg7IKs7Jqp7ZWgIOuVjCDtlYTsmpTtlZwg66qo64247J2EIOyduO2EsOuEt+yXkOyEnCDsnpDrj5nsnLzroZwg64K066Ck67Cb7Iq164uI64ukLgotIO2VnCDrsogg64K066Ck67Cb7J2AIOuqqOuNuOydgCBQYXBhU3RvcC5leGUg7JiG7J2YIC5wYXBhc3RvcF9kYXRhIO2PtOuNlCDslYjsl5Ag7KCA7J6l65Cp64uI64ukLgotIC5wYXBhc3RvcF9kYXRhIO2PtOuNlOulvCDsp4DsmrDrqbQg66qo64247J2EIOuLpOyLnCDrgrTroKTrsJvslYTslbwg7ZWp64uI64ukLgotIOyymOydjCDsgqzsmqntlZjripQgUEPsl5DshJzripQg7J247YSw64S3IOyXsOqysOydtCDtlYTsmpTtlZjqs6AsIOuqqOuNuCDtgazquLDsl5Ag65Sw6528IOyyqyDsgqzsmqkg7Iuc6rCE7J20IOyYpOuemCDqsbjrprQg7IiYIOyeiOyKteuLiOuLpC4KCkNVREEoTlZJRElBIOq3uOuemO2Uvey5tOuTnCDqsIDsho3smqkg65287J2067iM65+s66asKeuKlCDtj6ztlajrkJjslrQg7J6I7Iq164uI64ukLgpOVklESUEg6re4656Y7ZS97Lm065Oc6rCAIOyeiOycvOuptCDsnpDrj5nsnLzroZwgR1BVIOqwgOyGjeydhCDsgqzsmqntlaAg7IiYIOyeiOyKteuLiOuLpC4="
)
$Guide = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($GuideBase64))
Set-Content -Path (Join-Path $DistDir "PapaStop\README_MINIMUM.txt") -Value $Guide -Encoding UTF8

Write-Host ""
Write-Host "Minimum package created:"
Write-Host "  EXE   : dist_min\PapaStop\PapaStop.exe"
Write-Host "  Data  : dist_min\PapaStop\.papastop_data\nvidia  (CUDA only)"
Write-Host "  Guide : dist_min\PapaStop\README_MINIMUM.txt"
Write-Host "  NOTE  : Speech/translation models download automatically on first use (internet required)."
