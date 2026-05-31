$ErrorActionPreference = "Stop"

$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $AppDir

$VenvPython = Join-Path $AppDir ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating local Python environment..."
    python -m venv .venv
}

Write-Host "Upgrading pip..."
& $VenvPython -m pip install --upgrade pip

Write-Host "Installing PapaStop engines..."
& $VenvPython -m pip install --upgrade -r .\requirements.txt

# 정밀 자막(WhisperX)용 PyTorch + WhisperX. PyTorch는 용량이 큽니다(약 2~3GB).
# NVIDIA 그래픽카드가 있으면 GPU(CUDA) 빌드를, 없으면 CPU 빌드를 받습니다.
$HasNvidia = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
$TorchIndex = if ($HasNvidia) { "https://download.pytorch.org/whl/cu124" } else { "https://download.pytorch.org/whl/cpu" }
Write-Host "Installing PyTorch + WhisperX for precise subtitles (this download is large)..."
& $VenvPython -m pip install --upgrade torch==2.5.1 torchaudio==2.5.1 --index-url $TorchIndex
& $VenvPython -m pip install --upgrade whisperx

Write-Host "Done. Run .\run_papastop.ps1 to start PapaStop."
