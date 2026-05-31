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

Write-Host "Done. Run .\run_papastop.ps1 to start PapaStop."
