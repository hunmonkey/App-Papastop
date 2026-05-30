param(
    [switch]$PrepareOnly
)

$ErrorActionPreference = "Stop"

$AppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $AppDir

$VenvPython = Join-Path $AppDir ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating local Python environment..."
    python -m venv .venv
}

if ($PrepareOnly) {
    Write-Host "Local Python environment is ready."
    exit 0
}

Write-Host "Starting PapaStop..."
& $VenvPython .\papastop.py
