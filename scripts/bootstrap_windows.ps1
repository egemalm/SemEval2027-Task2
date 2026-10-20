<#
.SYNOPSIS
Creates the Windows virtual environment and downloads public DiCo-NLI Track 1 assets.

.DESCRIPTION
Run this from PowerShell after cloning the repository. It deliberately uses
the Python launcher rather than GNU Make, which Windows does not include.
#>

[CmdletBinding()]
param(
    [string]$Python = "py",
    [string]$PythonVersion = "3.12",
    [switch]$SkipData
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepositoryRoot

$LauncherArguments = @()
if ($Python -eq "py") {
    $LauncherArguments = @("-$PythonVersion")
}

$VersionOutput = & $Python @LauncherArguments --version
if ($LASTEXITCODE -ne 0) {
    throw "Could not run '$Python'. Install Python 3.10+ from https://www.python.org/downloads/."
}
if ($VersionOutput -notmatch "Python 3\.(1[0-9]|[2-9][0-9])") {
    throw "DiCo-NLI requires Python 3.10+. Found: $VersionOutput"
}

& $Python @LauncherArguments -m venv .venv
$VenvPython = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -e ".[dev]"

if (-not $SkipData) {
    & $VenvPython scripts\fetch_official_assets.py --include-scorer
}

Write-Host "Setup complete. Verify it with: .\.venv\Scripts\python.exe -m pytest"
