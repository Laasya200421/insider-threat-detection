@echo off
setlocal
set "BASE=%~dp0"

echo.
echo  ThreatLens - Insider Threat Detection System
echo  ============================================
echo.

:: ── Unzip cert_insider_threat_outputs.zip if not already done ────────────
if exist "%BASE%cert_insider_threat_outputs.zip" (
    if not exist "%BASE%cert_outputs\" (
        echo  Extracting cert_insider_threat_outputs.zip...
        powershell -NoProfile -Command "Expand-Archive -Path '%BASE%cert_insider_threat_outputs.zip' -DestinationPath '%BASE%cert_outputs' -Force"
        echo  Done.
    ) else (
        echo  cert_outputs\ already extracted - skipping unzip.
    )
)

:: ── Install dependencies ─────────────────────────────────────────────────
echo  Installing dependencies...
pip install -r "%BASE%requirements.txt" --quiet
echo.

:: ── Launch Flask ─────────────────────────────────────────────────────────
echo  Starting Flask server on http://localhost:5000
echo.
cd /d "%BASE%app"
python main.py
pause
