@echo off
title SuGaR vs 3DGS Web Dashboard (8 Tipologi Patung Bali)
echo ================================================================
echo    Web Dashboard Skripsi: 3DGS vs SuGaR (8 Tipologi Patung Bali)
echo ================================================================
echo Membuka browser default ke http://localhost:8080 ...
start http://localhost:8080
echo Menjalankan Web Runner Server...
echo (Tekan Ctrl+C di jendela ini untuk menutup server)
echo.
python "%~dp0web_dashboard\server.py" 8080
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Server berhenti dengan error code %ERRORLEVEL%.
    pause
)
