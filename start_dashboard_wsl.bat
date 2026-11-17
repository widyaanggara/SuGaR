n@echo off
setlocal
title TogogLab Dashboard - WSL Ubuntu-22.04-E
echo Menjalankan TogogLab di http://localhost:8080
echo Biarkan terminal ini terbuka selama dashboard digunakan.
echo Tekan Ctrl+C untuk menghentikan dashboard.
echo.
wsl.exe -d Ubuntu-22.04-E -u anggara -- bash -lc "cd /home/anggara/SuGaR && source /home/anggara/miniconda3/etc/profile.d/conda.sh && conda activate sugar && bash start_dashboard.sh 8080"
set "DASHBOARD_EXIT=%ERRORLEVEL%"
echo.
echo Dashboard berhenti dengan kode %DASHBOARD_EXIT%.
pause
exit /b %DASHBOARD_EXIT%
