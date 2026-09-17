@echo off
setlocal
call D:\anaconda\condabin\conda.bat activate D:\anaconda\envs\instrument-icg
if errorlevel 1 exit /b %errorlevel%
cd /d "%~dp0"
python track_icg.py --data-root "%~dp0data\simulated_data\data30" --instrument "%~dp0data\simulated_instrument" %*
exit /b %errorlevel%
