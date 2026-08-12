@echo off
cd /d "%~dp0.."
echo CricVision phone setup check
echo.

curl.exe -s http://127.0.0.1:8000/health 2>nul | findstr ok >nul
if errorlevel 1 (echo [FAIL] Backend not running on port 8000) else (echo [OK] Backend http://127.0.0.1:8000/health)

curl.exe -s -o NUL -w "%%{http_code}" http://127.0.0.1:3000/live 2>nul | findstr 200 >nul
if errorlevel 1 (echo [FAIL] Frontend not running on port 3000) else (echo [OK] Frontend http://127.0.0.1:3000/live)

curl.exe -s -o NUL -w "%%{http_code}" http://127.0.0.1:3000/cricvision-api/health 2>nul | findstr 200 >nul
if errorlevel 1 (echo [FAIL] API proxy /cricvision-api not working) else (echo [OK] API proxy for phone tunnel)

where cloudflared >nul 2>&1
if errorlevel 1 (echo [FAIL] cloudflared not installed) else (echo [OK] cloudflared installed)

echo.
echo Wi-Fi IP for LAN access (optional): 
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4" ^| findstr /v "172. 10.252"') do echo   %%a
echo.
pause
