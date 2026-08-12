@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0.."

echo.
echo ============================================================
echo  CricVision - Phone live test (free HTTPS tunnel)
echo  Turn OFF Norton VPN before continuing.
echo ============================================================
echo.

where cloudflared >nul 2>&1
if errorlevel 1 (
  echo ERROR: cloudflared not installed.
  echo Run: winget install Cloudflare.cloudflared
  pause
  exit /b 1
)

echo [1/4] Starting backend (port 8000)...
start "CricVision Backend" cmd /k "title CricVision Backend && cd /d %CD% && .venv\Scripts\activate && python -m uvicorn services.api.main:app --host 127.0.0.1 --port 8000"

timeout /t 4 /nobreak >nul

echo [2/4] Starting frontend (port 3000)...
start "CricVision Frontend" cmd /k "title CricVision Frontend && cd /d %CD%\apps\web && set NEXT_PUBLIC_USE_SAME_ORIGIN_API=true && npm run dev -- -H 127.0.0.1 -p 3000"

echo [3/4] Waiting for /live to compile (up to 3 minutes)...
set /a tries=0
:wait_live
set /a tries+=1
curl.exe -s -o NUL -w "%%{http_code}" http://127.0.0.1:8000/health | findstr 200 >nul
if errorlevel 1 (
  if !tries! geq 36 goto wait_fail
  timeout /t 5 /nobreak >nul
  goto wait_live
)
curl.exe -s -o NUL -w "%%{http_code}" http://127.0.0.1:3000/live | findstr 200 >nul
if errorlevel 1 (
  if !tries! geq 36 goto wait_fail
  echo       still compiling... !tries! / 36
  timeout /t 5 /nobreak >nul
  goto wait_live
)
echo       PC ready: http://127.0.0.1:3000/live
goto start_tunnel

:wait_fail
echo.
echo ERROR: Frontend or backend did not start in time.
echo Check the Backend and Frontend windows for red errors.
pause
exit /b 1

:start_tunnel
echo [4/4] Starting HTTPS tunnel...
start "CricVision Tunnel - COPY LINK" cmd /k "title CricVision Tunnel && cloudflared tunnel --url http://127.0.0.1:3000"

echo.
echo ============================================================
echo  READY
echo.
echo  PC test:  http://127.0.0.1:3000/live
echo.
echo  Phone: open the https link from the TUNNEL window
echo         then add /live at the end
echo         example: https://abc.trycloudflare.com/live
echo.
echo  On phone:
echo    - Live Session - Camera Calibration
echo    - Experimental Delivery Test - Start Detection
echo.
echo  Keep all 3 windows open. First phone load: wait 60s.
echo ============================================================
echo.
pause
