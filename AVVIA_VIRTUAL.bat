@echo off
rem Start Unity Capture and keep the camera/mode selectors available.
call "%~dp0AVVIA_LOCALE.bat" --virtual-camera --virtual-backend unitycapture %*
exit /b %errorlevel%
