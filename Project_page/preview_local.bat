@echo off
rem Local preview of the project page: double-click this file. Close the window to stop.
cd /d "%~dp0.."
set PY=python
where python >nul 2>nul || set PY="%LOCALAPPDATA%\anaconda3\python.exe"
start "" http://localhost:8000/Project_page/
%PY% -m http.server 8000
pause
