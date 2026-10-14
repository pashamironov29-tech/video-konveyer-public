@echo off
chcp 65001 >nul
cd /d "%~dp0"
if "%~1"=="" (
  set /p SRC=Ссылка или путь к видео / Video URL or path: 
) else (
  set "SRC=%~1"
)
py -m konveyer run "%SRC%"
pause
