@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -m pip install -r requirements.txt
if not exist konveyer.toml copy konveyer.toml.example konveyer.toml >nul
if not exist .env copy .env.example .env >nul
py -m konveyer doctor
pause
