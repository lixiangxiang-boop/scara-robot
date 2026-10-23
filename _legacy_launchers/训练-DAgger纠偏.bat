@echo off
chcp 65001 >nul
call "%~dp0..\menu.bat" --run train_dagger
