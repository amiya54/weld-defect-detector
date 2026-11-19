@echo off
chcp 65001 >nul
title 焊缝缺陷检测 - 检测我的照片
echo ============================================================
echo   检测自己的焊缝照片（批量）
echo ============================================================
echo.

cd /d "%~dp0"

if not exist "我的照片" (
    echo [提示] 没找到"我的照片"文件夹，正在创建。
    echo 请把脱敏后的焊缝照片放进去，再重新双击本文件。
    mkdir "我的照片"
    start "我的照片"
    pause
    exit /b 0
)

if not exist ".venv\Scripts\activate.bat" (
    echo [错误] 还没安装环境，请先双击"一键运行.bat"完成首次安装。
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
echo 正在检测"我的照片"文件夹里的所有图片...
echo.
python detect.py -i "我的照片" -o outputs
echo.
start outputs
pause
