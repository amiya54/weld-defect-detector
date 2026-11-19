@echo off
chcp 65001 >nul
title 焊缝缺陷检测 - 一键运行
echo ============================================================
echo   焊缝外观缺陷检测 - 一键安装并运行
echo   首次运行会自动安装环境，可能需要 5-15 分钟，请耐心等待
echo ============================================================
echo.

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 没找到 Python。
    echo 请先到 https://www.python.org/downloads/ 下载安装 Python 3.10 或 3.11，
    echo 安装时务必勾选 "Add Python to PATH"，装完重新双击本文件。
    echo.
    pause
    exit /b 1
)

python --version
echo.

if not exist ".venv" (
    echo [1/4] 正在创建虚拟环境...
    python -m venv .venv
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败，可能是 Python 版本问题，建议用 3.10 或 3.11。
        pause
        exit /b 1
    )
) else (
    echo [1/4] 虚拟环境已存在，跳过。
)

echo [2/4] 正在安装依赖（用清华镜像源加速）...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple -q
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo.
    echo [提示] 如果 torch 安装失败，尝试手动安装 CPU 版：
    echo   pip install torch --index-url https://download.pytorch.org/whl/cpu
    echo 然后重新双击本文件。
    pause
    exit /b 1
)

echo [3/4] 运行示例图检测...
python detect.py -i samples/demo.jpg -o outputs

echo [4/4] 完成！正在打开结果文件夹...
echo.
echo ============================================================
echo   结果说明：
echo   outputs\demo_vis.jpg    = 画框检测结果图
echo   outputs\demo_result.json = 缺陷类别/坐标/置信度
echo   outputs\summary.json     = 汇总
echo ============================================================
echo.
start outputs
pause
