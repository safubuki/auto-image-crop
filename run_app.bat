@echo off
REM 顔認識自動クロップツール起動スクリプト for Windows
REM このスクリプトはvenv環境をアクティベートし、アプリケーションを起動します

REM 日本語表示のための文字コード設定
chcp 65001 > nul

echo === 顔認識自動クロップツール ===
echo.

REM 仮想環境のパス
set VENV_PATH=venv\Scripts\activate.bat

REM 仮想環境が存在するか確認
if not exist %VENV_PATH% (
    echo エラー: 仮想環境が見つかりません。
    echo まずは以下のコマンドで環境を構築してください:
    echo.
    echo python -m venv venv
    echo pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo 仮想環境をアクティベートしています...
call %VENV_PATH%

echo アプリケーションを起動しています...
python run.py

REM アプリケーション終了後のクリーンアップ
deactivate

echo.
echo アプリケーションを終了しました。
pause