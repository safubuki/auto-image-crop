#!/bin/bash
# 顔認識自動クロップツール起動スクリプト for Linux/macOS
# このスクリプトはvenv環境をアクティベートし、アプリケーションを起動します

echo "=== 顔認識自動クロップツール ==="
echo ""

# 仮想環境のパス
VENV_PATH="venv/bin/activate"

# 仮想環境が存在するか確認
if [ ! -f "$VENV_PATH" ]; then
    echo "エラー: 仮想環境が見つかりません。"
    echo "まずは以下のコマンドで環境を構築してください:"
    echo ""
    echo "python3 -m venv venv"
    echo "pip install -r requirements.txt"
    echo ""
    read -p "Press Enter to continue..."
    exit 1
fi

echo "仮想環境をアクティベートしています..."
source "$VENV_PATH"

echo "アプリケーションを起動しています..."
python run.py

# アプリケーション終了後のクリーンアップ
deactivate

echo ""
echo "アプリケーションを終了しました。"
read -p "Press Enter to continue..."