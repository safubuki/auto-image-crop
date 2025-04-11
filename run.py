#!/usr/bin/env python
"""
顔認識自動クロップツールのスタートアップスクリプト
"""

import glob
import os
import sys

import cv2
from PyQt5.QtWidgets import QApplication

# GUIアプリケーションのインポート
from auto_crop_gui import FaceCropApp
from image_processor import ImageProcessor


def main():
    """
    アプリケーションのメイン関数
    
    コマンドライン引数に応じて、GUIモードまたはバッチ処理モードを実行します。
    引数なしの場合はGUIモードで起動します。
    
    引数:
        なし
        
    戻り値:
        なし
    """
    if len(sys.argv) < 2:
        # 引数がない場合はGUIモードで起動
        app = QApplication(sys.argv)
        window = FaceCropApp()
        window.show()
        sys.exit(app.exec_())
    else:
        # バッチ処理モード（コマンドライン引数から入力と出力のパスを取得）
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None

        if os.path.isfile(input_path):
            # 単一のファイル処理
            process_single_image(input_path, output_path)
        elif os.path.isdir(input_path):
            # ディレクトリ内のすべての画像を処理
            process_directory(input_path, output_path)
        else:
            print(f"エラー: 指定されたパス '{input_path}' が見つかりません。")
            sys.exit(1)


def process_single_image(input_path, output_path=None):
    """
    単一の画像を処理するメソッド
    
    指定された画像を読み込み、顔認識とクロップ処理を適用して保存します。
    
    引数:
        input_path: 入力画像のパス
        output_path: 出力画像のパス（Noneの場合は元のファイル名に '_cropped' を付加）
        
    戻り値:
        なし
    """
    try:
        # 画像を読み込み
        image = cv2.imread(input_path)
        if image is None:
            print(f"エラー: 画像 '{input_path}' を読み込めませんでした。")
            return False

        # ImageProcessorを使用して画像をクロップ
        processor = ImageProcessor()
        cropped_image = processor.crop_image(image)

        if cropped_image is None:
            print(f"警告: 画像 '{input_path}' から顔を検出できませんでした。")
            return False

        # 出力パスが指定されていない場合は、元のファイル名に '_cropped' を付加
        if output_path is None:
            base_name, ext = os.path.splitext(input_path)
            output_path = f"{base_name}_cropped{ext}"

        # クロップ画像を保存
        cv2.imwrite(output_path, cropped_image)
        print(f"クロップした画像を保存しました: '{output_path}'")
        return True

    except Exception as e:
        print(f"処理エラー: {str(e)}")
        return False


def process_directory(input_dir, output_dir=None):
    """
    ディレクトリ内のすべての画像を処理するメソッド
    
    指定されたディレクトリ内のすべての画像に顔認識とクロップ処理を適用します。
    
    引数:
        input_dir: 入力画像が格納されたディレクトリのパス
        output_dir: 出力画像を保存するディレクトリのパス（Noneの場合は入力ディレクトリに '_cropped' サブディレクトリを作成）
        
    戻り値:
        なし
    """
    # 出力ディレクトリが指定されていない場合は、入力ディレクトリに '_cropped' サブディレクトリを作成
    if output_dir is None:
        output_dir = os.path.join(input_dir, "cropped")

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    # サポートされている画像拡張子
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    # 画像ファイルのリストを取得
    image_files = []
    for ext in image_extensions:
        pattern = os.path.join(input_dir, ext)
        image_files.extend(glob.glob(pattern))
        # 大文字の拡張子もチェック
        pattern = os.path.join(input_dir, ext.upper())
        image_files.extend(glob.glob(pattern))

    if not image_files:
        print(f"警告: ディレクトリ '{input_dir}' に画像ファイルが見つかりませんでした。")
        return

    # 各画像を処理
    success_count = 0
    for input_path in image_files:
        # 出力ファイル名を生成
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)

        # 画像を処理
        if process_single_image(input_path, output_path):
            success_count += 1

    print(f"処理完了: {len(image_files)} 個の画像中 {success_count} 個を正常にクロップしました。")


if __name__ == "__main__":
    main()
