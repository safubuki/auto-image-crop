# Auto Face-Aware Image Cropper

正方形（1:1）の画像から顔を検出し、顔の位置を中心として16:9のアスペクト比に自動クロップするツールです。

## 機能概要

- 1:1の正方形画像から人物の顔を自動検出
- 検出した顔の位置を中心として16:9のアスペクト比に自動クロップ
- シンプルなGUIインターフェイスで簡単操作
- 元画像とクロップ後の画像をリアルタイムでプレビュー
- クロップした画像を任意の場所に保存可能

## 環境構築

### 必要条件

- Python 3.8以上
- Windows、macOS、またはLinux環境

### インストール手順

1. リポジトリをクローン

```
git clone https://github.com/yourusername/auto-image-crop.git
cd auto-image-crop
```

2. 仮想環境の作成とアクティブ化

```
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. 必要なライブラリのインストール

```
pip install -r requirements.txt
```

または、個別にインストールする場合：

```
pip install wheel cmake
pip install numpy pillow PyQt5 opencv-python opencv-contrib-python dlib
```

## 使用方法

1. アプリケーションの起動

```
python run.py
```
または
```
python auto_crop_gui.py
```

2. 「画像を読み込む」ボタンをクリックして、クロップしたい画像を選択
3. 「顔認識してクロップ」ボタンをクリックして、顔検出と自動クロップを実行
4. 「クロップ画像を保存」ボタンをクリックして、クロップした画像を保存

## 顔認識の判定基準

このアプリケーションでは、OpenCVのHaar Cascade分類器を使用して顔の検出を行っています。具体的には以下の判定基準で動作します：

- **検出アルゴリズム**: `haarcascade_frontalface_default.xml` を使用し、正面向きの顔を最も効率的に検出します
- **スケールファクター**: 1.3 - 顔検出の各スケールステップで画像サイズを30%縮小
- **最小近傍数**: 5 - 顔と判定するには最低5つの隣接する検出が必要

顔検出の特性：
- 正面を向いた顔の検出精度が最も高くなります
- 横顔や部分的に隠れた顔は検出精度が低下する場合があります
- 複数の顔が検出された場合、最も大きな顔（画像上で最も大きい面積を持つ顔）を中心にクロップします

## 制約事項

- 入力画像は正方形（1:1）に近いアスペクト比であることが望ましい
- 顔が明確に写っていない画像では正しく動作しない場合があります
- 複数の顔がある場合、最も大きい顔を基準にクロップします

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。
