import os
import sys

import cv2
import mediapipe as mp
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
                             QMessageBox, QPushButton, QVBoxLayout, QWidget)

# ImageProcessorクラスのインポート
from image_processor import ImageProcessor


class FaceCropApp(QMainWindow):
    """
    顔認識自動クロップツールのGUIアプリケーションクラス
    
    顔認識を使用して画像をクロップし、16:9のアスペクト比に調整するアプリケーション。
    """

    def __init__(self):
        """
        FaceCropAppクラスの初期化メソッド
        
        引数:
            なし
            
        戻り値:
            なし
        """
        super().__init__()

        self.title = '顔認識自動クロップツール'
        self.original_image = None  # 元画像を保持する変数
        self.cropped_image = None  # クロップ後の画像を保持する変数（保存用）
        self.display_image = None  # 表示用の画像（デバッグ情報付き）

        # ImageProcessorクラスの初期化
        self.image_processor = ImageProcessor()

        # デバッグモードの追加（表示用）
        self.debug_mode = True

        self.init_ui()

    def init_ui(self):
        """
        UIコンポーネントの初期化と配置を行うメソッド
        
        引数:
            なし
            
        戻り値:
            なし
        """
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 1000, 600)  # ウィンドウのサイズと位置を設定

        # メインウィジェットとレイアウト
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # 画像表示エリア
        image_layout = QHBoxLayout()

        # 元画像表示ラベル
        self.original_image_label = QLabel()
        self.original_image_label.setFrameStyle(QFrame.Box)
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setText("元画像がここに表示されます")
        self.original_image_label.setMinimumSize(400, 400)

        # クロップ後画像表示ラベル
        self.cropped_image_label = QLabel()
        self.cropped_image_label.setFrameStyle(QFrame.Box)
        self.cropped_image_label.setAlignment(Qt.AlignCenter)
        self.cropped_image_label.setText("クロップ後の画像がここに表示されます")
        self.cropped_image_label.setMinimumSize(400, 225)  # 16:9のアスペクト比

        image_layout.addWidget(self.original_image_label)
        image_layout.addWidget(self.cropped_image_label)

        # ボタンエリア
        button_layout = QHBoxLayout()

        # 画像読み込みボタン
        self.load_button = QPushButton('画像を読み込む')
        self.load_button.clicked.connect(self.load_image)

        # クロップボタン
        self.crop_button = QPushButton('顔認識してクロップ')
        self.crop_button.clicked.connect(self.crop_image)
        self.crop_button.setEnabled(False)  # 初期状態では無効化

        # 保存ボタン
        self.save_button = QPushButton('クロップ画像を保存')
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)  # 初期状態では無効化

        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.crop_button)
        button_layout.addWidget(self.save_button)

        # レイアウトをメインウィジェットに追加
        main_layout.addLayout(image_layout)
        main_layout.addLayout(button_layout)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def load_image(self):
        """
        画像ファイルを選択して読み込むメソッド
        
        引数:
            なし
            
        戻り値:
            なし
        """
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self,
                                                  "画像ファイルを開く",
                                                  "",
                                                  "画像ファイル (*.jpg *.jpeg *.png *.bmp);;すべてのファイル (*)",
                                                  options=options)

        if filepath:
            try:
                # 画像を読み込み
                self.original_image = cv2.imread(filepath)
                if self.original_image is None:
                    raise Exception("画像の読み込みに失敗しました")

                # 画像をQImageに変換して表示
                height, width, channels = self.original_image.shape
                bytes_per_line = channels * width
                q_img = QImage(self.original_image.data, width, height, bytes_per_line,
                               QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(q_img)

                # 元画像ラベルのサイズに合わせてスケーリング
                scaled_pixmap = pixmap.scaled(self.original_image_label.width(),
                                              self.original_image_label.height(),
                                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.original_image_label.setPixmap(scaled_pixmap)

                # クロップボタンを有効に
                self.crop_button.setEnabled(True)
                self.cropped_image = None
                self.display_image = None
                self.cropped_image_label.setText("クロップ後の画像がここに表示されます")
                self.save_button.setEnabled(False)

            except Exception as e:
                QMessageBox.critical(self, "エラー", f"画像の読み込みエラー: {str(e)}")

    def crop_image(self):
        """
        読み込まれた画像に顔認識を適用してクロップするメソッド
        
        引数:
            なし
            
        戻り値:
            なし
        """
        if self.original_image is None:
            return

        try:
            # 保存用の画像（デバッグ情報なし）
            self.cropped_image = self.image_processor.crop_image(self.original_image,
                                                                 debug_mode=False)

            if self.cropped_image is None:
                QMessageBox.warning(self, "警告", "画像から顔を検出できませんでした")
                return

            # 表示用の画像（デバッグ情報あり）
            if self.debug_mode:
                self.display_image = self.image_processor.crop_image(self.original_image,
                                                                     debug_mode=True)
            else:
                self.display_image = self.cropped_image.copy()

            # 表示用の画像をQPixmapに変換して表示
            height, width, channels = self.display_image.shape
            bytes_per_line = channels * width
            q_img = QImage(self.display_image.data, width, height, bytes_per_line,
                           QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)

            scaled_pixmap = pixmap.scaled(self.cropped_image_label.width(),
                                          self.cropped_image_label.height(), Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)
            self.cropped_image_label.setPixmap(scaled_pixmap)

            # 保存ボタンを有効化
            self.save_button.setEnabled(True)

        except Exception as e:
            import traceback
            error_message = f"クロップ処理エラー: {str(e)}\n{traceback.format_exc()}"
            QMessageBox.critical(self, "エラー", error_message)

    def save_image(self):
        """
        クロップされた画像をファイルに保存するメソッド
        
        引数:
            なし
            
        戻り値:
            なし
        """
        if self.cropped_image is None:
            return

        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getSaveFileName(self,
                                                  "クロップした画像を保存",
                                                  "",
                                                  "JPEGファイル (*.jpg);;PNGファイル (*.png);;すべてのファイル (*)",
                                                  options=options)

        if filepath:
            try:
                # ファイル拡張子の確認と追加
                _, ext = os.path.splitext(filepath)
                if not ext:
                    filepath += '.jpg'  # デフォルトはJPEG形式

                # デバッグ情報のない画像を保存
                cv2.imwrite(filepath, self.cropped_image)
                QMessageBox.information(self, "成功", f"画像を保存しました: {filepath}")

            except Exception as e:
                QMessageBox.critical(self, "エラー", f"画像の保存エラー: {str(e)}")


def main():
    """
    アプリケーションのメイン関数
    
    引数:
        なし
        
    戻り値:
        なし
    """
    app = QApplication(sys.argv)
    window = FaceCropApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
