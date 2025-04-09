import os
import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
                             QMessageBox, QPushButton, QVBoxLayout, QWidget)


class FaceCropApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.title = '顔認識自動クロップツール'
        self.original_image = None
        self.cropped_image = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                                  'haarcascade_frontalface_default.xml')

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 1000, 600)

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
        self.crop_button.setEnabled(False)

        # 保存ボタン
        self.save_button = QPushButton('クロップ画像を保存')
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)

        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.crop_button)
        button_layout.addWidget(self.save_button)

        # レイアウトをメインウィジェットに追加
        main_layout.addLayout(image_layout)
        main_layout.addLayout(button_layout)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def load_image(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self,
                                                  "画像ファイルを開く",
                                                  "",
                                                  "画像ファイル (*.jpg *.jpeg *.png *.bmp);;すべてのファイル (*)",
                                                  options=options)

        if filepath:
            try:
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
                self.cropped_image_label.setText("クロップ後の画像がここに表示されます")
                self.save_button.setEnabled(False)

            except Exception as e:
                QMessageBox.critical(self, "エラー", f"画像の読み込みエラー: {str(e)}")

    def crop_image(self):
        if self.original_image is None:
            return

        try:
            # グレースケールに変換
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

            # 顔検出
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                QMessageBox.warning(self, "警告", "画像から顔を検出できませんでした")
                return

            # 最も大きい顔を使用（複数ある場合）
            # 顔の大きさ（幅×高さ）で並べ替え
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]

            # 顔の中心を計算
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            # 画像の高さと幅を取得
            img_height, img_width = self.original_image.shape[:2]

            # 16:9のアスペクト比で計算（高さを基準にした場合の幅）
            target_width = int(img_height * 16 / 9)

            # 元画像の幅が計算した幅よりも小さい場合は、幅を基準にして高さを計算
            if img_width < target_width:
                target_height = int(img_width * 9 / 16)
                crop_width = img_width
                crop_height = target_height
            else:
                crop_width = target_width
                crop_height = img_height

            # 顔の中心を基に切り抜き範囲を決定
            start_x = max(0, face_center_x - crop_width // 2)
            end_x = start_x + crop_width

            # x座標がはみ出す場合の調整
            if end_x > img_width:
                end_x = img_width
                start_x = max(0, end_x - crop_width)

            # y座標（中央に顔がくるよう調整）
            start_y = max(0, face_center_y - crop_height // 2)
            end_y = start_y + crop_height

            # y座標がはみ出す場合の調整
            if end_y > img_height:
                end_y = img_height
                start_y = max(0, end_y - crop_height)

            # 画像をクロップ
            self.cropped_image = self.original_image[start_y:end_y, start_x:end_x].copy()

            # クロップした画像をQPixmapに変換して表示
            height, width, channels = self.cropped_image.shape
            bytes_per_line = channels * width
            q_img = QImage(self.cropped_image.data, width, height, bytes_per_line,
                           QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)

            # スケーリング
            scaled_pixmap = pixmap.scaled(self.cropped_image_label.width(),
                                          self.cropped_image_label.height(), Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)
            self.cropped_image_label.setPixmap(scaled_pixmap)

            # 保存ボタンを有効に
            self.save_button.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "エラー", f"クロップ処理エラー: {str(e)}")

    def save_image(self):
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
                    filepath += '.jpg'

                # 画像を保存
                cv2.imwrite(filepath, self.cropped_image)
                QMessageBox.information(self, "成功", f"画像を保存しました: {filepath}")

            except Exception as e:
                QMessageBox.critical(self, "エラー", f"画像の保存エラー: {str(e)}")


def main():
    app = QApplication(sys.argv)
    window = FaceCropApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
