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
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        # デバッグモードの追加（三分割線を表示するかどうか）
        self.debug_mode = True

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

            # 元画像のサイズを取得
            img_height, img_width = self.original_image.shape[:2]

            # 顔領域と目を検出
            face_roi = gray[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(face_roi)

            # 顔の中心を計算
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            # 目の位置を計算（検出された場合）
            eyes_y = face_center_y
            if len(eyes) >= 2:
                # 目の位置（縦）の平均を計算
                eye_y_positions = []
                for ex, ey, ew, eh in eyes:
                    eye_center_y = y + ey + eh // 2
                    eye_y_positions.append(eye_center_y)

                # 目の平均位置を使用
                eyes_y = sum(eye_y_positions) // len(eye_y_positions)

            # 16:9のアスペクト比で計算
            target_aspect_ratio = 16 / 9

            # 最終的なクロップの高さと幅を決定
            crop_height = min(img_height, int(img_width / target_aspect_ratio))
            crop_width = min(img_width, int(crop_height * target_aspect_ratio))

            # 三分割法の上側のライン位置を計算（クロップ後の上から1/3の位置）
            top_third_line = crop_height // 3

            # 顔または目が上側の横ラインに来るように調整
            # 目の位置が検出された場合は目を使用、そうでなければ顔の中心を使用
            target_y = eyes_y if len(eyes) >= 2 else face_center_y

            # クロップのトップ位置を計算 (target_yが上から1/3の位置に来るように)
            crop_top = max(0, target_y - top_third_line)

            # もし画像の下側がはみ出る場合は調整
            if crop_top + crop_height > img_height:
                crop_top = max(0, img_height - crop_height)

            # 横方向の中心を顔の中心に合わせる
            crop_left = max(0, face_center_x - crop_width // 2)

            # もし画像の右側がはみ出る場合は調整
            if crop_left + crop_width > img_width:
                crop_left = max(0, img_width - crop_width)

            # 横方向の三分割ラインを計算
            grid_left = crop_width // 3
            grid_right = (crop_width * 2) // 3

            # 顔の中心を横方向の近い方の三分割ラインに合わせる調整
            # 顔の中心位置（クロップ画像内の相対位置）
            face_rel_x = face_center_x - crop_left

            # どちらのグリッドラインに近いか判定
            if abs(face_rel_x - grid_left) < abs(face_rel_x - grid_right):
                # 左の線に近い場合、左の線に合わせる
                new_crop_left = max(0, face_center_x - grid_left)
            else:
                # 右の線に近い場合、右の線に合わせる
                new_crop_left = max(0, face_center_x - grid_right)

            # 境界チェック
            if new_crop_left + crop_width > img_width:
                new_crop_left = max(0, img_width - crop_width)

            # 調整後の左位置を適用
            crop_left = new_crop_left

            # 最終的なクロップ範囲を設定
            crop_right = min(crop_left + crop_width, img_width)
            crop_bottom = min(crop_top + crop_height, img_height)

            # この範囲で画像をクロップ
            self.cropped_image = self.original_image[int(crop_top):int(crop_bottom),
                                                     int(crop_left):int(crop_right)].copy()

            # クロップした画像内で顔を再検出
            cropped_gray = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2GRAY)
            cropped_faces = self.face_cascade.detectMultiScale(cropped_gray, 1.3, 5)

            # デバッグ用：三分割のグリッドと顔矩形を表示
            display_image = self.cropped_image.copy()
            h, w = display_image.shape[:2]

            # 縦線
            cv2.line(display_image, (w // 3, 0), (w // 3, h), (0, 255, 0), 1)
            cv2.line(display_image, (2 * w // 3, 0), (2 * w // 3, h), (0, 255, 0), 1)
            # 横線
            cv2.line(display_image, (0, h // 3), (w, h // 3), (0, 255, 0), 1)
            cv2.line(display_image, (0, 2 * h // 3), (w, 2 * h // 3), (0, 255, 0), 1)

            # クロップした画像内のすべての顔に矩形を描画
            # - 元の画像で最も大きかった顔は赤色(0,0,255)で表示
            # - その他の顔は灰色(128,128,128)で表示
            main_face_rel_x = face_center_x - crop_left
            main_face_rel_y = face_center_y - crop_top

            # 最初に赤い点を描画（基準となった顔の中心）
            if 0 <= main_face_rel_x < w and 0 <= main_face_rel_y < h:
                cv2.circle(display_image, (int(main_face_rel_x), int(main_face_rel_y)), 5,
                           (0, 0, 255), -1)

            # クロップ画像内で検出されたすべての顔に矩形を描画
            if len(cropped_faces) > 0:
                for i, (fx, fy, fw, fh) in enumerate(cropped_faces):
                    # 各顔の中心座標を計算
                    face_cx = fx + fw // 2
                    face_cy = fy + fh // 2

                    # メイン顔との距離を計算
                    distance = ((face_cx - main_face_rel_x)**2 +
                                (face_cy - main_face_rel_y)**2)**0.5

                    # 基準となった顔に近い顔は赤、それ以外は灰色
                    if distance < max(fw, fh) * 0.5:  # 顔の幅または高さの半分以内の距離なら同一人物と判定
                        cv2.rectangle(display_image, (fx, fy), (fx + fw, fy + fh), (0, 0, 255),
                                      2)  # 赤色
                    else:
                        cv2.rectangle(display_image, (fx, fy), (fx + fw, fy + fh), (128, 128, 128),
                                      2)  # 灰色

            # クロップした画像をQPixmapに変換して表示
            height, width, channels = display_image.shape
            bytes_per_line = channels * width
            q_img = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)

            # スケーリング
            scaled_pixmap = pixmap.scaled(self.cropped_image_label.width(),
                                          self.cropped_image_label.height(), Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)
            self.cropped_image_label.setPixmap(scaled_pixmap)

            # 保存ボタンを有効に
            self.save_button.setEnabled(True)

        except Exception as e:
            import traceback
            error_message = f"クロップ処理エラー: {str(e)}\n{traceback.format_exc()}"
            QMessageBox.critical(self, "エラー", error_message)

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
