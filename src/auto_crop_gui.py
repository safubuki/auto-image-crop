import os
import sys
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
                             QMessageBox, QPushButton, QVBoxLayout, QWidget)

# ImageProcessorクラスのインポート
from image_processor import ImageProcessor


def imread_unicode(path):
    """日本語パス対応のOpenCV画像読込"""
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


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

        # 一括処理用の状態変数
        self.image_paths = []
        self.original_images = []
        self.cropped_images = []
        self.debug_images = []  # 追加: デバッグ用画像セット
        self.current_index = 0

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
        self.setGeometry(100, 100, 900, 520)

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
        self.cropped_image_label.setMinimumSize(400, 225)

        image_layout.addWidget(self.original_image_label)
        image_layout.addWidget(self.cropped_image_label)

        # --- ボタンレイアウト (読み込み、ナビゲーション、保存を同一ラインに) ---
        button_layout = QHBoxLayout()
        self.batch_load_button = QPushButton('一括で画像を読み込む')
        self.batch_load_button.clicked.connect(self.batch_load_images)
        self.prev_button = QPushButton('前へ')
        self.prev_button.clicked.connect(self.show_prev_image)
        self.prev_button.setEnabled(False)
        self.next_button = QPushButton('次へ')
        self.next_button.clicked.connect(self.show_next_image)
        self.next_button.setEnabled(False)
        self.batch_save_button = QPushButton('一括でクロップ画像を保存')
        self.batch_save_button.clicked.connect(self.batch_save_images)
        self.batch_save_button.setEnabled(False)

        button_layout.addWidget(self.batch_load_button)
        button_layout.addStretch(1)  # スペーサー
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addStretch(1)  # スペーサー
        button_layout.addWidget(self.batch_save_button)

        # 情報表示用ラベル
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)

        # レイアウトをメインウィジェットに追加
        main_layout.addLayout(image_layout)  # 画像表示
        main_layout.addLayout(button_layout)  # 統合されたボタンレイアウト
        main_layout.addWidget(self.info_label)  # 情報ラベル
        main_layout.setSpacing(8)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def batch_load_images(self):
        """
        一括で画像を選択して読み込む
        """
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self,
                                                "画像ファイルを一括で開く",
                                                "",
                                                "画像ファイル (*.jpg *.jpeg *.png *.bmp);;すべてのファイル (*)",
                                                options=options)
        if files:
            self.image_paths = files
            self.original_images = []
            self.cropped_images = []
            self.debug_images = []  # 追加: デバッグ用画像セット
            self.current_index = 0
            failed_files = []
            for path in self.image_paths:
                img = imread_unicode(path)
                if img is not None:
                    self.original_images.append(img)
                    # デバッグ情報付きでクロップ
                    debug_result = self.image_processor.crop_image(img, debug_mode=True)
                    if isinstance(debug_result, dict):
                        # デバッグ画像セット
                        self.debug_images.append(debug_result)
                        # クロップ画像本体
                        cropped = debug_result.get('cropped_with_grid', None)
                        self.cropped_images.append(cropped)
                    else:
                        # 万一dictでなければ通常通り
                        self.debug_images.append(None)
                        self.cropped_images.append(debug_result)
                else:
                    failed_files.append(path)
            if len(self.original_images) == 0:
                QMessageBox.warning(self, "警告", "画像の読み込みに失敗しました")
                return
            self.info_label.setText(f"{len(self.original_images)}枚の画像を読み込みました")
            self.show_current_image()
            self.update_navigation_buttons()
            self.batch_save_button.setEnabled(True)
        else:
            self.info_label.setText("画像が選択されませんでした")

    def show_current_image(self):
        """
        現在のインデックスの画像を表示
        """
        if not self.original_images or not self.cropped_images:
            return
        idx = self.current_index
        debug = self.debug_images[idx] if hasattr(self,
                                                  "debug_images") and self.debug_images else None

        # 元画像表示
        if debug and 'original_with_faces' in debug:
            orig = debug['original_with_faces']
        else:
            orig = self.original_images[idx]
        # QPixmapに変換し、アスペクト比を維持してスケーリング
        height, width, channels = orig.shape
        bytes_per_line = channels * width
        q_img = QImage(orig.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.original_image_label.size(), Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)
        self.original_image_label.setPixmap(scaled_pixmap)

        # クロップ画像表示
        if debug and 'cropped_with_grid' in debug:
            cropped = debug['cropped_with_grid']
        else:
            cropped = self.cropped_images[idx]
        if cropped is not None:
            # QPixmapに変換し、アスペクト比を維持してスケーリング
            height, width, channels = cropped.shape
            bytes_per_line = channels * width
            q_img = QImage(cropped.data, width, height, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.cropped_image_label.size(), Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)
            self.cropped_image_label.setPixmap(scaled_pixmap)
            self.info_label.setText(f"{idx+1}/{len(self.cropped_images)}")
        else:
            self.cropped_image_label.setText("クロップ失敗")
            self.info_label.setText(f"{idx+1}/{len(self.cropped_images)} (クロップ失敗)")

    def show_next_image(self):
        """
        次の画像を表示
        """
        if self.current_index < len(self.original_images) - 1:
            self.current_index += 1
            self.show_current_image()
            self.update_navigation_buttons()

    def show_prev_image(self):
        """
        前の画像を表示
        """
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
            self.update_navigation_buttons()

    def update_navigation_buttons(self):
        """
        ナビゲーションボタンの有効/無効を更新
        """
        total = len(self.original_images)
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < total - 1)
        self.batch_save_button.setEnabled(total > 0)

    def batch_save_images(self):
        """
        一括でクロップ画像を保存
        """
        if not self.cropped_images:
            QMessageBox.warning(self, "警告", "保存する画像がありません")
            return
        folder = QFileDialog.getExistingDirectory(self, "保存先フォルダを選択")
        if not folder:
            return
        now = datetime.now()
        prefix = now.strftime("%H%M%S")
        count = 1
        for i, img in enumerate(self.cropped_images):
            if img is not None:
                # 元画像から再度クロップ（debug_mode=False）で保存
                orig_img = self.original_images[i]
                cropped_clean = self.image_processor.crop_image(orig_img, debug_mode=False)
                filename = f"{prefix}_{count}.jpg"
                path = os.path.join(folder, filename)
                # 日本語パス対応で保存
                try:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in [".jpg", ".jpeg"]:
                        ret, buf = cv2.imencode('.jpg', cropped_clean)
                    elif ext == ".png":
                        ret, buf = cv2.imencode('.png', cropped_clean)
                    else:
                        ret, buf = cv2.imencode('.jpg', cropped_clean)
                    if ret:
                        buf.tofile(path)
                    else:
                        raise Exception("画像エンコード失敗")
                except Exception as e:
                    QMessageBox.warning(self, "警告", f"画像の保存に失敗しました: {path}\n{str(e)}")
                count += 1
        # 標準のQMessageBoxではボタンの厳密な中央配置は保証されません。
        # 通常、単一ボタンの場合は中央に表示されます。
        QMessageBox.information(self, "完了", "全てのクロップが完了しました")


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
