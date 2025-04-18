import os
import sys
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QButtonGroup, QFileDialog, QFrame, QGroupBox,
                             QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
                             QRadioButton, QSizePolicy, QVBoxLayout, QWidget)

# ImageProcessorクラスのインポート
from image_processor import ImageProcessor


def imread_unicode(path):
    """日本語パス対応のOpenCV画像読込"""
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except (IOError, cv2.error):
        # Handle file reading or OpenCV decoding errors
        return None


class FaceCropApp(QMainWindow):
    """
    顔認識自動クロップツールのGUIアプリケーションクラス
    
    顔認識を使用して画像をクロップし、選択されたアスペクト比に調整するアプリケーション。
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

        self.title = "顔認識自動クロップツール"
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

        # アスペクト比の状態
        self.aspect_ratio_mode = "batch"  # 'batch' or 'individual'
        self.selected_aspect_ratio = "16:9"  # 一括モード時の値
        self.individual_aspect_ratios = []  # 個別モード時に画像ごとに保持するリスト

        self.init_ui()

    def init_ui(self):
        """
        UIコンポーネントの初期化と配置を行うメソッド
        """
        self.setWindowTitle(self.title)

        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # --- 画像表示エリア --- (変更なし)
        image_layout = QHBoxLayout()
        self.original_image_label = QLabel()
        self.original_image_label.setFrameStyle(QFrame.Box)
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setText("元画像がここに表示されます")
        self.original_image_label.setMinimumSize(400, 400)
        self.original_image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.cropped_image_label = QLabel()
        self.cropped_image_label.setFrameStyle(QFrame.Box)
        self.cropped_image_label.setAlignment(Qt.AlignCenter)
        self.cropped_image_label.setText("クロップ後の画像がここに表示されます")
        self.cropped_image_label.setMinimumSize(400, 225)  # 16:9比率の最小サイズ例
        self.cropped_image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        image_layout.addWidget(self.original_image_label)
        image_layout.addWidget(self.cropped_image_label)

        # --- アスペクト比モード選択 ---
        settings_layout = QVBoxLayout()
        mode_group = QGroupBox("アスペクト比モード")
        mode_layout = QVBoxLayout()
        self.mode_button_group = QButtonGroup(self)
        self.batch_mode_radio = QRadioButton("一括")
        self.batch_mode_radio.setChecked(True)
        self.individual_mode_radio = QRadioButton("個別")
        self.mode_button_group.addButton(self.batch_mode_radio)
        self.mode_button_group.addButton(self.individual_mode_radio)
        mode_layout.addWidget(self.batch_mode_radio)
        mode_layout.addWidget(self.individual_mode_radio)
        mode_group.setLayout(mode_layout)
        self.mode_button_group.buttonClicked.connect(self.on_mode_changed)
        settings_layout.addWidget(mode_group)

        # --- アスペクト比選択ラジオボタン ---
        aspect_ratio_group = QGroupBox("アスペクト比")
        aspect_ratio_layout = QHBoxLayout()
        self.aspect_ratio_button_group = QButtonGroup(self)

        self.aspect_ratio_buttons = {}
        ratios = ["16:9", "9:16", "4:3", "3:4"]
        for ratio in ratios:
            radio_button = QRadioButton(ratio)
            self.aspect_ratio_button_group.addButton(radio_button)
            aspect_ratio_layout.addWidget(radio_button)
            self.aspect_ratio_buttons[ratio] = radio_button
            if ratio == self.selected_aspect_ratio:
                radio_button.setChecked(True)

        self.aspect_ratio_button_group.buttonClicked.connect(self.on_aspect_ratio_changed)
        aspect_ratio_group.setLayout(aspect_ratio_layout)
        settings_layout.addWidget(aspect_ratio_group)

        # --- ボタンと情報ラベルのレイアウト ---
        # ナビゲーションボタンと情報ラベル用の中央レイアウト
        center_nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("前へ")
        self.prev_button.clicked.connect(self.show_prev_image)
        self.prev_button.setEnabled(False)
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.next_button = QPushButton("次へ")
        self.next_button.clicked.connect(self.show_next_image)
        self.next_button.setEnabled(False)

        center_nav_layout.addWidget(self.prev_button)
        center_nav_layout.addSpacing(10)
        center_nav_layout.addWidget(self.info_label)
        center_nav_layout.addSpacing(10)
        center_nav_layout.addWidget(self.next_button)

        # 全体のボタンレイアウト
        button_layout = QHBoxLayout()
        self.batch_load_button = QPushButton("一括で画像を読み込む")
        self.batch_load_button.clicked.connect(self.batch_load_images)
        self.batch_save_button = QPushButton("一括でクロップ画像を保存")
        self.batch_save_button.clicked.connect(self.batch_save_images)
        self.batch_save_button.setEnabled(False)

        button_layout.addWidget(self.batch_load_button)
        button_layout.addStretch(1)
        button_layout.addLayout(center_nav_layout)
        button_layout.addStretch(1)
        button_layout.addWidget(self.batch_save_button)

        # レイアウトをメインウィジェットに追加
        main_layout.addLayout(image_layout)  # 画像表示
        main_layout.addLayout(settings_layout)  # アスペクト比モードと選択
        main_layout.addLayout(button_layout)  # ボタンと情報ラベル
        main_layout.setSpacing(8)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def on_mode_changed(self, button):
        new_mode = "batch" if button == self.batch_mode_radio else "individual"
        if new_mode == self.aspect_ratio_mode:
            return
        self.aspect_ratio_mode = new_mode
        if not self.original_images:
            return
        if new_mode == "individual":
            # 一括->個別:全画像の個別設定を現在の一括値で初期化
            self.individual_aspect_ratios = [self.selected_aspect_ratio] * len(self.original_images)
        else:
            # 個別->一括:個別設定をクリア
            self.individual_aspect_ratios.clear()
        self._update_all_crop_results()
        self.show_current_image()

    def on_aspect_ratio_changed(self, button):
        new_ratio = button.text()
        if self.aspect_ratio_mode == "batch":
            self.selected_aspect_ratio = new_ratio
            if self.original_images:
                self._update_all_crop_results()
                self.show_current_image()
        else:
            idx = self.current_index
            if 0 <= idx < len(self.individual_aspect_ratios):
                self.individual_aspect_ratios[idx] = new_ratio
                self._update_crop_results(idx)
                self.show_current_image()
            else:
                self.selected_aspect_ratio = new_ratio

    def _get_current_aspect_ratio(self, index=None):
        if self.aspect_ratio_mode == "batch":
            return self.selected_aspect_ratio
        idx = self.current_index if index is None else index
        if 0 <= idx < len(self.individual_aspect_ratios):
            return self.individual_aspect_ratios[idx]
        return self.selected_aspect_ratio

    def _update_crop_results(self, index):
        if not (0 <= index < len(self.original_images)):
            return
        img = self.original_images[index]
        aspect = self._get_current_aspect_ratio(index)
        result = self.image_processor.crop_image(img, aspect, debug_mode=True)
        if isinstance(result, dict):
            self.debug_images[index] = result
            self.cropped_images[index] = result.get("cropped_clean")
        else:
            self.debug_images[index] = None
            self.cropped_images[index] = None

    def _update_all_crop_results(self):
        self.debug_images = [None] * len(self.original_images)
        self.cropped_images = [None] * len(self.original_images)
        self.individual_aspect_ratios = ([self.selected_aspect_ratio] *
                                         len(self.original_images) if self.aspect_ratio_mode
                                         == "individual" else self.individual_aspect_ratios)
        for i in range(len(self.original_images)):
            self._update_crop_results(i)

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
            self.debug_images = []
            self.current_index = 0
            failed_files = []
            for path in self.image_paths:
                img = imread_unicode(path)
                if img is not None:
                    self.original_images.append(img)
                    # 現在選択されているアスペクト比でクロップ
                    debug_result = self.image_processor.crop_image(img,
                                                                   self.selected_aspect_ratio,
                                                                   debug_mode=True)
                    if isinstance(debug_result, dict):
                        self.debug_images.append(debug_result)
                        self.cropped_images.append(debug_result.get("cropped_clean", None))
                    else:
                        self.debug_images.append(None)
                        self.cropped_images.append(None)
                else:
                    failed_files.append(path)
            if len(self.original_images) == 0:
                QMessageBox.warning(self, "警告", "画像の読み込みに失敗しました")
                return
            self.info_label.setText(f"{len(self.original_images)}枚の画像を読み込みました")
            self._update_all_crop_results()
            self.show_current_image()
            self.update_navigation_buttons()
            self.batch_save_button.setEnabled(True)
        else:
            self.info_label.setText("画像が選択されませんでした")

    def show_current_image(self):
        """
        現在のインデックスの画像を表示
        """
        if not self.original_images or self.current_index >= len(self.original_images):
            self.original_image_label.setText("元画像なし")
            self.cropped_image_label.setText("クロップ画像なし")
            self.info_label.setText("0/0")
            return

        idx = self.current_index
        debug_info = self.debug_images[idx] if idx < len(self.debug_images) else None

        # --- 元画像表示 ---
        orig = self.original_images[idx]
        display_orig = orig  # デフォルト
        if isinstance(debug_info, dict):
            orig_with_faces = debug_info.get("original_with_faces")
            if orig_with_faces is not None:
                display_orig = orig_with_faces

        if display_orig is not None:
            self.display_cv_image(self.original_image_label, display_orig)
        else:
            self.original_image_label.setText("元画像エラー")

        # --- クロップ画像表示 ---
        cropped = self.cropped_images[idx]  # 保存用クリーン画像
        display_cropped = cropped  # デフォルト
        if isinstance(debug_info, dict):
            cropped_with_grid = debug_info.get("cropped_with_grid")
            if cropped_with_grid is not None:
                display_cropped = cropped_with_grid

        if display_cropped is not None:
            self.display_cv_image(self.cropped_image_label, display_cropped)
            self.info_label.setText(f"{idx+1}/{len(self.original_images)}")
        else:
            self.cropped_image_label.setText("クロップ失敗")
            self.info_label.setText(f"{idx+1}/{len(self.original_images)} (クロップ失敗)")

        # アスペクト比ボタン状態同期
        current = self._get_current_aspect_ratio()
        blocked = self.aspect_ratio_button_group.signalsBlocked()
        self.aspect_ratio_button_group.blockSignals(True)
        if current in self.aspect_ratio_buttons:
            self.aspect_ratio_buttons[current].setChecked(True)
        self.aspect_ratio_button_group.blockSignals(blocked)

    def display_cv_image(self, label, cv_image):
        """QLabelにOpenCV画像を表示するヘルパー関数"""
        if cv_image is None:
            label.setText("画像表示エラー")
            return
        try:
            height, width, channels = cv_image.shape
            bytes_per_line = channels * width
            q_img = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)
            if pixmap.isNull():
                label.setText("Pixmap変換エラー")
            else:
                # ラベルのサイズに合わせてスケーリング
                scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation)
                label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"画像表示エラー: {e}")
            label.setText(f"表示エラー: {e}")

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
        prefix = now.strftime("%Y%m%d_%H%M%S")  # より詳細なプレフィックス
        count = 1
        saved_count = 0
        failed_paths = []

        for i, img in enumerate(self.cropped_images):
            if img is not None:
                # 元画像のパスからファイル名を取得
                original_path = self.image_paths[i]
                base_name, _ = os.path.splitext(os.path.basename(original_path))
                # 新しいファイル名を作成 (プレフィックス_元の名前_連番.jpg)
                filename = f"{prefix}_{base_name}_{count}.jpg"  # 保存はJPG形式に統一
                path = os.path.join(folder, filename)

                # 日本語パス対応で保存
                try:
                    # 保存する画像はデバッグ情報なしのクリーンなもの (self.cropped_images[i])
                    ret, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])  # 高画質で保存
                    if ret:
                        buf.tofile(path)
                        saved_count += 1
                    else:
                        raise RuntimeError(f"画像エンコード失敗: {path}")
                except Exception as e:
                    print(f"画像の保存に失敗しました: {path}\n{str(e)}")
                    failed_paths.append(path)
                count += 1
            else:
                print(f"インデックス {i} の画像はクロップに失敗したため保存されません。")
                count += 1  # スキップしても連番は進める

        if failed_paths:
            QMessageBox.warning(self, "警告", f"{len(failed_paths)}件の画像の保存に失敗しました。詳細はコンソールを確認してください。")
        else:
            QMessageBox.information(self, "完了", f"{saved_count}枚の画像のクロップと保存が完了しました。")


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


if __name__ == "__main__":
    main()
