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

        # --- 設定状態 --- #
        self.setting_mode = "batch"  # 'batch' or 'individual'
        # 一括モード時のデフォルト値
        self.selected_aspect_ratio = "16:9"
        self.selected_split_method = "phi"  # デフォルトを "phi" に変更
        # 個別モード時に画像ごとに保持するリスト
        self.individual_aspect_ratios = []
        self.individual_split_methods = []

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

        # --- 設定モード、分割法、アスペクト比選択エリア --- #
        settings_layout = QHBoxLayout()

        # --- 設定モード --- #
        mode_group = QGroupBox("設定モード")  # 名称変更
        mode_layout = QHBoxLayout()
        self.mode_button_group = QButtonGroup(self)
        self.batch_mode_radio = QRadioButton("一括")
        self.batch_mode_radio.setChecked(True)
        self.batch_mode_radio.setToolTip("一括モード: 全ての画像に同じ設定（分割法・アスペクト比）を適用します。")
        self.individual_mode_radio = QRadioButton("個別")
        self.individual_mode_radio.setToolTip("個別モード: 画像ごとに設定（分割法・アスペクト比）を保持・変更できます。")
        self.mode_button_group.addButton(self.batch_mode_radio)
        self.mode_button_group.addButton(self.individual_mode_radio)
        mode_layout.addWidget(self.batch_mode_radio)
        mode_layout.addWidget(self.individual_mode_radio)
        mode_group.setLayout(mode_layout)
        self.mode_button_group.buttonClicked.connect(self.on_setting_mode_changed)  # ハンドラ名変更
        settings_layout.addWidget(mode_group)

        # --- 分割法選択 --- #
        split_method_group = QGroupBox("分割法")
        split_method_layout = QHBoxLayout()
        self.split_method_button_group = QButtonGroup(self)

        # ファイグリッドを先に追加し、デフォルト選択にする
        self.phi_radio = QRadioButton("ファイグリッド")
        self.phi_radio.setObjectName("phi")
        self.phi_radio.setToolTip("黄金比(約1:1.618)に基づき画面を分割する構図法。三分割法より中央に線が寄ります。")
        self.phi_radio.setChecked(True)  # デフォルト選択
        self.split_method_button_group.addButton(self.phi_radio)
        split_method_layout.addWidget(self.phi_radio)  # 先に追加

        # 三分割法を後に追加
        self.thirds_radio = QRadioButton("三分割法")
        self.thirds_radio.setObjectName("thirds")
        self.thirds_radio.setToolTip("画面を縦横3分割し、線や交点に被写体を配置する基本的な構図法です。")
        self.split_method_button_group.addButton(self.thirds_radio)
        split_method_layout.addWidget(self.thirds_radio)  # 後に追加

        split_method_group.setLayout(split_method_layout)
        self.split_method_button_group.buttonClicked.connect(self.on_split_method_changed)
        settings_layout.addWidget(split_method_group)

        # --- アスペクト比選択 --- #
        aspect_ratio_group = QGroupBox("アスペクト比")
        aspect_ratio_layout = QHBoxLayout()
        self.aspect_ratio_button_group = QButtonGroup(self)

        self.aspect_ratio_buttons = {}
        ratios = ["16:9", "9:16", "4:3", "3:4"]
        for ratio in ratios:
            radio_button = QRadioButton(ratio)
            self.aspect_ratio_buttons[ratio] = radio_button
            self.aspect_ratio_button_group.addButton(radio_button)
            aspect_ratio_layout.addWidget(radio_button)
            if ratio == self.selected_aspect_ratio:  # デフォルト選択
                radio_button.setChecked(True)

        self.aspect_ratio_button_group.buttonClicked.connect(self.on_aspect_ratio_changed)
        aspect_ratio_group.setLayout(aspect_ratio_layout)
        settings_layout.addWidget(aspect_ratio_group)

        # レイアウトストレッチ: モード(1), 分割法(1), アスペクト比(2)
        settings_layout.setStretch(0, 1)
        settings_layout.setStretch(1, 1)
        settings_layout.setStretch(2, 2)

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
        self.batch_load_button = QPushButton("画像をまとめて読み込む")  # テキスト変更
        self.batch_load_button.clicked.connect(self.batch_load_images)
        self.batch_save_button = QPushButton("クロップ画像をまとめて保存")  # テキスト変更
        self.batch_save_button.clicked.connect(self.batch_save_images)
        self.batch_save_button.setEnabled(False)

        button_layout.addWidget(self.batch_load_button)
        button_layout.addStretch(1)
        button_layout.addLayout(center_nav_layout)
        button_layout.addStretch(1)
        button_layout.addWidget(self.batch_save_button)

        # レイアウトをメインウィジェットに追加
        main_layout.addLayout(image_layout)  # 画像表示
        main_layout.addLayout(settings_layout)  # 設定モード、分割法、アスペクト比
        main_layout.addLayout(button_layout)  # ボタンと情報ラベル
        main_layout.setSpacing(8)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def on_setting_mode_changed(self, button):
        new_mode = "batch" if button == self.batch_mode_radio else "individual"
        if new_mode == self.setting_mode:
            return

        self.setting_mode = new_mode
        if not self.original_images:
            return

        if new_mode == "individual":
            # 一括 -> 個別: 現在の一括設定を個別の初期値としてコピー
            num_images = len(self.original_images)
            # リストの長さが足りない場合のみ拡張・初期化
            if len(self.individual_aspect_ratios) < num_images:
                self.individual_aspect_ratios.extend(
                    [self.selected_aspect_ratio] *
                    (num_images - len(self.individual_aspect_ratios)))
            else:  # すでにリストが存在する場合は現在の一括設定で上書き
                self.individual_aspect_ratios = [self.selected_aspect_ratio] * num_images

            if len(self.individual_split_methods) < num_images:
                self.individual_split_methods.extend(
                    [self.selected_split_method] *
                    (num_images - len(self.individual_split_methods)))
            else:  # すでにリストが存在する場合は現在の一括設定で上書き
                self.individual_split_methods = [self.selected_split_method] * num_images
            # 個別モードに切り替えた直後は再計算不要（表示時に個別設定が反映されるため）
        else:  # individual -> batch
            # 個別 -> 一括: UI上の現在の選択を新しい一括設定とする
            current_split_button = self.split_method_button_group.checkedButton()
            current_aspect_button = self.aspect_ratio_button_group.checkedButton()
            if current_split_button:
                self.selected_split_method = current_split_button.objectName()
            if current_aspect_button:
                self.selected_aspect_ratio = current_aspect_button.text()
            # 一括モードに切り替えたら、全画像を新しい一括設定で再計算
            self._update_all_crop_results()

        self.show_current_image()  # モード切替後、UIの状態を正しく反映させる

    def on_split_method_changed(self, button):
        new_split = button.objectName()
        if self.setting_mode == "batch":
            if new_split == self.selected_split_method:
                return
            self.selected_split_method = new_split
            self._update_all_crop_results()
        else:  # individual モード
            if not self.original_images or not (0 <= self.current_index < len(
                    self.individual_split_methods)):
                return
            if new_split == self.individual_split_methods[self.current_index]:
                return
            self.individual_split_methods[self.current_index] = new_split
            self._update_crop_results(self.current_index)  # 現在の画像のみ更新
        self.show_current_image()  # UI同期と表示更新のため常に呼び出す

    def on_aspect_ratio_changed(self, button):
        new_ratio = button.text()
        if self.setting_mode == "batch":
            if new_ratio == self.selected_aspect_ratio:
                return
            self.selected_aspect_ratio = new_ratio
            self._update_all_crop_results()
        else:  # individual モード
            if not self.original_images or not (0 <= self.current_index < len(
                    self.individual_aspect_ratios)):
                return
            if new_ratio == self.individual_aspect_ratios[self.current_index]:
                return
            self.individual_aspect_ratios[self.current_index] = new_ratio
            self._update_crop_results(self.current_index)  # 現在の画像のみ更新
        self.show_current_image()  # UI同期と表示更新のため常に呼び出す

    def _get_current_settings(self, index=None):
        """指定されたインデックス（または現在）の設定を取得"""
        idx = self.current_index if index is None else index
        if self.setting_mode == "batch":
            return self.selected_aspect_ratio, self.selected_split_method
        else:
            aspect = self.selected_aspect_ratio  # デフォルト
            split = self.selected_split_method  # デフォルト
            if 0 <= idx < len(self.individual_aspect_ratios):
                aspect = self.individual_aspect_ratios[idx]
            if 0 <= idx < len(self.individual_split_methods):
                split = self.individual_split_methods[idx]
            return aspect, split

    def _update_crop_results(self, index):
        if not (0 <= index < len(self.original_images)):
            return
        img = self.original_images[index]
        aspect, split = self._get_current_settings(index)
        # crop_image に split_method を渡す
        result = self.image_processor.crop_image(img, aspect, split_method=split, debug_mode=True)

        if isinstance(result, dict):
            # デバッグ情報が含まれる場合
            self.cropped_images[index] = result.get('cropped_clean')  # 保存用画像
            # デバッグ描画済み画像を debug_images に格納
            self.debug_images[index] = {
                'original_with_faces': result.get('original_with_faces'),
                'cropped_with_grid': result.get('cropped_with_grid'),
                'split_method':
                    result.get('split_method')  # 分割法も保持
            }
        else:
            # クロップ画像のみの場合（エラーなど）
            self.cropped_images[index] = result
            self.debug_images[index] = None  # デバッグ情報なし

    def _update_all_crop_results(self):
        num_images = len(self.original_images)
        self.debug_images = [None] * num_images
        self.cropped_images = [None] * num_images

        # 個別モードへの移行時、または個別モードでの一括設定変更時にリストを初期化/更新
        if self.setting_mode == "individual":
            # リストの長さが足りなければデフォルト値で拡張
            if len(self.individual_aspect_ratios) < num_images:
                self.individual_aspect_ratios.extend(
                    [self.selected_aspect_ratio] *
                    (num_images - len(self.individual_aspect_ratios)))
            if len(self.individual_split_methods) < num_images:
                self.individual_split_methods.extend(
                    [self.selected_split_method] *
                    (num_images - len(self.individual_split_methods)))
        # 一括モードの場合は、個別リストは使用されないが、長さは維持しておく
        elif len(self.individual_aspect_ratios) != num_images:
            self.individual_aspect_ratios = [self.selected_aspect_ratio] * num_images
            self.individual_split_methods = [self.selected_split_method] * num_images

        for i in range(num_images):
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
            # 画像読み込みと初期クロップ処理
            for f in files:
                img = imread_unicode(f)
                if img is not None:
                    self.original_images.append(img)
                else:
                    QMessageBox.warning(self, "読込エラー",
                                        f"ファイル '{os.path.basename(f)}' を読み込めませんでした。スキップします。")

            if not self.original_images:
                QMessageBox.information(self, "情報", "読み込める画像がありませんでした。")
                return

            # 状態リセットと初期化
            self.current_index = 0
            num_images = len(self.original_images)
            # 個別設定リストも画像数に合わせて初期化（現在の一括設定で）
            self.individual_aspect_ratios = [self.selected_aspect_ratio] * num_images
            self.individual_split_methods = [self.selected_split_method] * num_images

            self._update_all_crop_results()  # 全画像のクロップ結果を計算
            self.show_current_image()  # 最初の画像を表示
            self.update_navigation_buttons()
            self.batch_save_button.setEnabled(True)
        else:
            # ファイル選択がキャンセルされた場合など
            pass  # 何もしない

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
        current_aspect, current_split = self._get_current_settings()

        # --- 元画像表示 --- #
        orig = self.original_images[idx]
        display_orig = orig  # デフォルト
        if isinstance(debug_info, dict) and 'original_with_faces' in debug_info:
            display_orig = debug_info['original_with_faces']  # 顔情報描画済み画像

        if display_orig is not None:
            self.display_cv_image(self.original_image_label, display_orig)
        else:
            self.original_image_label.setText("元画像表示エラー")

        # --- クロップ画像表示 --- #
        # cropped = self.cropped_images[idx] # 保存用クリーン画像は表示しない
        display_cropped = None  # デフォルト
        if isinstance(debug_info, dict) and 'cropped_with_grid' in debug_info:
            display_cropped = debug_info['cropped_with_grid']  # グリッド描画済み画像
        elif idx < len(self.cropped_images) and self.cropped_images[idx] is not None:
            # デバッグ情報がない場合（エラー時など）は、保存用画像を表示試行
            display_cropped = self.cropped_images[idx]

        if display_cropped is not None:
            self.display_cv_image(self.cropped_image_label, display_cropped)
        else:
            self.cropped_image_label.setText("クロップ画像表示エラー")

        # --- UI同期 --- #
        # 設定モードに関わらず、分割法とアスペクト比のラジオボタンは常に有効
        for button in self.split_method_button_group.buttons():
            button.setEnabled(True)  # 常に有効
        for button in self.aspect_ratio_button_group.buttons():
            button.setEnabled(True)  # 常に有効

        # 現在の設定を取得
        current_aspect, current_split = self._get_current_settings()

        # 分割法ラジオボタンの状態を更新
        split_blocked = self.split_method_button_group.signalsBlocked()
        self.split_method_button_group.blockSignals(True)
        if current_split == "thirds":
            self.thirds_radio.setChecked(True)
        elif current_split == "phi":
            self.phi_radio.setChecked(True)
        # 他にボタンが増えた場合も考慮 (念のため)
        else:
            for button in self.split_method_button_group.buttons():
                if button.objectName() == current_split:
                    button.setChecked(True)
                    break
        self.split_method_button_group.blockSignals(split_blocked)

        # アスペクト比ラジオボタンの状態を更新
        aspect_blocked = self.aspect_ratio_button_group.signalsBlocked()
        self.aspect_ratio_button_group.blockSignals(True)
        if current_aspect in self.aspect_ratio_buttons:
            self.aspect_ratio_buttons[current_aspect].setChecked(True)
        # 他にボタンが増えた場合も考慮 (念のため)
        else:
            for button in self.aspect_ratio_button_group.buttons():
                if button.text() == current_aspect:
                    button.setChecked(True)
                    break
        self.aspect_ratio_button_group.blockSignals(aspect_blocked)

        # --- 元画像表示 --- #
        # ...existing code...

        # --- クロップ画像表示 --- #
        # ...existing code...

        # 情報ラベル更新
        self.info_label.setText(f"{self.current_index + 1} / {len(self.original_images)}")
        self.update_navigation_buttons()

    def display_cv_image(self, label, cv_image):
        """QLabelにOpenCV画像を表示するヘルパー関数"""
        if cv_image is None:
            label.setText("画像表示エラー")
            return
        try:
            if len(cv_image.shape) == 3:  # カラー画像
                h, w, ch = cv_image.shape
                bytes_per_line = ch * w
                q_format = QImage.Format_BGR888
            elif len(cv_image.shape) == 2:  # グレースケール画像
                h, w = cv_image.shape
                bytes_per_line = w
                q_format = QImage.Format_Grayscale8
            else:
                label.setText("未対応の画像形式")
                return

            qt_image = QImage(cv_image.data, w, h, bytes_per_line, q_format)
            # QLabelのサイズに合わせてスケーリング
            pixmap = QPixmap.fromImage(qt_image).scaled(label.size(), Qt.KeepAspectRatio,
                                                        Qt.SmoothTransformation)
            label.setPixmap(pixmap)
        except Exception as e:
            print(f"画像表示エラー: {e}")
            label.setText("画像表示エラー")

    def show_next_image(self):
        """
        次の画像を表示
        """
        if self.current_index < len(self.original_images) - 1:
            self.current_index += 1
            self.show_current_image()

    def show_prev_image(self):
        """
        前の画像を表示
        """
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()

    def update_navigation_buttons(self):
        """
        ナビゲーションボタンの有効/無効を更新
        """
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.original_images) - 1)

    def batch_save_images(self):
        """
        一括でクロップ画像を保存
        """
        if not self.cropped_images:
            QMessageBox.warning(self, "保存エラー", "保存するクロップ画像がありません。")
            return

        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        save_dir = QFileDialog.getExistingDirectory(self, "クロップ画像の保存先フォルダを選択", "", options=options)

        if save_dir:
            saved_count = 0
            error_count = 0
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for i, cropped_img in enumerate(self.cropped_images):
                if cropped_img is None:
                    print(f"画像 {i+1} はクロップに失敗したためスキップします。")
                    error_count += 1
                    continue

                original_path = self.image_paths[i]
                base_name = os.path.splitext(os.path.basename(original_path))[0]
                extension = os.path.splitext(original_path)[1]
                # 設定に応じたファイル名を作成
                aspect, split = self._get_current_settings(i)
                aspect_str = aspect.replace(":", "-")
                split_str = split  # "thirds" or "phi"
                # ファイル名例: image01_cropped_16-9_thirds_20230101_120000.jpg
                # save_filename = f"{base_name}_cropped_{aspect_str}_{split_str}_{timestamp}{extension}"
                # シンプルなファイル名に変更
                save_filename = f"{base_name}_cropped_{aspect_str}_{split_str}{extension}"
                save_path = os.path.join(save_dir, save_filename)

                try:
                    # 日本語パス対応の保存
                    is_success, buffer = cv2.imencode(extension, cropped_img)
                    if is_success:
                        with open(save_path, 'wb') as f:
                            f.write(buffer)
                        saved_count += 1
                    else:
                        print(f"画像 {i+1} ('{save_filename}') のエンコードに失敗しました。")
                        error_count += 1
                except Exception as e:
                    print(f"画像 {i+1} ('{save_filename}') の保存中にエラーが発生しました: {e}")
                    error_count += 1

            # 保存結果のメッセージ
            if saved_count > 0:
                message = f"{saved_count} 件の画像をフォルダ '{os.path.basename(save_dir)}' に保存しました。"
                if error_count > 0:
                    message += f"\n{error_count} 件の画像の保存に失敗しました。"
                QMessageBox.information(self, "保存完了", message)
            else:
                QMessageBox.warning(self, "保存失敗", f"画像の保存に失敗しました ({error_count} 件のエラー)。")


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
