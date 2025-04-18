import traceback

import cv2
import numpy as np

from face_detector import FaceDetector
from visualization import Visualizer


class ImageCropper:
    """
    顔検出に基づいて画像をクロップするクラス
    
    検出された顔の位置を元に、三分割法に基づいて画像を最適な位置でクロップする
    機能を提供します。主に16:9のアスペクト比でのクロップに対応しています。
    """

    def __init__(self):
        """
        ImageCropperクラスの初期化メソッド
        
        引数:
            なし
            
        戻り値:
            なし
        """
        self.face_detector = FaceDetector()

        # スコアリング用の重み付け係数（各要素の重要度）
        self.weight_size = 0.4  # 顔サイズの重み（大きい顔を優先）
        self.weight_center = 0.4  # 中央度の重み（中央に近い顔を優先）
        self.weight_sharpness = 0.2  # 鮮明度の重み（鮮明な顔を優先）

        # 最後に評価した顔のスコア情報
        self.last_scored_faces = []

    def select_best_face(self, image, faces):
        """
        複数の顔から最適な顔を選択するメソッド
        
        「最も大きい顔」「中央に近い顔」「顔領域の鮮明度」の3つの要素を
        スコアリングして最も適切な顔を選択します。
        
        引数:
            image: 入力画像（OpenCV形式のndarray）
            faces: 検出された顔のリスト [(x, y, w, h), ...]
            
        戻り値:
            最適な顔 (x, y, w, h)
        """
        if len(faces) == 0:
            return None

        if len(faces) == 1:
            # 顔が1つしかない場合は選択の必要なし
            self.last_scored_faces = [{
                'face': faces[0],
                'score': 1.0,
                'size_score': 1.0,
                'center_score': 1.0,
                'sharpness_score': 1.0
            }]
            return faces[0]

        # 画像の中心座標を計算
        img_height, img_width = image.shape[:2]
        img_center_x = img_width // 2
        img_center_y = img_height // 2

        # 最大のサイズ、最大距離、最大鮮明度を求めるための初期値
        max_size = 0
        max_dist = 0
        max_sharpness = 0.1  # ゼロ除算を避けるため小さな値を初期値とする

        # 各顔について評価値を計算するための情報を収集
        face_metrics = []

        for face in faces:
            x, y, w, h = face

            # 1. サイズ評価（顔の面積）
            size = w * h
            max_size = max(max_size, size)

            # 2. 中央からの距離評価（距離が近いほど良い）
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            dist_from_center = np.sqrt((face_center_x - img_center_x)**2 +
                                       (face_center_y - img_center_y)**2)
            max_dist = max(max_dist, dist_from_center)

            # 3. 鮮明度評価
            sharpness = self.face_detector.evaluate_face_sharpness(image, face)
            max_sharpness = max(max_sharpness, sharpness)

            face_metrics.append({
                'face': face,
                'size': size,
                'dist_from_center': dist_from_center,
                'sharpness': sharpness
            })

        # 各顔のスコアを正規化して計算
        scored_faces = []
        for metrics in face_metrics:
            # 正規化（各メトリクスを0～1の範囲に変換）
            norm_size = metrics['size'] / max_size  # 大きいほど良い
            norm_center = 1.0 - (metrics['dist_from_center'] / max_dist)  # 中央に近いほど良い
            norm_sharpness = metrics['sharpness'] / max_sharpness  # 鮮明なほど良い

            # 重み付けして合計スコアを計算
            total_score = (self.weight_size * norm_size + self.weight_center * norm_center +
                           self.weight_sharpness * norm_sharpness)

            scored_faces.append({
                'face': metrics['face'],
                'score': total_score,
                'size_score': norm_size,
                'center_score': norm_center,
                'sharpness_score': norm_sharpness
            })

        # スコアの降順でソート
        scored_faces.sort(key=lambda x: x['score'], reverse=True)

        # スコア情報を保存（可視化用）
        self.last_scored_faces = scored_faces

        # 最もスコアの高い顔を返す
        return scored_faces[0]['face']

    def crop_image(self,
                   original_image,
                   aspect_ratio_str='16:9',
                   split_method='thirds',
                   debug_mode=False):
        """
        顔検出に基づいて画像をクロップするメソッド

        検出された顔の位置と目の位置を基準に、三分割法またはファイグリッドを適用して
        最適な位置で画像をクロップします。デバッグモードでは
        グリッド線と検出結果を可視化します。

        引数:
            original_image: 入力画像（OpenCV形式のndarray）
            aspect_ratio_str: 目標のアスペクト比 ('16:9', '9:16', '4:3', '3:4')
            split_method: 分割方法 ('thirds' または 'phi')
            debug_mode: デバッグモードフラグ（True=可視化情報を含む画像を返す）

        戻り値:
            クロップされた画像または、デバッグモードの場合は可視化された画像
            顔が検出できなかった場合はNoneを返す
        """
        try:
            if original_image is None:
                return None

            # 顔検出
            faces = self.face_detector.detect_faces(original_image)
            if len(faces) == 0:
                # 顔がない場合は中央クロップを試みる
                print("顔が検出されませんでした。中央クロップを試みます。")
                h_orig, w_orig = original_image.shape[:2]
                try:
                    w_ratio, h_ratio = map(int, aspect_ratio_str.split(':'))
                    target_aspect = w_ratio / h_ratio
                except ValueError:
                    target_aspect = 16 / 9  # デフォルト

                target_w, target_h = self._calculate_target_dimensions(
                    w_orig, h_orig, target_aspect)
                crop_left = max(0, (w_orig - target_w) // 2)
                crop_top = max(0, (h_orig - target_h) // 2)
                cropped_image = original_image[crop_top:min(crop_top + target_h, h_orig),
                                               crop_left:min(crop_left + target_w, w_orig)]
                if debug_mode:
                    return {
                        'original_with_faces': original_image.copy(),  # 元画像そのまま
                        'cropped_with_grid': cropped_image.copy(),  # グリッドなし
                        'cropped_clean': cropped_image.copy(),
                        'split_method': split_method
                    }
                else:
                    return cropped_image

            # 複合的な評価に基づいて最適な顔を選択
            best_face = self.select_best_face(original_image, faces)
            if best_face is None:
                print("最適な顔が見つかりませんでした。")
                # 顔はあるが最適なものがない場合も中央クロップ
                h_orig, w_orig = original_image.shape[:2]
                try:
                    w_ratio, h_ratio = map(int, aspect_ratio_str.split(':'))
                    target_aspect = w_ratio / h_ratio
                except ValueError:
                    target_aspect = 16 / 9  # デフォルト

                target_w, target_h = self._calculate_target_dimensions(
                    w_orig, h_orig, target_aspect)
                crop_left = max(0, (w_orig - target_w) // 2)
                crop_top = max(0, (h_orig - target_h) // 2)
                cropped_image = original_image[crop_top:min(crop_top + target_h, h_orig),
                                               crop_left:min(crop_left + target_w, w_orig)]
                if debug_mode:
                    # 元画像には検出された顔を描画
                    visualizer = Visualizer()
                    original_with_faces = visualizer.draw_face_info(original_image.copy(), faces,
                                                                    self.last_scored_faces)
                    return {
                        'original_with_faces': original_with_faces,
                        'cropped_with_grid': cropped_image.copy(),  # グリッドなし
                        'cropped_clean': cropped_image.copy(),
                        'split_method': split_method
                    }
                else:
                    return cropped_image

            x, y, w, h = best_face

            # 元画像のサイズを取得
            img_height, img_width = original_image.shape[:2]

            # 顔の中心を計算
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            # 目の検出 (垂直方向の基準点として使用)
            _, eyes_y = self.face_detector.detect_eyes(original_image, x, y, w, h)
            # 目の位置が検出できなかった場合は顔の中心Yを使う
            reference_y = eyes_y if eyes_y is not None else face_center_y

            # アスペクト比文字列を数値に変換
            try:
                width_ratio, height_ratio = map(int, aspect_ratio_str.split(':'))
                target_aspect_ratio = width_ratio / height_ratio
            except ValueError:
                print(f"無効なアスペクト比文字列: {aspect_ratio_str}. デフォルトの16:9を使用します。")
                target_aspect_ratio = 16 / 9  # デフォルト

            # --- クロップサイズの計算 ---
            crop_width, crop_height = self._calculate_target_dimensions(
                img_width, img_height, target_aspect_ratio)

            # --- クロップ位置の計算 ---
            crop_left, crop_top = self._calculate_crop_position(img_width, img_height,
                                                                face_center_x, reference_y,
                                                                crop_width, crop_height,
                                                                split_method)

            crop_right = min(crop_left + crop_width, img_width)
            crop_bottom = min(crop_top + crop_height, img_height)

            cropped_image = original_image[crop_top:crop_bottom, crop_left:crop_right].copy()

            if debug_mode:
                visualizer = Visualizer()

                original_with_faces = visualizer.draw_face_info(original_image.copy(), faces,
                                                                self.last_scored_faces)

                cropped_with_grid = visualizer.draw_debug_info(cropped_image.copy(),
                                                               split_method=split_method)

                return {
                    'original_with_faces': original_with_faces,
                    'cropped_with_grid': cropped_with_grid,
                    'cropped_clean': cropped_image,
                    'split_method': split_method
                }
            else:
                return cropped_image

        except Exception as e:
            print(f"クロップ処理エラー: {str(e)}\n{traceback.format_exc()}")
            return None

    def _calculate_target_dimensions(self, img_width, img_height, target_aspect_ratio):
        """
        目標アスペクト比に合わせたクロップサイズを計算
        """
        img_aspect_ratio = img_width / img_height

        if img_aspect_ratio > target_aspect_ratio:
            crop_height = img_height
            crop_width = int(round(crop_height * target_aspect_ratio))
            if crop_width > img_width:
                crop_width = img_width
                crop_height = int(round(crop_width / target_aspect_ratio))
        else:
            crop_width = img_width
            crop_height = int(round(crop_width / target_aspect_ratio))
            if crop_height > img_height:
                crop_height = img_height
                crop_width = int(round(crop_height * target_aspect_ratio))

        crop_width = max(1, int(round(crop_width)))
        crop_height = max(1, int(round(crop_height)))
        return crop_width, crop_height

    def _calculate_crop_position(self,
                                 img_w,
                                 img_h,
                                 face_center_x,
                                 face_center_y,
                                 target_w,
                                 target_h,
                                 split_method='thirds'):
        """
        顔の中心と目標サイズに基づいてクロップ位置を計算する。
        顔の中心が最も近い分割線に合うように調整する。
        """
        # 1. 目の高さを上部1/3またはファイグリッドの上部ラインに合わせる (垂直方向)
        if split_method == 'thirds':
            crop_top = max(0, face_center_y - target_h // 3)
        elif split_method == 'phi':
            phi = (1 + np.sqrt(5)) / 2
            top_ratio = 1 / (2 + (phi - 1))
            crop_top = max(0, face_center_y - int(target_h * top_ratio))
        else:
            crop_top = max(0, face_center_y - target_h // 3)

        # 2. 顔の中心を最も近い縦の分割線に合わせる (水平方向)
        if split_method == 'thirds':
            line1_target_x = target_w // 3
            line2_target_x = 2 * target_w // 3
            crop_left_if_line1 = face_center_x - line1_target_x
            crop_left_if_line2 = face_center_x - line2_target_x
            center_crop_left = img_w // 2 - target_w // 2
            face_x_in_center_crop = face_center_x - center_crop_left
            if abs(face_x_in_center_crop - line1_target_x) <= abs(face_x_in_center_crop -
                                                                  line2_target_x):
                crop_left = crop_left_if_line1
            else:
                crop_left = crop_left_if_line2
        elif split_method == 'phi':
            phi = (1 + np.sqrt(5)) / 2
            total_ratio = 2 + (phi - 1)
            line1_target_x = int(target_w / total_ratio)
            line2_target_x = int(target_w * (1 + (phi - 1)) / total_ratio)
            crop_left_if_line1 = face_center_x - line1_target_x
            crop_left_if_line2 = face_center_x - line2_target_x
            center_crop_left = img_w // 2 - target_w // 2
            face_x_in_center_crop = face_center_x - center_crop_left
            if abs(face_x_in_center_crop - line1_target_x) <= abs(face_x_in_center_crop -
                                                                  line2_target_x):
                crop_left = crop_left_if_line1
            else:
                crop_left = crop_left_if_line2
        else:
            line1_target_x = target_w // 3
            line2_target_x = 2 * target_w // 3
            crop_left_if_line1 = face_center_x - line1_target_x
            crop_left_if_line2 = face_center_x - line2_target_x
            center_crop_left = img_w // 2 - target_w // 2
            face_x_in_center_crop = face_center_x - center_crop_left
            if abs(face_x_in_center_crop - line1_target_x) <= abs(face_x_in_center_crop -
                                                                  line2_target_x):
                crop_left = crop_left_if_line1
            else:
                crop_left = crop_left_if_line2

        crop_left = int(round(crop_left))
        crop_top = int(round(crop_top))
        crop_left = max(0, min(crop_left, img_w - target_w))
        crop_top = max(0, min(crop_top, img_h - target_h))

        return crop_left, crop_top
