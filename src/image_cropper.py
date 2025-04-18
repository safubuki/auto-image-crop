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
        self.visualizer = Visualizer()

        # スコアリング用の重み付け係数（各要素の重要度）
        self.weight_size = 0.4  # 顔サイズの重み（大きい顔を優先）
        self.weight_center = 0.4  # 中央度の重み（中央に近い顔を優先）
        self.weight_sharpness = 0.2  # 鮮明度の重み（鮮明な顔を優先）

        # 最後に評価した顔のスコア情報
        self.last_scored_faces = []

    def _calculate_sharpness(self, image, face_rect):
        """顔領域の鮮明度（Laplacian variance）を計算する"""
        x, y, w, h = face_rect
        # 顔領域を切り出し、グレースケールに変換
        face_roi = image[y:y + h, x:x + w]
        if face_roi.size == 0:
            return 0
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        # Laplacian varianceを計算
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        return laplacian_var

    def _score_faces(self, faces_info, img_w, img_h, original_image):
        """検出された顔のリストをスコアリングする"""
        scored_faces = []
        if not faces_info:
            return []

        img_center_x, img_center_y = img_w / 2, img_h / 2
        max_possible_distance = np.sqrt(img_center_x**2 + img_center_y**2)
        max_area = img_w * img_h

        # 各スコアの最大値を計算（正規化のため）
        max_size_score_raw = 0
        max_center_score_raw = 0  # 距離なので最小が良い -> 1 - dist/max_dist なので最大が良い
        max_sharpness_score_raw = 0
        raw_scores = []

        for x, y, w, h, _ in faces_info:
            # 1. サイズスコア (raw)
            area = w * h
            size_score_raw = area / max_area if max_area > 0 else 0
            max_size_score_raw = max(max_size_score_raw, size_score_raw)

            # 2. 中央度スコア (raw)
            face_center_x, face_center_y = x + w / 2, y + h / 2
            distance_from_center = np.sqrt((face_center_x - img_center_x)**2 +
                                           (face_center_y - img_center_y)**2)
            # 距離が0に近いほどスコアが高くなるように変換 (1 - 正規化距離)
            center_score_raw = 1.0 - (distance_from_center /
                                      max_possible_distance) if max_possible_distance > 0 else 0
            max_center_score_raw = max(max_center_score_raw, center_score_raw)

            # 3. 鮮明度スコア (raw)
            sharpness_score_raw = self._calculate_sharpness(original_image, (x, y, w, h))
            max_sharpness_score_raw = max(max_sharpness_score_raw, sharpness_score_raw)

            raw_scores.append({
                'face': (x, y, w, h),
                'size_raw': size_score_raw,
                'center_raw': center_score_raw,
                'sharpness_raw': sharpness_score_raw
            })

        # 正規化と最終スコア計算
        for score_data in raw_scores:
            face = score_data['face']
            # 正規化 (最大値が0の場合は0とする)
            size_score_norm = score_data[
                'size_raw'] / max_size_score_raw if max_size_score_raw > 0 else 0
            center_score_norm = score_data['center_raw']  # 既に0-1の範囲のはずだが、念のため最大値を使う場合
            # center_score_norm = score_data['center_raw'] / max_center_score_raw if max_center_score_raw > 0 else 0
            sharpness_score_norm = score_data[
                'sharpness_raw'] / max_sharpness_score_raw if max_sharpness_score_raw > 0 else 0

            # 重み付けして最終スコアを計算
            final_score = (self.weight_size * size_score_norm +
                           self.weight_center * center_score_norm +
                           self.weight_sharpness * sharpness_score_norm)

            scored_faces.append({
                'face': face,
                'score': final_score,
                'size_score': size_score_norm,
                'center_score': center_score_norm,
                'sharpness_score': sharpness_score_norm
            })

        # スコアで降順ソート
        scored_faces.sort(key=lambda x: x['score'], reverse=True)
        self.last_scored_faces = scored_faces  # 最後に評価したスコアを保存
        return scored_faces

    def _select_best_face(self, scored_faces):
        """
        スコアリングされた顔の中から最適な顔（スコアが最も高い顔）を選択する
        戻り値: {'face': (x, y, w, h), 'score': ..., ...} または None
        """
        if not scored_faces:
            return None
        # scored_faces は既にスコアでソートされている前提
        return scored_faces[0]

    def _calculate_target_dimensions(self, img_width, img_height,
                                     target_aspect_ratio_str):  # 引数名を変更
        """
        目標アスペクト比に合わせたクロップサイズを計算
        """
        # 文字列のアスペクト比を数値に変換
        try:
            w_str, h_str = target_aspect_ratio_str.split(':')
            target_aspect_ratio = float(w_str) / float(h_str)
        except (ValueError, ZeroDivisionError):
            print(f'警告: 不正なアスペクト比文字列です: {target_aspect_ratio_str}。デフォルトの16:9を使用します。')
            target_aspect_ratio = 16 / 9

        img_aspect_ratio = img_width / img_height

        if img_aspect_ratio > target_aspect_ratio:  # 数値同士で比較
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

    def _calculate_crop_position(
            self,
            img_w,
            img_h,
            face_x,  # face_center_x の代わりに face_x を使用
            face_y,  # face_center_y の代わりに face_y を使用
            face_w,  # 水平方向の計算で使用
            face_h,  # 目の高さを計算するために使用
            target_w,
            target_h,
            split_method='thirds',
            yaw_direction='front'):
        """
        顔の位置、目標サイズ、顔の向きに基づいてクロップ位置を計算する。
        垂直方向：目の高さがグリッド線に来るように調整する。
        水平方向：顔の中心が最も近い分割線に合うように調整する。
        顔の向きに応じて空間を空けるように調整する。
        """
        # 1. 目の高さを合わせる (垂直方向)
        # 顔の上端から約1/3の位置を目と仮定
        eye_y = face_y + face_h // 3

        if split_method == 'phi':
            phi = (1 + np.sqrt(5)) / 2
            # ファイグリッドの上側の線の位置（クロップ後の高さに対する割合）
            # 黄金比の分割点 (小さい方) は 1 / phi^2 ~= 0.382
            top_ratio = 1 / (phi**2)
            target_eye_y_in_crop = int(target_h * top_ratio)
        else:  # デフォルトは三分割法
            # 三分割法の上側の線の位置（クロップ後の高さに対する割合）
            target_eye_y_in_crop = target_h // 3

        # クロップの上端を計算 (目の位置 - 目標の目の位置)
        crop_top = eye_y - target_eye_y_in_crop

        # 2. 顔の中心を合わせる (水平方向) - 変更なしのロジックだが、顔の向きの考慮を修正
        face_center_x = face_x + face_w // 2  # 水平方向は中心を使う
        if split_method == 'phi':
            phi = (1 + np.sqrt(5)) / 2
            # ファイグリッドの左右の線のX座標（クロップ後の幅に対する割合）
            # 比率 1 : phi-1 : 1 -> 分割点は 1/phi^2 と 1 - 1/phi^2 = 1/phi
            left_ratio = 1 / (phi**2)
            right_ratio = 1 / phi
            line1_target_x = int(target_w * left_ratio)
            line2_target_x = int(target_w * right_ratio)  # 右側の線
        else:  # デフォルトは三分割法
            line1_target_x = target_w // 3
            line2_target_x = 2 * target_w // 3

        target_face_x_in_crop = 0
        if yaw_direction == 'right':
            # 顔が右向き -> 視線の先に空間 -> 左側の線に合わせる
            target_face_x_in_crop = line1_target_x
        elif yaw_direction == 'left':
            # 顔が左向き -> 視線の先に空間 -> 右側の線に合わせる
            target_face_x_in_crop = line2_target_x
        else:  # 正面向き
            # 顔の中心がどちらの線に近いかで決定
            center_crop_left = img_w // 2 - target_w // 2
            face_x_in_center_crop = face_center_x - center_crop_left
            if abs(face_x_in_center_crop - line1_target_x) <= abs(face_x_in_center_crop -
                                                                  line2_target_x):
                target_face_x_in_crop = line1_target_x
            else:
                target_face_x_in_crop = line2_target_x

        # クロップの左端を計算 (顔の中心 - 目標の顔の中心位置)
        crop_left = face_center_x - target_face_x_in_crop

        # crop_left, crop_top を整数に変換し、範囲内に収める
        crop_left = int(round(crop_left))
        crop_top = int(round(crop_top))
        crop_left = max(0, min(crop_left, img_w - target_w))
        crop_top = max(0, min(crop_top, img_h - target_h))

        return crop_left, crop_top

    def crop_image(self,
                   original_image,
                   aspect_ratio_str='16:9',
                   split_method='thirds',
                   debug_mode=False):
        """
        画像を顔認識に基づいてクロップするメソッド
        
        引数:
            original_image: 元画像（OpenCV形式のndarray）
            aspect_ratio_str: 目標のアスペクト比（例: '16:9'）
            split_method: 分割法 ('thirds' または 'phi')
            debug_mode: デバッグ情報を返すかどうか
            
        戻り値:
            クロップされた画像（ndarray）、またはデバッグ情報を含む辞書
        """
        if original_image is None:
            return None

        try:
            img_h, img_w = original_image.shape[:2]
            # _calculate_target_dimensions に文字列のアスペクト比を渡す
            target_w, target_h = self._calculate_target_dimensions(img_w, img_h, aspect_ratio_str)

            if target_w <= 0 or target_h <= 0:
                print('エラー: 目標サイズが不正です。クロップできません。')
                return original_image  # または None

            # 顔検出 (向き情報を含む)
            faces_info = self.face_detector.detect_faces(original_image)

            # 顔のスコアリング
            scored_faces = self._score_faces(faces_info, img_w, img_h, original_image)

            # 最適な顔を選択 (スコアベース)
            best_face_data = self._select_best_face(scored_faces)

            # Extract just the face rectangles (x, y, w, h) for drawing
            # Note: faces_info contains yaw, scored_faces does not directly, but corresponds by rect
            faces_rects = [(x, y, w, h) for x, y, w, h, _ in faces_info] if faces_info else []

            if best_face_data:
                # best_face_data から顔の座標と、元の faces_info から対応する向きを取得
                best_face_rect = best_face_data['face']
                face_x, face_y, face_w, face_h = best_face_rect
                # best_face_rect に対応する yaw_direction を faces_info から探す
                yaw_direction = 'front'  # デフォルト
                for x, y, w, h, direction in faces_info:
                    if (x, y, w, h) == best_face_rect:
                        yaw_direction = direction
                        break

                # クロップ位置を計算 (顔の矩形情報と向きを渡す)
                crop_x, crop_y = self._calculate_crop_position(img_w, img_h, face_x, face_y, face_w,
                                                               face_h, target_w, target_h,
                                                               split_method, yaw_direction)
            else:
                # 顔が検出されなかった場合、中央クロップ
                print('警告: 顔が検出されませんでした。中央クロップを実行します。')
                crop_x = (img_w - target_w) // 2
                crop_y = (img_h - target_h) // 2
                yaw_direction = 'front'  # デフォルト

            # 画像をクロップ
            cropped_clean = original_image[crop_y:crop_y + target_h, crop_x:crop_x + target_w]

            if debug_mode:
                # デバッグ用に元画像に情報を描画 (draw_face_info を使用)
                # スコアリング結果を渡す
                original_with_faces = self.visualizer.draw_face_info(original_image, faces_rects,
                                                                     scored_faces)

                cropped_with_grid = self.visualizer.draw_debug_info(cropped_clean.copy(),
                                                                    split_method)
                # クロップ領域を元画像に描画 (これは残す)
                cv2.rectangle(original_with_faces, (crop_x, crop_y),
                              (crop_x + target_w, crop_y + target_h), (0, 0, 255),
                              3)  # 赤色の太線でクロップ領域

                return {
                    'original_with_faces': original_with_faces,
                    'cropped_with_grid': cropped_with_grid,
                    'cropped_clean': cropped_clean,  # 保存用のクリーンな画像
                    'split_method': split_method,
                    'aspect_ratio': aspect_ratio_str,
                    'face_detected': bool(best_face_data),
                    'yaw_direction': yaw_direction if best_face_data else 'N/A'
                }
            else:
                return cropped_clean

        except Exception as e:
            print(f'クロップ処理中にエラーが発生しました: {e}')
            traceback.print_exc()
            return original_image  # エラー時は元画像を返す
