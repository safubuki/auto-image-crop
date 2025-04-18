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

    def _select_best_face(self, faces_info):
        """
        検出された顔の中から最適な顔を選択する（現在は最大の顔を選択）
        戻り値: (x, y, w, h, yaw_direction) または None
        """
        if not faces_info:
            return None
        # 最も面積が大きい顔を選択
        # faces_info の各要素は (x, y, w, h, yaw_direction)
        return max(faces_info, key=lambda face: face[2] * face[3])

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

    def _calculate_crop_position(self,
                                 img_w,
                                 img_h,
                                 face_center_x,
                                 face_center_y,
                                 target_w,
                                 target_h,
                                 split_method='thirds',
                                 yaw_direction='front'):  # 顔の向きを追加
        """
        顔の中心、目標サイズ、顔の向きに基づいてクロップ位置を計算する。
        顔の中心が最も近い分割線に合うように調整する。
        顔の向きに応じて空間を空けるように調整する。
        """
        # 1. 目の高さを合わせる (垂直方向) - 変更なし
        if split_method == 'phi':
            phi = (1 + np.sqrt(5)) / 2
            top_ratio = 1 / (2 + (phi - 1))
            crop_top = max(0, face_center_y - int(target_h * top_ratio))
        else:  # デフォルトは三分割法
            crop_top = max(0, face_center_y - target_h // 3)

        # 2. 顔の中心を合わせる (水平方向)
        # グリッド線の目標X座標（クロップ後の画像内）
        if split_method == 'phi':
            phi = (1 + np.sqrt(5)) / 2
            total_ratio = 2 + (phi - 1)
            line1_target_x = int(target_w / total_ratio)
            line2_target_x = int(target_w * (1 + (phi - 1)) / total_ratio)
        else:  # デフォルトは三分割法
            line1_target_x = target_w // 3
            line2_target_x = 2 * target_w // 3

        # 顔を合わせる目標のX座標（クロップ後の画像内）を決定
        target_face_x_in_crop = 0

        if yaw_direction == 'right':  # 顔が右向き -> 右側の線に合わせる (これで期待通りになるか試す)
            target_face_x_in_crop = line2_target_x
            # print("DEBUG: Face Right -> Align Right Line (Trying reverse)")
        elif yaw_direction == 'left':  # 顔が左向き -> 左側の線に合わせる (これで期待通りになるか試す)
            target_face_x_in_crop = line1_target_x
            # print("DEBUG: Face Left -> Align Left Line (Trying reverse)")
        else:
            # 顔が正面向き、または向き不明の場合 -> 最も近い線に合わせる
            # 一旦中央揃えでクロップした場合の顔の位置を考える
            center_crop_left = img_w // 2 - target_w // 2
            # 元画像内での顔の中心座標 face_center_x から、中央クロップの左端座標を引く
            face_x_in_center_crop = face_center_x - center_crop_left

            # 中央揃えクロップ内の顔の位置が、どちらの線に近いか
            if abs(face_x_in_center_crop - line1_target_x) <= abs(face_x_in_center_crop -
                                                                  line2_target_x):
                target_face_x_in_crop = line1_target_x  # line1 に近い
                # print("DEBUG: Align Nearest Line (Line 1)")
            else:
                target_face_x_in_crop = line2_target_x  # line2 に近い
                # print("DEBUG: Align Nearest Line (Line 2)")

        # 決定した目標X座標に基づいて crop_left を計算
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

            # 最適な顔を選択
            best_face_info = self._select_best_face(faces_info)

            if best_face_info:
                face_x, face_y, face_w, face_h, yaw_direction = best_face_info
                face_center_x = face_x + face_w // 2
                face_center_y = face_y + face_h // 2

                # クロップ位置を計算 (顔の向きを渡す)
                crop_x, crop_y = self._calculate_crop_position(img_w, img_h, face_center_x,
                                                               face_center_y, target_w, target_h,
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
                # デバッグ用に元画像とクロップ画像に情報を描画
                original_with_faces = original_image.copy()
                if faces_info:
                    for x, y, w, h, direction in faces_info:
                        cv2.rectangle(original_with_faces, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # 顔の向きを描画
                        cv2.putText(original_with_faces, f"Yaw: {direction}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cropped_with_grid = self.visualizer.draw_debug_info(cropped_clean.copy(),
                                                                    split_method)
                # クロップ領域を元画像に描画
                cv2.rectangle(original_with_faces, (crop_x, crop_y),
                              (crop_x + target_w, crop_y + target_h), (0, 0, 255),
                              3)  # 赤色の太線でクロップ領域

                return {
                    'original_with_faces': original_with_faces,
                    'cropped_with_grid': cropped_with_grid,
                    'cropped_clean': cropped_clean,  # 保存用のクリーンな画像
                    'split_method': split_method,
                    'aspect_ratio': aspect_ratio_str,
                    'face_detected': bool(best_face_info),
                    'yaw_direction': yaw_direction if best_face_info else 'N/A'
                }
            else:
                return cropped_clean

        except Exception as e:
            print(f'クロップ処理中にエラーが発生しました: {e}')
            traceback.print_exc()
            return original_image  # エラー時は元画像を返す
