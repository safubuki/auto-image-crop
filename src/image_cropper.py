import cv2
import numpy as np

from face_detector import FaceDetector


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

    def crop_image(self, original_image, debug_mode=False):
        """
        顔検出に基づいて画像をクロップするメソッド
        
        検出された顔の位置と目の位置を基準に、三分割法を適用して
        最適な位置で画像をクロップします。デバッグモードでは
        グリッド線と検出結果を可視化します。
        
        引数:
            original_image: 入力画像（OpenCV形式のndarray）
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
                return None

            # 最も大きい顔を使用（複数ある場合）
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]

            # 元画像のサイズを取得
            img_height, img_width = original_image.shape[:2]

            # 顔の中心を計算
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            # 目の検出
            _, eyes_y = self.face_detector.detect_eyes(original_image, x, y, w, h)

            # 16:9のアスペクト比で計算
            target_aspect_ratio = 16 / 9

            # 最終的なクロップの高さと幅を決定
            crop_height = min(img_height, int(img_width / target_aspect_ratio))
            crop_width = min(img_width, int(crop_height * target_aspect_ratio))

            # 三分割法の上側のライン位置を計算（クロップ後の上から1/3の位置）
            top_third_line = crop_height // 3

            # 顔または目が上側の横ラインに来るように調整
            target_y = eyes_y  # 目の位置または推定位置を使用

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
            cropped_image = original_image[int(crop_top):int(crop_bottom),
                                           int(crop_left):int(crop_right)].copy()

            # MediaPipeで再検出
            cropped_faces = self.face_detector.detect_faces(cropped_image)

            if debug_mode:
                # デバッグモードの場合は、可視化した画像を返す
                from visualization import Visualizer
                visualizer = Visualizer()
                display_image = visualizer.draw_debug_info(cropped_image, faces, crop_left,
                                                           crop_top, face_center_x, face_center_y,
                                                           cropped_faces)
                return display_image
            else:
                return cropped_image

        except Exception as e:
            import traceback
            print(f"クロップ処理エラー: {str(e)}\n{traceback.format_exc()}")
            return None
