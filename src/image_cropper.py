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

    def crop_image(self, original_image, aspect_ratio_str="16:9", debug_mode=False):
        """
        顔検出に基づいて画像をクロップするメソッド

        検出された顔の位置と目の位置を基準に、三分割法を適用して
        最適な位置で画像をクロップします。デバッグモードでは
        グリッド線と検出結果を可視化します。

        引数:
            original_image: 入力画像（OpenCV形式のndarray）
            aspect_ratio_str: 目標のアスペクト比 ("16:9", "9:16", "4:3", "3:4")
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

            # 複合的な評価に基づいて最適な顔を選択
            best_face = self.select_best_face(original_image, faces)
            if best_face is None:
                return None

            x, y, w, h = best_face

            # 元画像のサイズを取得
            img_height, img_width = original_image.shape[:2]

            # 顔の中心を計算
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            # 目の検出
            _, eyes_y = self.face_detector.detect_eyes(original_image, x, y, w, h)

            # アスペクト比文字列を数値に変換
            try:
                width_ratio, height_ratio = map(int, aspect_ratio_str.split(':'))
                target_aspect_ratio = width_ratio / height_ratio
            except ValueError:
                print(f"無効なアスペクト比文字列: {aspect_ratio_str}. デフォルトの16:9を使用します。")
                target_aspect_ratio = 16 / 9  # デフォルト

            # --- クロップサイズの計算 ---
            img_aspect_ratio = img_width / img_height

            if img_aspect_ratio > target_aspect_ratio:
                # 元画像の方が横長 -> 高さを基準に幅を計算
                crop_height = img_height
                crop_width = int(crop_height * target_aspect_ratio)
                # 幅が元画像を超えないように調整
                if crop_width > img_width:
                    crop_width = img_width
                    crop_height = int(crop_width / target_aspect_ratio)
            else:
                # 元画像の方が縦長または同じ -> 幅を基準に高さを計算
                crop_width = img_width
                crop_height = int(crop_width / target_aspect_ratio)
                # 高さが元画像を超えないように調整
                if crop_height > img_height:
                    crop_height = img_height
                    crop_width = int(crop_height * target_aspect_ratio)

            # 整数に丸める
            crop_width = int(round(crop_width))
            crop_height = int(round(crop_height))

            # --- クロップ位置の計算 ---
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

            # 最終的なクロップ範囲を設定 (整数に変換)
            crop_top = int(round(crop_top))
            crop_left = int(round(crop_left))
            crop_right = min(crop_left + crop_width, img_width)
            crop_bottom = min(crop_top + crop_height, img_height)

            # この範囲で画像をクロップ
            cropped_image = original_image[crop_top:crop_bottom, crop_left:crop_right].copy()

            # クロップ後の画像サイズが意図通りか確認（デバッグ用）
            # print(f"Cropped size: {cropped_image.shape[1]}x{cropped_image.shape[0]}, Target: {crop_width}x{crop_height}")

            # MediaPipeで再検出（デバッグ表示用）
            cropped_faces = self.face_detector.detect_faces(cropped_image)

            if debug_mode:
                # デバッグモードの場合は、可視化した画像を返す
                from visualization import Visualizer
                visualizer = Visualizer()

                # 元画像に顔情報を描画
                original_with_faces = visualizer.draw_face_info(original_image.copy(), faces,
                                                                self.last_scored_faces)

                # クロップ後の画像に三分割線を描画
                cropped_with_grid = visualizer.draw_debug_info(cropped_image, faces, crop_left,
                                                               crop_top, face_center_x,
                                                               face_center_y, cropped_faces,
                                                               self.last_scored_faces)

                # 元の画像の矩形+スコア情報と、クロップ後のグリッド線を別々に返す
                return {
                    'original_with_faces': original_with_faces,
                    'cropped_with_grid': cropped_with_grid,
                    'cropped_clean': cropped_image  # 保存用にクリーンな画像も返す
                }
            else:
                # 通常モードではクロップ画像のみ返す
                return cropped_image

        except Exception as e:
            import traceback
            print(f"クロップ処理エラー: {str(e)}\n{traceback.format_exc()}")
            return None
