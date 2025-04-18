import os

import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    """
    顔と目の検出を行うクラス
    
    MediaPipeを使用して画像から顔と目を検出する機能を提供します。
    """

    def __init__(self):
        """
        FaceDetectorクラスの初期化メソッド
        
        引数:
            なし
            
        戻り値:
            なし
        """
        # MediaPipeの顔検出モジュールを初期化
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1,
                                                                   min_detection_confidence=0.5)

    def detect_faces(self, image):
        """
        画像から顔を検出するメソッド
        
        MediaPipeを使用して画像から顔を検出し、検出された顔の領域と
        推定された顔の向き（ヨー方向）を返します。
        顔領域は適切に拡張されます。
        
        引数:
            image: 入力画像（OpenCV形式のndarray）
            
        戻り値:
            検出された顔のリスト、各顔は(x, y, w, h, yaw_direction)のタプルで表される
            yaw_direction は 'left', 'right', 'front' のいずれか
        """
        if image is None:
            return []

        # MediaPipeで顔検出
        results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detected_faces_info = []
        if results.detections:
            ih, iw, _ = image.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x, y, w, h = (int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw),
                              int(bboxC.height * ih))

                # 顔向き推定 (ヨー方向)
                yaw_direction = 'front'  # デフォルトは正面
                try:
                    # キーポイントを取得 (0:右目, 1:左目, 2:鼻先)
                    kp_right_eye = detection.location_data.relative_keypoints[0]
                    kp_left_eye = detection.location_data.relative_keypoints[1]
                    kp_nose_tip = detection.location_data.relative_keypoints[2]

                    # x座標のみ使用
                    re_x = kp_right_eye.x
                    le_x = kp_left_eye.x
                    nt_x = kp_nose_tip.x

                    # 目の中心と鼻先の水平方向のずれを計算
                    eye_center_x = (re_x + le_x) / 2.0
                    nose_offset = nt_x - eye_center_x

                    # 目の間の距離で正規化 (ゼロ除算回避)
                    eye_dist = abs(le_x - re_x)
                    if eye_dist > 1e-6:  # 小さすぎる距離は無視
                        normalized_offset = nose_offset / eye_dist

                        # 閾値に基づいて向きを判断 (値は調整可能)
                        yaw_threshold = 0.15
                        yaw = normalized_offset * 100  # ヨー角を計算
                        # ヨー角に基づいて向きを判定 (修正)
                        if yaw < -15:  # 負のヨー角は「左向き」
                            yaw_direction = 'left'
                        elif yaw > 15:  # 正のヨー角は「右向き」
                            yaw_direction = 'right'
                        else:
                            yaw_direction = 'front'
                        # print(f"DEBUG: re={re_x:.2f}, le={le_x:.2f}, nt={nt_x:.2f}, center={(eye_center_x):.2f}, offset={normalized_offset:.2f}, dir={yaw_direction}")
                except IndexError:
                    # キーポイントが取得できない場合がある
                    print("警告: 顔のキーポイントが取得できず、向きを推定できませんでした。")
                    pass  # yaw_direction は 'front' のまま
                except Exception as e:
                    print(f"警告: 顔向き推定中にエラーが発生しました: {e}")
                    pass  # yaw_direction は 'front' のまま

                # 顔領域を拡張（10%拡張）
                padding = int(0.1 * min(w, h))
                exp_x = max(0, x - padding)
                exp_y = max(0, y - padding)
                exp_w = min(iw - exp_x, w + 2 * padding)
                exp_h = min(ih - exp_y, h + 2 * padding)

                detected_faces_info.append((exp_x, exp_y, exp_w, exp_h, yaw_direction))

        return detected_faces_info

    def evaluate_face_sharpness(self, image, face):
        """
        顔領域の鮮明度を評価するメソッド
        
        Laplacian varianceを使用して、顔領域の鮮明度を評価します。
        値が大きいほど鮮明であることを示します。
        
        引数:
            image: 入力画像（OpenCV形式のndarray）
            face: 顔領域のタプル (x, y, w, h)
            
        戻り値:
            鮮明度スコア（float値）
        """
        if image is None or len(face) != 4:
            return 0.0

        x, y, w, h = face

        # 画像のサイズ内に収まるよう座標を調整
        x = max(0, x)
        y = max(0, y)
        w = min(image.shape[1] - x, w)
        h = min(image.shape[0] - y, h)

        if w <= 0 or h <= 0:
            return 0.0

        # 顔領域を切り出す
        face_region = image[y:y + h, x:x + w]

        # グレースケールに変換（色情報は鮮明度評価に不要）
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        # Laplacianフィルタを適用
        laplacian = cv2.Laplacian(gray_face, cv2.CV_64F)

        # 分散を計算（鮮明度スコア）
        sharpness = laplacian.var()

        return sharpness

    def detect_eyes(self, image, face_x, face_y, face_w, face_h):
        """
        顔領域内で目の位置を推定するメソッド
        
        MediaPipeではアイポイントの詳細な検出ができないため、
        顔の位置から目の位置を推定します。
        
        引数:
            image: 入力画像（OpenCV形式のndarray）
            face_x: 顔のx座標
            face_y: 顔のy座標
            face_w: 顔の幅
            face_h: 顔の高さ
            
        戻り値:
            空のリストと推定された目の高さ位置のタプル
        """
        if image is None:
            return [], None

        # 目の検出は行わず、顔の上部1/3あたりを目の位置と推定
        eyes_y = face_y + face_h // 3

        return [], eyes_y
