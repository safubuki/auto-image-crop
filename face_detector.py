import os

import cv2
import dlib
import mediapipe as mp
import numpy as np


class FaceDetector:

    def __init__(self):
        # MediaPipeの顔検出モジュールを初期化
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1,
                                                                   min_detection_confidence=0.5)

        # 複数の顔検出カスケード分類器を読み込む
        cascade_path = cv2.data.haarcascades
        self.face_cascade_default = cv2.CascadeClassifier(cascade_path +
                                                          'haarcascade_frontalface_default.xml')
        self.face_cascade_alt = cv2.CascadeClassifier(cascade_path +
                                                      'haarcascade_frontalface_alt.xml')
        self.face_cascade_alt2 = cv2.CascadeClassifier(cascade_path +
                                                       'haarcascade_frontalface_alt2.xml')

        # LBPベースの分類器は存在確認してから読み込む
        self.lbp_face_cascade = None
        lbp_path = os.path.join(cascade_path, 'lbpcascade_frontalface.xml')
        if os.path.exists(lbp_path):
            self.lbp_face_cascade = cv2.CascadeClassifier(lbp_path)

        # DlibのHOGベースの顔検出器
        self.dlib_face_detector = dlib.get_frontal_face_detector()

        # 目検出用の分類器
        self.eye_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_eye.xml')
        self.eye_cascade_glasses = cv2.CascadeClassifier(cascade_path +
                                                         'haarcascade_eye_tree_eyeglasses.xml')

        # 横顔検出用の分類器
        self.profile_face_cascade = cv2.CascadeClassifier(cascade_path +
                                                          'haarcascade_profileface.xml')

        # OpenCV DNNベースの顔検出モデルを読み込む
        self.dnn_model_path = os.path.join(cascade_path, 'deploy.prototxt')
        self.dnn_weights_path = os.path.join(cascade_path,
                                             'res10_300x300_ssd_iter_140000.caffemodel')
        if os.path.exists(self.dnn_model_path) and os.path.exists(self.dnn_weights_path):
            self.dnn_face_detector = cv2.dnn.readNetFromCaffe(self.dnn_model_path,
                                                              self.dnn_weights_path)
        else:
            self.dnn_face_detector = None

    def detect_faces(self, image):
        """
        画像から顔を検出する関数
        
        Args:
            image: 入力画像
            
        Returns:
            検出された顔のリスト(x, y, w, h)
        """
        if image is None:
            return []

        # グレースケールに変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # MediaPipeで顔検出
        results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mediapipe_faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = (int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw),
                              int(bboxC.height * ih))
                mediapipe_faces.append((x, y, w, h))

        # すべての検出結果を集約
        all_faces = mediapipe_faces

        # 重複する検出を統合
        if len(all_faces) > 0:
            unique_faces = []
            for (x, y, w, h) in all_faces:
                is_unique = True
                for (ux, uy, uw, uh) in unique_faces:
                    center_dist = np.sqrt((x + w / 2 - ux - uw / 2)**2 +
                                          (y + h / 2 - uy - uh / 2)**2)
                    overlap_threshold = (w + uw) / 4
                    if center_dist < overlap_threshold:
                        is_unique = False
                        if w * h > uw * uh:
                            unique_faces.remove((ux, uy, uw, uh))
                            unique_faces.append((x, y, w, h))
                        break
                if is_unique:
                    # 顔領域を拡張（10%拡張）
                    padding = int(0.1 * min(w, h))
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(image.shape[1] - x, w + 2 * padding)
                    h = min(image.shape[0] - y, h + 2 * padding)
                    unique_faces.append((x, y, w, h))

            faces = unique_faces
        else:
            faces = []

        return faces

    def detect_eyes(self, image, face_x, face_y, face_w, face_h):
        """
        顔領域内で目を検出する関数
        
        Args:
            image: 入力画像
            face_x, face_y, face_w, face_h: 顔の座標と大きさ
            
        Returns:
            検出された目のリスト(x, y, w, h)と推定された目の高さの位置
        """
        if image is None:
            return [], None

        # グレースケールに変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 顔領域を取得
        face_roi = gray[face_y:face_y + face_h, face_x:face_x + face_w]

        # 目の検出を試みる（通常の目検出器とメガネ用の検出器を両方使用）
        eyes = self.eye_cascade.detectMultiScale(face_roi)
        if len(eyes) < 2:  # 通常の検出器で2つ未満の場合はメガネ用検出器を試す
            eyes = self.eye_cascade_glasses.detectMultiScale(face_roi, 1.1, 3)

        # 顔の中心を計算
        face_center_y = face_y + face_h // 2

        # 目の位置を計算（検出された場合）
        eyes_y = face_center_y
        if len(eyes) >= 2:
            # 目の位置（縦）の平均を計算
            eye_y_positions = []
            for ex, ey, ew, eh in eyes:
                eye_center_y = face_y + ey + eh // 2
                eye_y_positions.append(eye_center_y)

            # 目の平均位置を使用
            eyes_y = sum(eye_y_positions) // len(eye_y_positions)
        else:
            # 目が検出できない場合は、顔の上部1/3あたりを目の位置と推定
            eyes_y = face_y + face_h // 3

        return eyes, eyes_y
