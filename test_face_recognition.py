import os
import time

import cv2
import pytest

from face_detector import FaceDetector
from image_processor import ImageProcessor


class TestFaceRecognition:
    """
    画像の顔認識機能をテストするクラス
    """

    @pytest.fixture
    def face_detector(self):
        """
        FaceDetectorインスタンスを提供するフィクスチャ
        
        引数:
            なし
            
        戻り値:
            FaceDetectorのインスタンス
        """
        return FaceDetector()

    @pytest.fixture
    def image_processor(self):
        """
        ImageProcessorインスタンスを提供するフィクスチャ
        
        引数:
            なし
            
        戻り値:
            ImageProcessorのインスタンス
        """
        return ImageProcessor()

    @pytest.fixture
    def test_images_path(self):
        """
        テスト画像フォルダのパスを提供するフィクスチャ
        
        引数:
            なし
            
        戻り値:
            テスト画像フォルダの絶対パス
        """
        # 現在のスクリプトのディレクトリからの相対パスで取得
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "test_images")

    def test_image_loading(self, test_images_path):
        """
        すべてのテスト画像が正しくロードできることをテストする
        
        引数:
            test_images_path: テスト画像フォルダのパス
            
        戻り値:
            なし
        """
        print("\n===== 画像ロードテスト =====")
        image_files = [
            f for f in os.listdir(test_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        assert len(image_files) >= 5, f"テスト用の画像が5枚未満です（{len(image_files)}枚検出）"
        print(f"検出された画像ファイル数: {len(image_files)}")

        for image_file in image_files:
            image_path = os.path.join(test_images_path, image_file)
            image = cv2.imread(image_path)
            assert image is not None, f"画像 {image_file} を読み込めませんでした"
            assert image.shape[0] > 0 and image.shape[1] > 0, f"画像 {image_file} のサイズが無効です"
            print(f"✓ 画像 {image_file} を正常に読み込みました（{image.shape[1]}x{image.shape[0]}）")

    def test_face_detection(self, face_detector, test_images_path):
        """
        すべてのテスト画像から顔が検出できることをテストする
        
        引数:
            face_detector: FaceDetectorのインスタンス
            test_images_path: テスト画像フォルダのパス
            
        戻り値:
            なし
        """
        print("\n===== 顔検出テスト =====")
        image_files = [
            f for f in os.listdir(test_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        print(f"テスト対象ファイル: {image_files}")

        for image_file in image_files:
            image_path = os.path.join(test_images_path, image_file)
            image = cv2.imread(image_path)

            # 顔を検出
            faces = face_detector.detect_faces(image)

            # 少なくとも1つの顔が検出されることを確認
            assert len(faces) > 0, f"画像 {image_file} から顔を検出できませんでした"

            # 検出された顔の座標とサイズが有効であることを確認
            for i, (x, y, w, h) in enumerate(faces):
                assert x >= 0 and y >= 0, f"画像 {image_file} で検出された顔の座標が無効です"
                assert w > 0 and h > 0, f"画像 {image_file} で検出された顔のサイズが無効です"
                print(f"  - 顔 {i+1}: 位置({x},{y}), サイズ({w}x{h})")

            print(f"✓ 画像 {image_file} から {len(faces)} 個の顔を検出しました")

    def test_image_cropping(self, image_processor, test_images_path):
        """
        すべてのテスト画像が正しくクロップできることをテストする
        
        引数:
            image_processor: ImageProcessorのインスタンス
            test_images_path: テスト画像フォルダのパス
            
        戻り値:
            なし
        """
        print("\n===== 画像クロップテスト =====")
        image_files = [
            f for f in os.listdir(test_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        # テスト用の一時フォルダ（クロップ画像を保存する場合）
        temp_dir = os.path.join(os.path.dirname(test_images_path), "test_cropped")
        save_cropped = False  # クロップ画像を保存するかどうかのフラグ

        if save_cropped and not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            print(f"クロップされた画像の保存先を作成しました: {temp_dir}")

        for image_file in image_files:
            image_path = os.path.join(test_images_path, image_file)
            image = cv2.imread(image_path)

            # 画像をクロップ
            cropped_image = image_processor.crop_image(image, debug_mode=False)

            # クロップ結果が生成されていることを確認
            assert cropped_image is not None, f"画像 {image_file} のクロップに失敗しました"

            # クロップされた画像のアスペクト比が約16:9であることを確認
            height, width = cropped_image.shape[:2]
            aspect_ratio = width / height
            assert abs(aspect_ratio -
                       (16 / 9)) < 0.1, f"画像 {image_file} のクロップ後のアスペクト比が16:9ではありません（{aspect_ratio}）"

            print(f"✓ 画像 {image_file} を正常にクロップしました（{width}x{height}, アスペクト比: {aspect_ratio:.2f}）")

            # オプション: クロップした画像を保存する場合
            if save_cropped:
                base_name, ext = os.path.splitext(image_file)
                output_path = os.path.join(temp_dir, f"{base_name}_cropped{ext}")
                cv2.imwrite(output_path, cropped_image)
                print(f"  クロップ画像を保存しました: {output_path}")
            else:
                print(f"  クロップ画像はテスト後に破棄されます（メモリ上のみで処理）")

    def test_individual_face_detection_engines(self, test_images_path):
        """
        各顔認識エンジンを個別にテストし、性能を評価する
        
        引数:
            test_images_path: テスト画像フォルダのパス
            
        戻り値:
            なし
        """
        print("\n===== 各顔認識エンジンの評価テスト =====")
        image_files = [
            f for f in os.listdir(test_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        # テスト対象の顔認識エンジン
        engines = [
            "mediapipe", "haar_default", "haar_alt", "haar_alt2", "lbp", "dlib", "opencv_dnn"
        ]

        # 各エンジンの結果を格納する辞書
        results = {
            engine: {
                "detected_faces": 0,
                "total_images": len(image_files),
                "time": 0
            } for engine in engines
        }

        # テスト専用のFaceDetectorを作成
        detector = FaceDetector()

        for image_file in image_files:
            image_path = os.path.join(test_images_path, image_file)
            image = cv2.imread(image_path)
            print(f"\n画像: {image_file}")

            # グレースケールに変換（複数のエンジンで使用）
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 各エンジンでテスト
            for engine in engines:
                faces = []
                start_time = time.time()

                # MediaPipe
                if engine == "mediapipe":
                    # MediaPipeを直接使わず、FaceDetectorのメソッドを使用
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results_mp = detector.face_detection.process(rgb_image)
                    if results_mp.detections:
                        for detection in results_mp.detections:
                            bboxC = detection.location_data.relative_bounding_box
                            ih, iw, _ = image.shape
                            x, y, w, h = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                                          int(bboxC.width * iw), int(bboxC.height * ih))
                            faces.append((x, y, w, h))

                # OpenCV Haar Cascade - default
                elif engine == "haar_default":
                    faces = detector.face_cascade_default.detectMultiScale(gray, 1.3, 5)

                # OpenCV Haar Cascade - alt
                elif engine == "haar_alt":
                    faces = detector.face_cascade_alt.detectMultiScale(gray, 1.3, 5)

                # OpenCV Haar Cascade - alt2
                elif engine == "haar_alt2":
                    faces = detector.face_cascade_alt2.detectMultiScale(gray, 1.3, 5)

                # OpenCV LBP Cascade
                elif engine == "lbp" and detector.lbp_face_cascade is not None:
                    faces = detector.lbp_face_cascade.detectMultiScale(gray, 1.3, 5)

                # Dlib HOG
                elif engine == "dlib":
                    dlib_faces = detector.dlib_face_detector(gray, 1)
                    for face in dlib_faces:
                        x, y = face.left(), face.top()
                        w, h = face.right() - x, face.bottom() - y
                        faces.append((x, y, w, h))

                # OpenCV DNN
                elif engine == "opencv_dnn" and detector.dnn_face_detector is not None:
                    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                                 (104.0, 177.0, 123.0))
                    detector.dnn_face_detector.setInput(blob)
                    detections = detector.dnn_face_detector.forward()

                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.5:  # 信頼度閾値
                            box = detections[0, 0, i, 3:7] * [
                                image.shape[1], image.shape[0], image.shape[1], image.shape[0]
                            ]
                            (x, y, x2, y2) = box.astype("int")
                            w, h = x2 - x, y2 - y
                            faces.append((x, y, w, h))

                elapsed_time = time.time() - start_time
                results[engine]["time"] += elapsed_time

                # 顔が検出されたかどうか
                if len(faces) > 0:
                    results[engine]["detected_faces"] += 1
                    status = "✓"
                else:
                    status = "✗"

                print(f"  {status} {engine}: {len(faces)}個の顔を検出 ({elapsed_time:.3f}秒)")

        # 結果のサマリーを表示
        print("\n===== 顔認識エンジンの評価結果 =====")
        print(f"{'エンジン名':<15} {'検出率':<10} {'平均処理時間':<15}")
        print("-" * 40)

        for engine, data in results.items():
            detection_rate = (data["detected_faces"] / data["total_images"]) * 100
            avg_time = data["time"] / data["total_images"]
            print(f"{engine:<15} {detection_rate:>5.1f}% {avg_time:>12.3f}秒")
