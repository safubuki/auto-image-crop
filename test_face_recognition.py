import os
import pytest
import cv2

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
        image_files = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
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
        image_files = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
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
        image_files = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
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
            assert abs(aspect_ratio - (16/9)) < 0.1, f"画像 {image_file} のクロップ後のアスペクト比が16:9ではありません（{aspect_ratio}）"
            
            print(f"✓ 画像 {image_file} を正常にクロップしました（{width}x{height}, アスペクト比: {aspect_ratio:.2f}）")
            
            # オプション: クロップした画像を保存する場合
            if save_cropped:
                base_name, ext = os.path.splitext(image_file)
                output_path = os.path.join(temp_dir, f"{base_name}_cropped{ext}")
                cv2.imwrite(output_path, cropped_image)
                print(f"  クロップ画像を保存しました: {output_path}")
            else:
                print(f"  クロップ画像はテスト後に破棄されます（メモリ上のみで処理）")