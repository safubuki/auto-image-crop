from face_detector import FaceDetector
from image_cropper import ImageCropper


class ImageProcessor:

    def __init__(self):
        self.face_detector = FaceDetector()
        self.image_cropper = ImageCropper()

    def crop_image(self, original_image, debug_mode=False):
        """
        画像を顔検出に基づいてクロップする関数
        
        Args:
            original_image: 入力画像
            debug_mode: デバッグモード
            
        Returns:
            クロップされた画像または、デバッグモードの場合は可視化された画像
        """
        return self.image_cropper.crop_image(original_image, debug_mode)
