from face_detector import FaceDetector
from image_cropper import ImageCropper


class ImageProcessor:
    """
    画像処理の統合インターフェースを提供するクラス
    
    顔検出とクロップ処理を組み合わせた統合APIを提供し、
    アプリケーションからの利用を簡素化します。
    """

    def __init__(self):
        """
        ImageProcessorクラスの初期化メソッド
        
        顔検出とクロップ処理のためのオブジェクトを初期化します。
        
        引数:
            なし
            
        戻り値:
            なし
        """
        self.face_detector = FaceDetector()
        self.image_cropper = ImageCropper()

    def crop_image(self,
                   original_image,
                   aspect_ratio_str='16:9',
                   split_method='thirds',
                   debug_mode=False):
        '''
        画像を顔検出に基づいてクロップするメソッド
        
        指定された画像に顔検出を適用し、最適な位置でクロップします。
        このメソッドはImageCropperクラスの同名メソッドに処理を委譲します。
        
        引数:
            original_image: 入力画像（OpenCV形式のndarray）
            aspect_ratio_str: 目標のアスペクト比 ("16:9", "9:16", "4:3", "3:4")
            split_method (str): 分割方法 ('thirds' または 'phi')
            debug_mode: デバッグモードフラグ（True=可視化情報を含む画像を返す）
            
        戻り値:
            クロップされた画像または、デバッグモードの場合は可視化された画像
        '''
        return self.image_cropper.crop_image(original_image, aspect_ratio_str, split_method,
                                             debug_mode)
