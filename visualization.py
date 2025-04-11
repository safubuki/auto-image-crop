import cv2


class Visualizer:

    def draw_debug_info(self, cropped_image, faces, crop_left, crop_top, face_center_x,
                        face_center_y, cropped_faces):
        """
        デバッグ情報を画像に描画する関数
        
        Args:
            cropped_image: クロップされた画像
            faces: 元の画像で検出された顔のリスト
            crop_left: クロップ開始位置（左）
            crop_top: クロップ開始位置（上）
            face_center_x: 元画像内の顔の中心X座標
            face_center_y: 元画像内の顔の中心Y座標
            cropped_faces: クロップした画像内で検出された顔のリスト
            
        Returns:
            デバッグ情報が描画された画像
        """
        # デバッグ用：三分割のグリッドと顔矩形を表示
        display_image = cropped_image.copy()
        h, w = display_image.shape[:2]

        # 縦線
        cv2.line(display_image, (w // 3, 0), (w // 3, h), (0, 255, 0), 1)
        cv2.line(display_image, (2 * w // 3, 0), (2 * w // 3, h), (0, 255, 0), 1)
        # 横線
        cv2.line(display_image, (0, h // 3), (w, h // 3), (0, 255, 0), 1)
        cv2.line(display_image, (0, 2 * h // 3), (w, 2 * h // 3), (0, 255, 0), 1)

        # クロップした画像内のすべての顔に矩形を描画
        # - 元の画像で最も大きかった顔は赤色(0,0,255)で表示
        # - その他の顔は灰色(128,128,128)で表示
        main_face_rel_x = face_center_x - crop_left
        main_face_rel_y = face_center_y - crop_top

        # 最初に赤い点を描画（基準となった顔の中心）
        if 0 <= main_face_rel_x < w and 0 <= main_face_rel_y < h:
            cv2.circle(display_image, (int(main_face_rel_x), int(main_face_rel_y)), 5, (0, 0, 255),
                       -1)

        # 元画像で検出されたすべての顔を表示（デバッグ用）
        for i, (fx, fy, fw, fh) in enumerate(faces):
            if i == 0:  # 最も大きい顔（クロップの基準となった顔）
                # 赤い矩形で表示
                rx = fx - crop_left
                ry = fy - crop_top
                if (0 <= rx < w and 0 <= ry < h and 0 <= rx + fw < w and 0 <= ry + fh < h):
                    cv2.rectangle(display_image, (int(rx), int(ry)), (int(rx + fw), int(ry + fh)),
                                  (0, 0, 255), 2)
            else:
                # その他の顔は灰色で表示
                rx = fx - crop_left
                ry = fy - crop_top
                if (0 <= rx < w and 0 <= ry < h and 0 <= rx + fw < w and 0 <= ry + fh < h):
                    cv2.rectangle(display_image, (int(rx), int(ry)), (int(rx + fw), int(ry + fh)),
                                  (128, 128, 128), 2)

        # クロップした画像内で再検出した顔も表示
        if len(cropped_faces) > 0:
            for i, (fx, fy, fw, fh) in enumerate(cropped_faces):
                # 各顔の中心座標を計算
                face_cx = fx + fw // 2
                face_cy = fy + fh // 2

                # メイン顔との距離を計算
                distance = ((face_cx - main_face_rel_x)**2 + (face_cy - main_face_rel_y)**2)**0.5

                # 基準となった顔に近い顔は赤線、それ以外は灰色線
                # LINE_DASHはOpenCVのバージョンによっては存在しないため、実線を使用
                if distance < max(fw, fh) * 0.5:  # 顔の幅または高さの半分以内の距離なら同一人物と判定
                    cv2.rectangle(display_image, (fx, fy), (fx + fw, fy + fh), (0, 0, 255),
                                  1)  # 赤色線
                else:
                    cv2.rectangle(display_image, (fx, fy), (fx + fw, fy + fh), (128, 128, 128),
                                  1)  # 灰色線

        return display_image
