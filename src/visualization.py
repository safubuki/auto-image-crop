import cv2


class Visualizer:
    """
    画像の可視化機能を提供するクラス
    
    デバッグ情報（顔の検出結果、三分割グリッド線など）を
    画像上に描画するための機能を提供します。
    """

    def draw_debug_info(self,
                        cropped_image,
                        faces,
                        crop_left,
                        crop_top,
                        face_center_x,
                        face_center_y,
                        cropped_faces,
                        scored_faces=None):
        """
        デバッグ情報を画像に描画するメソッド
        
        クロップされた画像に三分割線、検出された顔の矩形、
        顔の中心点などのデバッグ情報を描画します。
        
        引数:
            cropped_image: クロップされた画像（OpenCV形式のndarray）
            faces: 元の画像で検出された顔のリスト（各顔は(x, y, w, h)のタプル）
            crop_left: クロップ開始位置（左）
            crop_top: クロップ開始位置（上）
            face_center_x: 元画像内の顔の中心X座標
            face_center_y: 元画像内の顔の中心Y座標
            cropped_faces: クロップした画像内で検出された顔のリスト
            scored_faces: 顔のスコア情報のリスト（各要素は辞書型 {face, score, ...}）
            
        戻り値:
            デバッグ情報が描画された画像
        """
        # デバッグ用：三分割のグリッドと顔矩形を表示
        display_image = cropped_image.copy()
        h, w = display_image.shape[:2]

        # 縦線（三分割の垂直線）
        cv2.line(display_image, (w // 3, 0), (w // 3, h), (0, 255, 0), 1)
        cv2.line(display_image, (2 * w // 3, 0), (2 * w // 3, h), (0, 255, 0), 1)
        # 横線（三分割の水平線）
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

        # スコア情報が利用可能かどうかをチェック
        has_scores = scored_faces is not None and len(scored_faces) > 0

        # 顔のスコア情報を紐づけるための辞書
        face_score_map = {}
        if has_scores:
            for i, face_data in enumerate(scored_faces):
                face = face_data['face']
                face_key = (face[0], face[1], face[2], face[3])  # タプルをキーとして使用
                face_score_map[face_key] = {'index': i, 'score': face_data['score']}

        # 元画像で検出されたすべての顔を表示（デバッグ用）
        for i, (fx, fy, fw, fh) in enumerate(faces):
            rx = fx - crop_left
            ry = fy - crop_top

            # クロップ範囲内に顔が含まれるかチェック
            if (0 <= rx < w and 0 <= ry < h and rx + fw > 0 and ry + fh > 0):
                # 矩形の描画位置を調整（画面内に収める）
                draw_x1 = max(0, rx)
                draw_y1 = max(0, ry)
                draw_x2 = min(w - 1, rx + fw)
                draw_y2 = min(h - 1, ry + fh)

                # 最も高いスコアの顔（クロップの基準となった顔）は赤、それ以外は灰色
                face_key = (fx, fy, fw, fh)
                is_highest_score = False

                if has_scores and face_key in face_score_map:
                    # この顔のスコア情報を取得
                    score_info = face_score_map[face_key]

                    # 最もスコアが高い顔かどうかを判断
                    is_highest_score = True
                    for other_key, other_info in face_score_map.items():
                        if other_info['score'] > score_info['score']:
                            is_highest_score = False
                            break

                color = (0, 0, 255) if is_highest_score else (128, 128, 128)
                cv2.rectangle(display_image, (int(draw_x1), int(draw_y1)),
                              (int(draw_x2), int(draw_y2)), color, 2)

                # スコア情報を表示
                face_key = (fx, fy, fw, fh)
                if has_scores and face_key in face_score_map:
                    score_info = face_score_map[face_key]
                    score_text = f"No.{score_info['index']+1} {score_info['score']:.2f}"

                    # 文字の配置位置（右下）
                    text_x = int(draw_x2) - 10
                    text_y = int(draw_y2) - 10

                    # フォントサイズを約3倍に拡大
                    font_scale = 1.2  # 0.4から1.2に変更 (3倍)
                    text_thickness = 2  # 1から2に変更（太さも増加）
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # テキストのサイズを取得
                    (text_width, text_height), _ = cv2.getTextSize(score_text, font, font_scale,
                                                                   text_thickness)

                    # 文字が見やすいように背景を描画（サイズを調整）
                    cv2.rectangle(display_image,
                                  (text_x - text_width - 5, text_y - text_height - 5),
                                  (text_x + 5, text_y + 10), (0, 0, 0), -1)  # 黒色の背景

                    # テキストを描画（右揃え）
                    cv2.putText(
                        display_image,
                        score_text,
                        (text_x - text_width, text_y),
                        font,
                        font_scale,
                        (255, 255, 255),  # 白色のテキスト
                        text_thickness,
                        cv2.LINE_AA)

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
