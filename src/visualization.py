import cv2
import numpy as np


class Visualizer:
    """
    画像の可視化機能を提供するクラス
    
    デバッグ情報（顔の検出結果、三分割グリッド線など）を
    画像上に描画するための機能を提供します。
    """

    def draw_debug_info(self, cropped_image, split_method='thirds'):  # 分割方法のみ引数として残す
        """
        デバッグ情報（分割線）をクロップ後の画像に描画するメソッド

        Args:
            cropped_image: クロップされた画像
            split_method (str): 分割方法 ('thirds' または 'phi')

        Returns:
            デバッグ情報が描画された画像
        """
        display_image = cropped_image.copy()
        h, w = display_image.shape[:2]

        # 分割線を描画
        if split_method == 'thirds':
            # 三分割法
            cv2.line(display_image, (w // 3, 0), (w // 3, h), (0, 255, 0), 1)
            cv2.line(display_image, (2 * w // 3, 0), (2 * w // 3, h), (0, 255, 0), 1)
            cv2.line(display_image, (0, h // 3), (w, h // 3), (0, 255, 0), 1)
            cv2.line(display_image, (0, 2 * h // 3), (w, 2 * h // 3), (0, 255, 0), 1)
        elif split_method == 'phi':
            # ファイグリッド (黄金比)
            phi = (1 + np.sqrt(5)) / 2
            total_ratio = 2 + (phi - 1)
            line1_x = int(w / total_ratio)
            line2_x = int(w * (1 + (phi - 1)) / total_ratio)
            line1_y = int(h / total_ratio)
            line2_y = int(h * (1 + (phi - 1)) / total_ratio)

            cv2.line(display_image, (line1_x, 0), (line1_x, h), (0, 255, 0), 1)
            cv2.line(display_image, (line2_x, 0), (line2_x, h), (0, 255, 0), 1)
            cv2.line(display_image, (0, line1_y), (w, line1_y), (0, 255, 0), 1)
            cv2.line(display_image, (0, line2_y), (w, line2_y), (0, 255, 0), 1)
        else:
            # 不明な場合は三分割法を描画
            cv2.line(display_image, (w // 3, 0), (w // 3, h), (0, 255, 0), 1)
            cv2.line(display_image, (2 * w // 3, 0), (2 * w // 3, h), (0, 255, 0), 1)
            cv2.line(display_image, (0, h // 3), (w, h // 3), (0, 255, 0), 1)
            cv2.line(display_image, (0, 2 * h // 3), (w, 2 * h // 3), (0, 255, 0), 1)

        # クロップ範囲を示す青い枠を追加
        cv2.rectangle(display_image, (0, 0), (w - 1, h - 1), (255, 0, 0), 2)  # 青色の矩形

        return display_image

    def draw_face_info(self, original_image, faces, scored_faces=None):
        """
        顔の検出結果とスコア情報を元画像に描画するメソッド
        
        元の画像に検出された顔の矩形と、スコア情報を描画します。
        
        引数:
            original_image: 元画像（OpenCV形式のndarray）
            faces: 検出された顔のリスト（各顔は(x, y, w, h)のタプル）
            scored_faces: 顔のスコア情報のリスト（各要素は辞書型 {face, score, ...}）
            
        戻り値:
            顔情報が描画された元画像
        """
        display_image = original_image.copy()

        # スコア情報が利用可能かどうかをチェック
        has_scores = scored_faces is not None and len(scored_faces) > 0

        # 顔のスコア情報を紐づけるための辞書
        face_score_map = {}
        scored_face_keys = []
        if has_scores:
            for i, face_data in enumerate(scored_faces):
                face = face_data['face']
                face_key = (face[0], face[1], face[2], face[3])  # タプルをキーとして使用
                scored_face_keys.append(face_key)  # スコア付きの顔キーを保存
                face_score_map[face_key] = {
                    'index': i,
                    'score': face_data['score'],
                    'size_score': face_data.get('size_score', 0),
                    'center_score': face_data.get('center_score', 0),
                    'sharpness_score': face_data.get('sharpness_score', 0)
                }

        # 最も高いスコアの顔を特定
        best_face_key = scored_face_keys[0] if has_scores and scored_face_keys else None

        # 検出された顔をすべて描画
        for i, face in enumerate(faces):
            x, y, w, h = face
            face_key = (x, y, w, h)

            # 顔の矩形を描画（最高スコアの顔は赤、それ以外は灰色）
            is_best_face = has_scores and face_key == best_face_key
            color = (0, 0, 255) if is_best_face else (128, 128, 128)  # 赤または灰色
            thickness = 2  # 全ての矩形を2pxに統一

            cv2.rectangle(display_image, (x, y), (x + w, y + h), color, thickness)

            # スコア情報が存在する場合、表示（「No.4 0.567」形式、白の太字に黒の背景）
            if has_scores and face_key in face_score_map:
                score_info = face_score_map[face_key]
                score_text = f"No.{i+1} {score_info['score']:.3f}"

                # 文字の位置（矩形の右下内側に表示）
                font_scale = 1.3  # フォントサイズ
                font = cv2.FONT_HERSHEY_SIMPLEX

                # テキストのサイズと位置を計算
                text_size, baseline = cv2.getTextSize(score_text, font, font_scale, 2)
                text_pos = (x + w - text_size[0] - 5, y + h - 5)

                # 黒い背景を描画
                bg_rect_pt1 = (text_pos[0] - 2, text_pos[1] - text_size[1] - 2)
                bg_rect_pt2 = (text_pos[0] + text_size[0] + 2, text_pos[1] + 2)
                cv2.rectangle(display_image, bg_rect_pt1, bg_rect_pt2, (0, 0, 0), -1)  # 塗りつぶし

                # 白い文字を描画
                cv2.putText(display_image, score_text, text_pos, font, font_scale, (255, 255, 255),
                            2)

        return display_image
