import os
import onnxruntime as ort
import numpy as np
import cv2
from src.config import MODEL_PATH, IMG_SIZE

class HornetDetector:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"AIモデル '{MODEL_PATH}' が見つかりません。")
        self.session = ort.InferenceSession(MODEL_PATH)
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame):
        original_h, original_w = frame.shape[:2]
        
        # 画像の前処理
        img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        input_tensor = np.expand_dims(img_transposed, axis=0)

        # AIの推論実行
        outputs = self.session.run(None, {self.input_name: input_tensor})
        predictions = outputs[0]
        
        # 結果の解析
        scores = predictions[0][4:, :]
        max_confidence = np.max(scores)
        max_box_idx = np.argmax(np.max(scores, axis=0))
        
        best_box_coords = predictions[0][:4, max_box_idx]
        x_center, y_center, w, h = best_box_coords
        
        # 元の画像サイズにおける座標を計算
        x1 = int((x_center - w/2) * original_w / IMG_SIZE)
        y1 = int((y_center - h/2) * original_h / IMG_SIZE)
        x2 = int((x_center + w/2) * original_w / IMG_SIZE)
        y2 = int((y_center + h/2) * original_h / IMG_SIZE)

        return max_confidence, x1, y1, x2, y2
