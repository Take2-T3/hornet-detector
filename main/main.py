import cv2
import time
import os
import glob
from datetime import datetime
import numpy as np           # 【追加】計算用のライブラリ
import onnxruntime as ort    # 【追加】AIを動かすライブラリ
import requests              # ⚠️【超重要】Discordと通信するためのライブラリ！

# ====== 設定項目 ======
SAVE_DIR = "captures"      # 画像保存フォルダ
INTERVAL = 0.5             # 0.5秒おきに撮影
MAX_IMAGES = 1200          # 10分間（1200枚）保持
MODEL_PATH = "best.onnx"  # ONNXファイルのファイル名
IMG_SIZE = 640             # 画像サイズ（YOLOは基本640）
CONF_THRESHOLD = 0.20      # 蜂の認識自信度
WEBHOOK_URL = "https://discord.com/api/webhooks/1506874693947363328/dL9r2sDhpzFtv3LG5-8Y5sILVmXokLdRVHQeUlFi93_-c1w9KlGcBJomZO7mOKPnh5dE"
# ======================

# 1. フォルダの準備
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 2. AI（脳）の起動
print("🧠 AIモデルを読み込み中...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ エラー: '{MODEL_PATH}' が見つかりません！フォルダに置いてください。")
    exit()
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
print("✅ AIの準備完了！")

# 3. カメラ（目）の準備
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ エラー: カメラに接続できませんでした。")
    exit()

print(f"👁️ {INTERVAL}秒おきの監視システムを開始します。（停止: Ctrl + C）")

try:
    while True:
        ret, frame = cap.read()
        
        if ret:
            # --- 【目】画像の保存とローテーション ---
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = os.path.join(SAVE_DIR, f"{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            
            files = sorted(glob.glob(os.path.join(SAVE_DIR, "*.jpg")))
            if len(files) > MAX_IMAGES:
                os.remove(files[0])
            
            # --- 【脳】AIによるハチ判定（YOLO専用の前処理） ---
            img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            img_transposed = np.transpose(img_normalized, (2, 0, 1))
            input_tensor = np.expand_dims(img_transposed, axis=0)

            # AIに推論（計算）させる！
            outputs = session.run(None, {input_name: input_tensor})
            
            predictions = outputs[0]
            max_confidence = np.max(predictions[0][4:, :]) 
            
            # 確率が設定値（50%）を超えていたらハチと判定！
            if max_confidence >= CONF_THRESHOLD:
                print(f"🐝 ⚠️ 警告：ハチを検知しました！！ (確率: {max_confidence*100:.1f}%)")
                
                # --- 【口】Discordへ通知 ---
                try:
                    with open(filename, "rb") as f:
                        image_data = f.read()
                    
                    # メッセージと画像を用意して送信！
                    payload = {"content": f"🐝 ⚠️ 警告：ハチを検知しました！！ (AI自信度: {max_confidence*100:.1f}%)"}
                    file_data = {"file": (filename, image_data, "image/jpeg")}
                    requests.post(WEBHOOK_URL, data=payload, files=file_data)
                    
                    print("✅ Discordに警告と画像を送信しました！")
                except Exception as e:
                    print(f"❌ Discord通知エラー: {e}")
                
            else:
                print(f"📸 {filename} を保存 (ハチなし: 最大確率 {max_confidence*100:.1f}%)")

        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\n🛑 監視システムを安全に停止しました。")

finally:
    cap.release()