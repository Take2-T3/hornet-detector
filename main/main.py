import cv2
import time
import os
import glob
from datetime import datetime
import numpy as np           # 【追加】計算用のライブラリ
import onnxruntime as ort    # 【追加】AIを動かすライブラリ

# ====== 設定項目 ======
SAVE_DIR = "captures"      # 画像保存フォルダ
INTERVAL = 0.5             # 0.5秒おきに撮影
MAX_IMAGES = 1200          # 10分間（1200枚）保持
MODEL_PATH = "model.onnx"  # ONNXファイルのファイル名（変更必要）
IMG_SIZE = 640             #　画像サイズ（YOLOは基本640）
CONF_THRESHOLD = 0.50      # 蜂の認識自信度
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
            # ① AIの好きなサイズ（640x640）に変える
            img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            # ② 色の順番を OpenCVの(青,緑,赤) から AI用の(赤,緑,青) に変える
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            # ③ ピクセルデータ(0〜255)を AIが計算しやすい小数(0.0〜1.0)にする
            img_normalized = img_rgb.astype(np.float32) / 255.0
            # ④ データの並び順をYOLOのルールに合わせる
            img_transposed = np.transpose(img_normalized, (2, 0, 1))
            input_tensor = np.expand_dims(img_transposed, axis=0)

            # ⑤ AIに推論（計算）させる！
            outputs = session.run(None, {input_name: input_tensor})
            
            # ⑥ 結果から一番高い「確率（コンフィデンス）」を抜き出す
            # ※YOLOv8の標準出力形式を想定しています
            predictions = outputs[0]
            max_confidence = np.max(predictions[0][4:, :]) 
            
            # 確率が設定値（70%）を超えていたらハチと判定！
            if max_confidence >= CONF_THRESHOLD:
                print(f"🐝 ⚠️ 警告：ハチを検知しました！！ (確率: {max_confidence*100:.1f}%)")
                
                # --- 【口】Discordへ通知 ---
                # （※ここに test_notify.py の画像を送信するコードを組み込みます）
                
            else:
                print(f"📸 {filename} を保存 (ハチなし: 最大確率 {max_confidence*100:.1f}%)")

        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\n監視システムを停止しました。")

finally:
    cap.release()
