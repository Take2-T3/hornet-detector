import cv2
import time
import os
import glob
from datetime import datetime
import numpy as np
import onnxruntime as ort
import requests

# ====== 設定項目 ======
SAVE_DIR = "captures"      # 画像保存フォルダ
INTERVAL = 0.5             # 0.5秒おきに撮影
MAX_IMAGES = 1200          # 10分間（1200枚）保持
MODEL_PATH = "best.onnx"   # ONNXファイルのファイル名
IMG_SIZE = 640             # 画像サイズ（YOLOは基本640）
CONF_THRESHOLD = 0.20      # 蜂の認識自信度（20%以上で検知）
WEBHOOK_URL = "https://discord.com/api/webhooks/1506874693947363328/dL9r2sDhpzFtv3LG5-8Y5sILVmXokLdRVHQeUlFi93_-c1w9KlGcBJomZO7mOKPnh5dE"

# --- 【追加】運用・防衛設定 ---
ALERT_COOLDOWN = 60        # 通知のクールダウン時間（秒）※60秒＝1分
HEARTBEAT_HOUR = "08"      # 生存報告を送る時間（時）
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

# --- 【追加】状態記憶用の変数 ---
last_alert_time = 0
last_heartbeat_date = ""

try:
    while True:
        ret, frame = cap.read()
        
        if ret:
            # --- 情報の取得 ---
            original_h, original_w = frame.shape[:2]
            
            # 【変更】タイムスタンプ生成の基礎となる現在時刻をオブジェクトで取得するように微調整
            current_time_obj = datetime.now()
            timestamp = current_time_obj.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
            
            # --- 【追加】画像への時刻スタンプ焼き付け ---
            display_time = current_time_obj.strftime("%Y/%m/%d %H:%M:%S")
            cv2.putText(frame, display_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)       # 黒フチ
            cv2.putText(frame, display_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # 白文字

            # --- 【追加】生存報告（ハートビート） ---
            current_date_str = current_time_obj.strftime("%Y-%m-%d")
            current_hour_str = current_time_obj.strftime("%H")
            if current_hour_str == HEARTBEAT_HOUR and last_heartbeat_date != current_date_str:
                try:
                    payload = {"content": f"🤖 【生存報告】{current_date_str} {current_hour_str}:00 - システムは正常に稼働中です。"}
                    requests.post(WEBHOOK_URL, data=payload)
                    print("✅ 生存報告をDiscordに送信しました。")
                    last_heartbeat_date = current_date_str
                except Exception as e:
                    print(f"❌ 生存報告エラー: {e}")
            
            # --- 【脳】AIによるハチ判定（計算） ---
            img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            img_transposed = np.transpose(img_normalized, (2, 0, 1))
            input_tensor = np.expand_dims(img_transposed, axis=0)

            outputs = session.run(None, {input_name: input_tensor})
            predictions = outputs[0]
            
            # 確率と、確率が一番高かった場所（インデックス）を特定
            scores = predictions[0][4:, :]
            max_confidence = np.max(scores)
            max_box_idx = np.argmax(np.max(scores, axis=0))
            
            # --- 【目】判定結果に応じた処理と描画 ---
            if max_confidence >= CONF_THRESHOLD:
                # 🐝閾値を超えた場合：名前の最後に確率を追加
                filename = os.path.join(SAVE_DIR, f"{timestamp}_conf{max_confidence*100:.1f}.jpg")
                
                # ボックスの座標を計算し、元の画像サイズに合わせる
                best_box_coords = predictions[0][:4, max_box_idx]
                x_center, y_center, w, h = best_box_coords
                x1 = int((x_center - w/2) * original_w / IMG_SIZE)
                y1 = int((y_center - h/2) * original_h / IMG_SIZE)
                x2 = int((x_center + w/2) * original_w / IMG_SIZE)
                y2 = int((y_center + h/2) * original_h / IMG_SIZE)

                # 黄色い枠線と「HORNET XX.X%」という文字を画像に直接書き込む
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                label = f"HORNET: {max_confidence*100:.1f}%"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            else:
                # 📸超えていない場合：今まで通りの名前（描画はしない）
                filename = os.path.join(SAVE_DIR, f"{timestamp}.jpg")
            
            # 枠線を描画した（あるいは描画していない）画像を実際に保存する
            cv2.imwrite(filename, frame) 
            
            # --- 画像のローテーション管理 ---
            files = sorted(glob.glob(os.path.join(SAVE_DIR, "*.jpg")))
            if len(files) > MAX_IMAGES:
                os.remove(files[0])
            
            # --- 【口】Discordへ通知 ---
            if max_confidence >= CONF_THRESHOLD:
                print(f"🐝 ⚠️ 警告：ハチを検知しました！！ (確率: {max_confidence*100:.1f}%)")
                
                # --- 【追加】クールダウン判定 ---
                current_unix_time = time.time()
                if current_unix_time - last_alert_time > ALERT_COOLDOWN:
                    try:
                        with open(filename, "rb") as f:
                            image_data = f.read()
                        
                        payload = {"content": f"🐝 ⚠️ 警告：ハチを検知しました！！ (AI自信度: {max_confidence*100:.1f}%)"}
                        file_data = {"file": (filename, image_data, "image/jpeg")}
                        requests.post(WEBHOOK_URL, data=payload, files=file_data)
                        
                        print(f"✅ Discordに警告と画像を送信しました！ ({filename})")
                        last_alert_time = current_unix_time # 【追加】最終通知時間を更新
                    except Exception as e:
                        print(f"❌ Discord通知エラー: {e}")
                else:
                    # --- 【追加】クールダウン中の場合 ---
                    remain_time = int(ALERT_COOLDOWN - (current_unix_time - last_alert_time))
                    print(f"⏳ クールダウン中... 通知をスキップ。（残り {remain_time} 秒）")
            else:
                print(f"📸 {filename} を保存 (ハチなし: 最大確率 {max_confidence*100:.1f}%)")

        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\n🛑 監視システムを安全に停止しました。")

finally:
    cap.release()