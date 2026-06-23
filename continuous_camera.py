import cv2
import time
import os
import glob
from datetime import datetime

# ====== 設定項目 ======
SAVE_DIR = "captures"      # 画像を保存するフォルダ名
MAX_IMAGES = 600           # 保存する最大枚数（1秒に1枚なら、600枚＝10分間）
# ======================

# 保存用フォルダがなければ作る
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("エラー: カメラに接続できませんでした。")
    exit()

print(f"1秒おきに撮影を開始します。（直近 {MAX_IMAGES}枚（約10分間）のみ保持）")
print("終了するには Ctrl + C を押してください。")

try:
    while True:
        ret, frame = cap.read()
        
        if ret:
            # 現在の時刻をファイル名にする（例: captures/20260623_170530.jpg）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SAVE_DIR, f"{timestamp}.jpg")
            
            # 画像を保存
            cv2.imwrite(filename, frame)
            
            # --- ここから「古いファイルを消す」魔法の仕組み ---
            # フォルダ内のJPGファイルを古い順（名前順）にリストアップ
            files = sorted(glob.glob(os.path.join(SAVE_DIR, "*.jpg")))
            
            # もし設定枚数を超えていたら、一番古いファイルを削除
            if len(files) > MAX_IMAGES:
                oldest_file = files[0]
                os.remove(oldest_file)
                print(f"📸 撮影: {filename} を保存 / ♻️ 古い削除: {oldest_file}")
            else:
                print(f"📸 撮影: {filename} を保存 (現在 {len(files)}/{MAX_IMAGES}枚)")
                
        time.sleep(1.0)

except KeyboardInterrupt:
    print("\n監視カメラを安全に停止しました。")

finally:
    cap.release()
