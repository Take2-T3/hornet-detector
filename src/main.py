import time
import os
from datetime import datetime

# 作成した部品（モジュール）をインポート
from src.config import INTERVAL, CONF_THRESHOLD, SAVE_DIR
from src.camera import CameraManager
from src.detector import HornetDetector
from src.notifier import DiscordNotifier

def main():
    print("🚀 システムを起動しています...")
    
    # 部品の初期化
    camera = CameraManager()
    detector = HornetDetector()
    notifier = DiscordNotifier()

    print(f"👁️ {INTERVAL}秒おきの監視システムを開始します。（停止: Ctrl + C）")

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret:
                continue

            current_time_obj = datetime.now()
            timestamp = current_time_obj.strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
            display_time = current_time_obj.strftime("%Y/%m/%d %H:%M:%S")

            # 画像に時刻を書き込む
            frame = camera.draw_time(frame, display_time)

            # 生存報告チェック
            notifier.send_heartbeat(current_time_obj)

            # AIでハチを検知
            confidence, x1, y1, x2, y2 = detector.detect(frame)

            if confidence >= CONF_THRESHOLD:
                # 検知時の処理
                filename = os.path.join(SAVE_DIR, f"{timestamp}_conf{confidence*100:.1f}.jpg")
                frame = camera.draw_bbox(frame, x1, y1, x2, y2, confidence)
                camera.save_image(frame, filename)
                
                print(f"🐝 ⚠️ ハチ検知！ (確率: {confidence*100:.1f}%)")
                notifier.send_alert(filename, confidence)
            else:
                # 未検知時の処理
                filename = os.path.join(SAVE_DIR, f"{timestamp}.jpg")
                camera.save_image(frame, filename)
                print(f"📸 {filename} を保存 (ハチなし: 最大確率 {confidence*100:.1f}%)")

            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("\n🛑 監視システムを安全に停止しました。")
    finally:
        camera.release()

if __name__ == "__main__":
    main()