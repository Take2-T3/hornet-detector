import requests
import time
from src.config import WEBHOOK_URL, ALERT_COOLDOWN, HEARTBEAT_HOUR

class DiscordNotifier:
    def __init__(self):
        self.last_alert_time = 0
        self.last_heartbeat_date = ""

    def send_heartbeat(self, current_time_obj):
        current_date_str = current_time_obj.strftime("%Y-%m-%d")
        current_hour_str = current_time_obj.strftime("%H")
        
        if current_hour_str == HEARTBEAT_HOUR and self.last_heartbeat_date != current_date_str:
            try:
                payload = {"content": f"🤖 【生存報告】{current_date_str} {current_hour_str}:00 - システムは正常に稼働中です。"}
                requests.post(WEBHOOK_URL, data=payload)
                print("✅ 生存報告をDiscordに送信しました。")
                self.last_heartbeat_date = current_date_str
            except Exception as e:
                print(f"❌ 生存報告エラー: {e}")

    def send_alert(self, filename, confidence):
        current_unix_time = time.time()
        
        if current_unix_time - self.last_alert_time > ALERT_COOLDOWN:
            try:
                with open(filename, "rb") as f:
                    image_data = f.read()
                
                payload = {"content": f"🐝 ⚠️ 警告：ハチを検知しました！！ (AI自信度: {confidence*100:.1f}%)"}
                file_data = {"file": (filename, image_data, "image/jpeg")}
                requests.post(WEBHOOK_URL, data=payload, files=file_data)
                
                print(f"✅ Discordに警告と画像を送信しました！ ({filename})")
                self.last_alert_time = current_unix_time
            except Exception as e:
                print(f"❌ Discord通知エラー: {e}")
        else:
            remain_time = int(ALERT_COOLDOWN - (current_unix_time - self.last_alert_time))
            print(f"⏳ クールダウン中... 通知をスキップ。（残り {remain_time} 秒）")