import os
from dotenv import load_dotenv

# .envファイルから秘密のURLを読み込む
load_dotenv()
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# --- 設定項目 ---
SAVE_DIR = "captures"      # 画像保存フォルダ
INTERVAL = 0.5             # 撮影間隔（秒）
MAX_IMAGES = 1200          # 保持する最大画像数
MODEL_PATH = "models/best.onnx" # AIモデルの場所
IMG_SIZE = 640             # AIの入力画像サイズ
CONF_THRESHOLD = 0.20      # 蜂の認識自信度（20%以上）

ALERT_COOLDOWN = 60        # 通知のクールダウン時間（秒）
HEARTBEAT_HOUR = "08"      # 生存報告を送る時間（時）