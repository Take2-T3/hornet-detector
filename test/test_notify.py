import requests

# 【重要】あなたのDiscordのWebhook URLをここに貼り付けてください
WEBHOOK_URL = "https://discord.com/api/webhooks/1506874693947363328/dL9r2sDhpzFtv3LG5-8Y5sILVmXokLdRVHQeUlFi93_-c1w9KlGcBJomZO7mOKPnh5dE"

data = {
    "content": "🚨 【テスト】教室からの警報：ハチを検知する準備ができました！"
}

print("Discordに通知を送信中...")

response = requests.post(WEBHOOK_URL, json=data)

if response.status_code == 204:
    print("送信成功！スマホのDiscordを確認してください。")
else:
    print(f"エラーが発生しました。ステータスコード: {response.status_code}")
