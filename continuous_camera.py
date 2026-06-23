import cv2
import time

# カメラの準備
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("エラー: カメラに接続できませんでした。")
    exit()

print("監視カメラモードを起動します。")
print("※終了するにはキーボードの『Ctrl + C』を押してください！")

try:
    # 無限ループ（Trueである限り永遠に繰り返す）
    while True:
        ret, frame = cap.read()
        
        if ret:
            # 【重要】常に「latest.jpg」という同じ名前に上書き保存する
            cv2.imwrite("latest.jpg", frame)
            print("最新の画像をキャプチャしました！ (停止は Ctrl + C)")
        else:
            print("カメラからの映像取得に失敗しました。")
            
        # 次の撮影まで1秒待つ（ここで間隔を調整できます）
        time.sleep(1.0)

except KeyboardInterrupt:
    # ユーザーが Ctrl + C を押した時の処理
    print("\n監視を終了します。お疲れ様でした！")

finally:
    # 後片付け
    cap.release()