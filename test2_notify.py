import cv2
import time  # 時間を計るためのライブラリ

# カメラの準備
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("エラー: カメラに接続できませんでした。")
    exit()

print("1秒おきに5枚の写真を撮影します...")

# 5回繰り返す
for i in range(1, 6):
    # 1秒間、プログラムを一時停止（待つ）
    time.sleep(1.0)
    
    # カメラの最新のコマを読み込む
    ret, frame = cap.read()
    
    if ret:
        # ファイル名を "photo_1.jpg", "photo_2.jpg" のように変えて保存する
        filename = f"photo_{i}.jpg"
        cv2.imwrite(filename, frame)
        print(f"【{i}枚目】 {filename} を保存しました！")
    else:
        print(f"【{i}枚目】 撮影に失敗しました。")

# 後片付け
cap.release()
print("すべて完了しました。")
