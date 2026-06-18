import cv2

# カメラの準備（0はパソコンに内蔵されている標準カメラを指します）
# もしUSBカメラなどを繋いでいて映らない場合は、0を1や2に変更してみてください
cap = cv2.VideoCapture(0)

print("カメラを起動しました。終了するには映像のウィンドウを選択した状態で「q」キーを押してください。")

# 無限ループで映像のコマ（フレーム）を取得し続ける
while True:
    # カメラから1コマ分の画像を読み込む
    # ret: 正しく読み込めたかどうかの結果（True/False）
    # frame: 実際の画像データ
    ret, frame = cap.read()

    # 画像が正しく読み込めなかった場合はループを抜ける
    if not ret:
        print("映像を取得できませんでした。")
        break

    # ウィンドウを開いて、取得した画像（frame）を表示する
    cv2.imshow('Camera Capture', frame)

    # 1ミリ秒だけ待機し、その間にキーボードの「q」が押されたらループを終了する
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("終了します...")
        break

# 使い終わったカメラを解放して、ウィンドウをすべて閉じる（後片付け）
cap.release()
cv2.destroyAllWindows()