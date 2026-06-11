import cv2
from flask import Flask, Response
import cv2
from flask import Flask, Response

app = Flask(__name__)

# --- カメラを自動で探して、安全な設定にする関数 ---
def get_working_camera():
    # 0番から4番まで順番に試す
    for i in range(5):
        print(f"テスト中: {i}番のカメラ...")
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        
        if cap.isOpened():
            # 本当に映像が取れるか1枚テスト
            ret, frame = cap.read()
            if ret:
                print(f"大成功！ [{i}番] のカメラで接続しました！")
                # 画質を安全なサイズ(640x480)に固定（ここでパンクを防ぐ）
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
            else:
                cap.release()
    return None

# カメラを準備
cap = get_working_camera()

if cap is None:
    print("エラー: どの番号でも映像を引き出せませんでした。")
    exit()

# --- 映像をブラウザに送る仕組み ---
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("起動完了！Windowsのブラウザで http://pi.local:5000 にアクセスしてください。")
    app.run(host='0.0.0.0', port=5000)
