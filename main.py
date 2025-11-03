from flask import Flask, render_template
from flask_socketio import SocketIO
import base64
import ml

app = Flask(__name__)
socketio = SocketIO(app)
camera = ml.init_camera()
device = ml.init_device()
model, ocr_reader = ml.init_lpr(device)

@app.route('/')
def index():
    return render_template('index.html')

def run_model():
    while True:
        buffer = ml.generate_frames(camera, model, ocr_reader, device)
        if buffer is None:
            print("Video finalizat/eroare camera.")
            return
        else:
            socketio.emit('video_frame', {'data': base64.b64encode(buffer).decode()})
        socketio.sleep(0.033)

@socketio.on('connect')
def handle_connect():
    socketio.start_background_task(run_model)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
