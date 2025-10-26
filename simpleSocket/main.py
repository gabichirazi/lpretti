from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64

app = Flask(__name__)
socketio = SocketIO(app)
camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        _, frame = camera.read()
        _, buffer = cv2.imencode('.jpg', frame)
        socketio.emit('video_frame', {'data': base64.b64encode(buffer).decode()})
        socketio.sleep(0.03)

@socketio.on('connect')
def handle_connect():
    socketio.start_background_task(generate_frames)

if __name__ == '__main__':
    socketio.run(app, port=5000)
