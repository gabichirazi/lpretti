from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
from ultralytics import YOLO
import torch
import os
import easyocr

app = Flask(__name__)
socketio = SocketIO(app)

# Optimize camera settings
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 5)
camera.set(cv2.CAP_PROP_FPS, 30)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Inference tuning (can also be overridden via env vars)
# Examples: INFER_IMG_SIZE=1280 INFER_CONF=0.5 INFER_IOU=0.6
INFER_IMG_SIZE = int(os.getenv("INFER_IMG_SIZE", "1920"))  # 640 | 960 | 1280
INFER_CONF = float(os.getenv("INFER_CONF", "0.45"))       # 0.35–0.5 typical
INFER_IOU = float(os.getenv("INFER_IOU", "0.6"))          # 0.5–0.7 typical
print(f"YOLO params -> imgsz={INFER_IMG_SIZE}, conf={INFER_CONF}, iou={INFER_IOU}")

# YOLOv8n model with GPU support
model_path = 'license_plate_detector.pt' if os.path.exists('license_plate_detector.pt') else 'yolov8n.pt'
model = YOLO(model_path)
model.to(device)

# EasyOCR Reader for license plate text (Romanian + English)
print("Loading EasyOCR (ro, en)...")
ocr_reader = easyocr.Reader(['ro', 'en'], gpu=(device == 'cuda'))
print("EasyOCR ready")

if model_path == 'yolov8n.pt':
    print("INFO: Using YOLOv8n base model. For license plate detection, add a trained model as 'license_plate_detector.pt'")

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        ret, frame = camera.read()
        if not ret:
            continue
        
        # Run detection with optimized/tunable settings
        results = model(
            frame,
            verbose=False,
            conf=INFER_CONF,
            iou=INFER_IOU,
            device=device,
            half=(device == 'cuda'),
            imgsz=INFER_IMG_SIZE,
        )
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Get coordinates
                    coords = box.xyxy[0]
                    if device == 'cuda':
                        coords = coords.cpu()
                    x1, y1, x2, y2 = map(int, coords)
                    
                    # Get confidence and class
                    conf = box.conf[0]
                    cls = box.cls[0]
                    if device == 'cuda':
                        conf = conf.cpu()
                        cls = cls.cpu()
                    confidence = float(conf)
                    class_id = int(cls)
                    class_name = model.names[class_id]
                    
                    # Only process detections with confidence > 55%
                    if confidence < 0.55:
                        continue
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with confidence
                    label = f'{class_name}: {confidence:.1%}'
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # OCR: crop detected region and read plate text
                    # Ensure coordinates are within frame bounds
                    h, w = frame.shape[:2]
                    x1_crop = max(0, x1)
                    y1_crop = max(0, y1)
                    x2_crop = min(w, x2)
                    y2_crop = min(h, y2)
                    
                    plate_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop].copy()
                    if plate_crop.size > 0 and plate_crop.shape[0] > 10 and plate_crop.shape[1] > 10:
                        # Allowlist: only uppercase letters and digits (typical for license plates)
                        ocr_results = ocr_reader.readtext(
                            plate_crop, 
                            detail=0,
                            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                        )
                        plate_text = ''.join(ocr_results).replace(' ', '') if ocr_results else ''
                        if plate_text:
                            # Draw OCR text below the bounding box
                            cv2.putText(frame, plate_text, (x1, y2+20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Optimize JPEG encoding
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        socketio.emit('video_frame', {'data': base64.b64encode(buffer).decode()})
        socketio.sleep(0.02)

@socketio.on('connect')
def handle_connect():
    socketio.start_background_task(generate_frames)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
