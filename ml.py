from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
from ultralytics import YOLO
import torch
import os
import easyocr

def init_camera():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 5)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return camera

def init_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device

def init_lpr(model_path, device):
    model_path = 'license_plate_detector.pt'
    model = YOLO(model_path)
    model.to(device)
    ocr_reader = easyocr.Reader(['ro', 'en'], gpu=(device == 'cuda'))
    print("ML incarcat")
    return model, ocr_reader

def generate_ocr(ocr_reader, plate_crop):
    ocr_results = ocr_reader.readtext(
        plate_crop, 
        detail=1,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    high_conf_texts = [text for (bbox, text, conf) in ocr_results if conf > 0.98]
    plate_text = ''.join(high_conf_texts).replace(' ', '') if high_conf_texts else ''
    return plate_text

def generate_frames(camera, model, ocr_reader, device):
        ret, frame = camera.read()
        if not ret:
            return None

        # Run detection with optimized/tunable settings
        results = model(
            frame,
            verbose=False,
            conf=0.45,
            iou=0.6,
            device=device,
            half=(device == 'cuda'),
            imgsz=1920,
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
                        # Request detailed results to get confidence scores
                        plate_text = generate_ocr(ocr_reader, plate_crop)
                        if plate_text:
                            # Draw OCR text below the bounding box
                            cv2.putText(frame, plate_text, (x1, y2+20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Optimize JPEG encoding
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer