from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
from ultralytics import YOLO
import torch
import os
import pytesseract
from PIL import Image
import numpy as np
from collections import Counter, deque

def init_camera():
    # camera = cv2.VideoCapture(0)
    camera = cv2.VideoCapture("demo.mp4") # 0 pt webcam
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
    # Configure Tesseract for license plate OCR
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    print("ML incarcat (Tesseract OCR)")
    return model, None  # No separate OCR reader needed for Tesseract

def generate_ocr(ocr_reader, plate_crop):
    # Preprocess image for better OCR using HSV
    hsv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
    
    # Extract V channel (Value/Brightness) - best for OCR
    _, _, v_channel = cv2.split(hsv)
    
    # Resize if too small (Tesseract works better with larger images)
    h, w = v_channel.shape
    if h < 50 or w < 150:
        scale = max(50 / h, 150 / w, 2.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        v_channel = cv2.resize(v_channel, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Apply threshold to get better contrast
    _, thresh = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Use only PSM 7 (fastest and best for license plates)
    custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    try:
        # Get text directly (fastest approach)
        text = pytesseract.image_to_string(thresh, config=custom_config).strip()
        text = ''.join(c for c in text if c.isalnum()).upper()
        
        return text if len(text) > 4 else ''
    except Exception as e:
        return ''

# Buffer for voting system
ocr_buffer = deque(maxlen=2)

def get_most_common_plate():
    """Return the most common plate from last 5 readings"""
    if not ocr_buffer:
        return ''
    
    # Filter out empty strings
    valid_plates = [p for p in ocr_buffer if p]
    if not valid_plates:
        return ''
    
    # Get most common plate
    counter = Counter(valid_plates)
    most_common = counter.most_common(1)[0]
    plate_text, count = most_common
    
    # Only return if appears at least 2 times (majority voting)
    if count >= 2:
        print(f"OCR: '{plate_text}' ({count}/4 frame-uri)")
        return plate_text
    return ''

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
                    if confidence < 0.35:
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
                        # Run OCR and add to buffer
                        plate_text = generate_ocr(ocr_reader, plate_crop)
                        if plate_text:
                            ocr_buffer.append(plate_text)
                        else:
                            ocr_buffer.append('')  # Add empty to maintain buffer size
                        
                        # Get most common plate from last 5 frames
                        verified_plate = get_most_common_plate()
                        
                        if verified_plate:
                            # Draw verified OCR text below the bounding box
                            cv2.putText(frame, verified_plate, (x1, y2+20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Optimize JPEG encoding
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer