from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
from ultralytics import YOLO
import torch
import os
from paddleocr import PaddleOCR
import numpy as np
from collections import Counter, deque
import time
import logging

def init_camera():
    # camera = cv2.VideoCapture(0)
    camera = cv2.VideoCapture("demo.mp4") # 0 pt webcam
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 3)
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
    
    # Use PaddleOCR with English (auto-download optimized models)
    # print("Încărcare PaddleOCR (English optimized for alphanumeric)...")

    # Silence verbose PaddleOCR debug logging from the underlying ppocr logger
    logging.getLogger('ppocr').setLevel(logging.ERROR)
    logging.getLogger('ppocr').propagate = False
    
    ocr_reader = PaddleOCR(
        ocr_version='PP-OCRv3',
        text_detection_model_dir="en_PP-OCRv3_det_slim_infer",
        text_recognition_model_dir='en_PP-OCRv3_rec_slim_infer',
        # use_space_char=True,
        lang='en',
        use_angle_cls=True,
        logging=False,
    )

    print(f"ML incarcat (PaddleOCR)")
    return model, ocr_reader

def generate_ocr(ocr_reader, plate_crop):
    try:
        # PaddleOCR expects BGR image (OpenCV default)
        # Resize if too small for better accuracy
        h, w = plate_crop.shape[:2]
        if h < 64:
            scale = 64 / h
            new_w = int(w * scale)
            plate_crop = cv2.resize(plate_crop, (new_w, 64), interpolation=cv2.INTER_LINEAR)

        # Run PaddleOCR (without cls parameter)
        result = ocr_reader.ocr(plate_crop)
        
        if result is None or len(result) == 0:
            return ''
        
        # Check if first element is None (no text detected)
        if result[0] is None:
            return ''
        
        # PaddleOCR format: [[[bbox, (text, conf)], [bbox, (text, conf)], ...]]
        # result[0] = list of detected text regions
        # Each region: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)]
        plate_text = ''
        
        for region in result[0]:
            if region and len(region) == 2:
                bbox, text_info = region
                text, conf = text_info
                
                # print(f"OCR Debug: text='{text}', conf={conf:.3f}")
                
                # Filter by confidence and alphanumeric only
                if conf > 0.5:  # 50% threshold
                    clean_text = ''.join(c for c in text if c.isalnum()).upper()
                    plate_text += clean_text
        
        # print(f"OCR Final: '{plate_text}'")
        return plate_text if len(plate_text) > 4 else ''
    except Exception as e:
        print(f"OCR Error: {e}")
        import traceback
        traceback.print_exc()
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
        # print(f"OCR: '{plate_text}' ({count}/4 frame-uri)")
        return plate_text
    return ''

def generate_frames(camera, model, ocr_reader, device):
        ret, frame = camera.read()
        if not ret:
            return None

        # Run YOLO detection ONCE per frame
        results = model(
            frame,
            verbose=False,
            conf=0.45,
            iou=0.6,
            device=device,
            half=(device == 'cuda'),
            imgsz=960,
        )
        
        # OCR only on the BEST (highest confidence) detection per frame
        best_box = None
        best_confidence = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            # boxes is a Boxes object, iterate through detected boxes
            for i in range(len(boxes)):
                # Get coordinates - xyxy is a tensor of shape [N, 4]
                xyxy = boxes.xyxy[i]
                if device == 'cuda':
                    xyxy = xyxy.cpu()
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                
                # Get confidence - conf is a tensor of shape [N]
                conf = boxes.conf[i]
                if device == 'cuda':
                    conf = conf.cpu()
                confidence = float(conf.item())
                
                # Get class - cls is a tensor of shape [N]
                cls = boxes.cls[i]
                if device == 'cuda':
                    cls = cls.cpu()
                class_id = int(cls.item())
                class_name = model.names[class_id]
                
                # Only process detections with confidence > 35%
                if confidence < 0.35:
                    continue
                
                # Track the best box for OCR (process only once per frame)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_box = {
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'confidence': confidence,
                        'class_name': class_name
                    }
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with confidence
                    label = f'{class_name}: {confidence:.1%}'
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Process OCR ONLY ONCE per frame on the best detection
        if best_box is not None:
            x1, y1, x2, y2 = best_box['x1'], best_box['y1'], best_box['x2'], best_box['y2']
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1_crop = max(0, x1)
            y1_crop = max(0, y1)
            x2_crop = min(w, x2)
            y2_crop = min(h, y2)
            
            plate_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop].copy()
            if plate_crop.size > 0 and plate_crop.shape[0] > 10 and plate_crop.shape[1] > 10:
                # Run OCR ONCE per frame
                plate_text = generate_ocr(ocr_reader, plate_crop)
                if plate_text:
                    ocr_buffer.append(plate_text)
                else:
                    ocr_buffer.append('')  # Add empty to maintain buffer size
                
                # Get most common plate from buffer
                verified_plate = get_most_common_plate()
                
                if verified_plate:
                    # Draw verified OCR text below the bounding box
                    cv2.putText(frame, verified_plate, (x1, y2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Optimize JPEG encoding
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return buffer