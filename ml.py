from flask_socketio import SocketIO
import cv2
from ultralytics import YOLO
import torch
from paddleocr import PaddleOCR
import numpy as np
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

def init_lpr(device):
    model_path = 'license_plate_detector.pt'
    model = YOLO(model_path)
    model.to(device)
    
    logging.getLogger('ppocr').setLevel(logging.ERROR)
    logging.getLogger('ppocr').propagate = False
    
    ocr_reader = PaddleOCR(
        ocr_version='PP-OCRv3',
        text_detection_model_dir="en_PP-OCRv3_det_slim_infer",
        text_recognition_model_dir='en_PP-OCRv3_rec_slim_infer',
        lang='en',
        use_angle_cls=True,
        logging=False,
    )

    print("OCR incarcat")
    return model, ocr_reader

def generate_ocr(ocr_reader, plate_crop):
    try:
        h, w = plate_crop.shape[:2]
        if h < 64:
            scale = 64 / h
            new_w = int(w * scale)
            plate_crop = cv2.resize(plate_crop, (new_w, 64), interpolation=cv2.INTER_LINEAR)

        result = ocr_reader.ocr(plate_crop)
        
        if result is None or len(result) == 0:
            return ''
        
        if result[0] is None:
            return ''
        
        # PaddleOCR format: [[[bbox, (text, conf)], [bbox, (text, conf)], ...]]
        # result[0] = list of detected text regions
        # Each region: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)]
        plate_text = ''
        
        for region in result[0]:
            if region and len(region) == 2:
                _, text_info = region
                text, conf = text_info
                if conf > 0.93: 
                    clean_text = ''.join(c for c in text if c.isalnum()).upper()
                    plate_text += clean_text
        
        # print(f"OCR: '{plate_text}'")
        return plate_text if len(plate_text) > 4 else ''
    except Exception as e:
        print(f"OCR Error: {e}")
        import traceback
        traceback.print_exc()
        return ''

def generate_frames(camera, model, ocr_reader, device):
        ret, frame = camera.read()
        if not ret:
            return None

        # yolo
        results = model(
            frame,
            verbose=False,
            conf=0.45,
            iou=0.6, #factor intersectare peste reuniune
            device=device,
            half=(device == 'cuda'),
            imgsz=960,
        )
        
        # cautam cea mai buna prezicere 1/2
        best_box = None
        best_confidence = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for i in range(len(boxes)):
                # [xyxy, i]
                xyxy = boxes.xyxy[i]
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                
                # [i]
                conf = boxes.conf[i]
                confidence = float(conf.item())
                
                if confidence < 0.65:
                    continue

                # cautam cea mai buna prezicere 2/2
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_box = {
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'confidence': confidence,
                        'class_name': 'Placuta'
                    }
                    
                    # desenam cutia verde in jurul placutei
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # atasam eticheta
                    label = f'Placuta: {confidence:.1%}'
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if best_box is not None:
            x1, y1, x2, y2 = best_box['x1'], best_box['y1'], best_box['x2'], best_box['y2']

            plate_crop = frame[y1:y2, x1:x2].copy()
            if plate_crop.size > 0: #and plate_crop.shape[0] > 10 and plate_crop.shape[1] > 10:
                plate_text = generate_ocr(ocr_reader, plate_crop)
                if plate_text:
                    cv2.putText(frame, plate_text, (x1, y2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Optimize JPEG encoding
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return buffer