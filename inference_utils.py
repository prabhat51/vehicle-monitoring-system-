# inference_utils.py
import os, cv2, re
import torch
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
from paddleocr import PaddleOCR
from difflib import get_close_matches

# Load models from Hugging Face
def load_models():
    vehicle_detector = YOLO("https://huggingface.co/Prabhat51/veh-detect/resolve/main/veh_detect.pt")
    vehicle_classifier = YOLO("https://huggingface.co/Prabhat51/veh-class/resolve/main/veh_class.pt")
    plate_detector = YOLO("https://huggingface.co/Prabhat51/plate-detect/resolve/main/plate_detect.pt")
    ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')
    return vehicle_detector, vehicle_classifier, plate_detector, ocr_reader

# Validate Indian number plate
valid_rto_codes = { ... }  # use your RTO set here

def correct_plate_text(text):
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    text = text.replace('O', '0').replace('I', '1')
    match = re.match(r'^([A-Z]{2})([0-9]{2})([A-Z]{1,2})([0-9]{3,4})$', text)
    if match and match.group(1) in valid_rto_codes:
        return text
    return None

# Inference on single frame
def process_frame(frame, vehicle_detector, vehicle_classifier, plate_detector, ocr_reader):
    results = []
    detections = vehicle_detector(frame)[0].boxes
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        vehicle_crop = frame[y1:y2, x1:x2]

        cls_result = vehicle_classifier(vehicle_crop)
        if not cls_result[0].probs:
            continue
        vehicle_type = cls_result[0].names[cls_result[0].probs.top1]

        plate_boxes = plate_detector(vehicle_crop)[0].boxes
        for pb in plate_boxes:
            px1, py1, px2, py2 = map(int, pb.xyxy[0])
            plate_crop = vehicle_crop[py1:py2, px1:px2]

            ocr_result = ocr_reader.ocr(plate_crop, cls=True)
            if not ocr_result or not ocr_result[0]:
                continue

            raw_text = ocr_result[0][0][1][0]
            plate_text = correct_plate_text(raw_text)
            if not plate_text:
                continue

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results.append((timestamp, vehicle_type, plate_text))
    return results
