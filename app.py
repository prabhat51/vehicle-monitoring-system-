# repo structure:
# .
# ├── app.py
# ├── models/                 <- will be created at runtime
# └── requirements.txt

import os
import urllib.request
import cv2
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re
from difflib import get_close_matches

# --------------------- DOWNLOAD MODELS FROM HUGGINGFACE ---------------------
HUGGINGFACE_BASE = "https://huggingface.co/Prabhat51/number-plate-models/resolve/main"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

model_files = [
    "veh_detect.pt",
    "veh_class.pt",
    "plate_detect.pt",
    "ocr_infer.pdparams"
]

def download_model(filename):
    url = f"{HUGGINGFACE_BASE}/{filename}"
    path = os.path.join(model_dir, filename)
    if not os.path.isfile(path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, path)
        print(f"Saved to {path}")

for model_file in model_files:
    download_model(model_file)

# --------------------- LOAD MODELS ---------------------
vehicle_detector = YOLO(os.path.join(model_dir, "veh_detect.pt"))
vehicle_classifier = YOLO(os.path.join(model_dir, "veh_class.pt"))
plate_detector = YOLO(os.path.join(model_dir, "plate_detect.pt"))
ocr_reader = PaddleOCR(rec_model_dir=model_dir)

# --------------------- INFERENCE UTILS ---------------------
valid_rto_codes = {'AP','AR','AS','BR','CH','CG','DD','DL','DN','GA','GJ','HP','HR','JH','JK','KA','KL','LD','MH','ML','MN','MP','MZ','NL','OD','PB','PY','RJ','SK','TN','TR','TS','UK','UP','WB'}

def correct_common_ocr_errors(text):
    text = text.upper().replace('D','0').replace('O','0').replace('I','1').replace('Z','2')
    return re.sub(r'[^A-Z0-9]', '', text)

def fuzzy_correct_plate(text):
    text = correct_common_ocr_errors(text)
    match = re.match(r'^([A-Z]{2})([0-9]{2})([A-Z]{1,2})([0-9]{3,4})$', text)
    if not match:
        return None
    state, district, series, num = match.groups()
    if state not in valid_rto_codes:
        corrected_state = get_close_matches(state, valid_rto_codes, n=1, cutoff=0.6)
        state = corrected_state[0] if corrected_state else None
    return f"{state}{district}{series}{num}" if state else None

def is_valid_indian_plate(text):
    match = re.match(r'^([A-Z]{2})([0-9]{2})([A-Z]{1,2})([0-9]{3,4})$', text)
    return bool(match) and match.group(1) in valid_rto_codes

def clean_plate_text(raw_text):
    corrected = fuzzy_correct_plate(raw_text)
    return corrected if corrected and is_valid_indian_plate(corrected) else None

# --------------------- MAIN FUNCTION ---------------------
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    log_path = os.path.join("logs", "logs.csv")
    os.makedirs("logs", exist_ok=True)
    recent_logs = {}
    frame_skip = 5
    frame_count = 0

    df_log = pd.DataFrame(columns=["timestamp", "vehicle_type", "plate_text"])
    df_log.to_csv(log_path, index=False)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        detections = vehicle_detector(frame)

        for det in detections[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            vehicle_crop = frame[y1:y2, x1:x2]
            v_cls_result = vehicle_classifier(vehicle_crop)

            if v_cls_result[0].probs is None:
                continue

            vehicle_cls_index = v_cls_result[0].probs.top1
            vehicle_type = v_cls_result[0].names[vehicle_cls_index]

            plate_result = plate_detector(vehicle_crop)
            if len(plate_result[0].boxes) == 0:
                continue

            for plate_box in plate_result[0].boxes:
                px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                plate_crop = vehicle_crop[py1:py2, px1:px2]

                ocr_result = ocr_reader.ocr(plate_crop, cls=True)
                if not ocr_result or not ocr_result[0] or not ocr_result[0][0]:
                    continue

                raw_text = ocr_result[0][0][1][0].strip()
                plate_text = clean_plate_text(raw_text)

                if not plate_text:
                    continue

                now = datetime.now()
                if plate_text in recent_logs and (now - recent_logs[plate_text]).total_seconds() < 60:
                    continue

                recent_logs[plate_text] = now
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                df_log = pd.DataFrame([[timestamp, vehicle_type, plate_text]],
                                      columns=["timestamp", "vehicle_type", "plate_text"])
                df_log.to_csv(log_path, mode='a', header=False, index=False)

                cv2.rectangle(frame, (x1+px1, y1+py1), (x1+px2, y1+py2), (0, 255, 0), 2)
                cv2.putText(frame, f"{vehicle_type}: {plate_text}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# To run:
# process_video("test_video.mp4")
