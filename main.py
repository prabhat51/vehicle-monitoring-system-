import os
import cv2
import pandas as pd
from datetime import datetime
from utils.detection import vehicle_detector, plate_detector
from utils.classification import vehicle_classifier
from utils.ocr_utils import ocr_reader, clean_plate_text
from utils.logger import log_entry

# Setup paths
log_dir = "content"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "logs.csv")

# Initialize logging
if not os.path.exists(log_path):
    pd.DataFrame(columns=["timestamp", "vehicle_type", "plate_text"]).to_csv(log_path, index=False)

# Video
video_path = "sample_video.mp4"
cap = cv2.VideoCapture(video_path)

frame_skip = 5
frame_count = 0
recent_logs = {}

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
            log_entry(log_path, now, vehicle_type, plate_text)

            cv2.rectangle(frame, (x1+px1, y1+py1), (x1+px2, y1+py2), (0, 255, 0), 2)
            cv2.putText(frame, f"{vehicle_type}: {plate_text}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Uncomment to view
    # cv2.imshow("Result", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
