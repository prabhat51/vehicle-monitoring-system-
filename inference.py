import cv2
import os
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.ocr_utils import clean_plate_text
from utils.logger import init_log, append_log

recent_logs = {}

class VehiclePipeline:
    def __init__(self, model_dir="models"):
        self.detector = YOLO(os.path.join(model_dir, "veh_detect.pt"))
        self.classifier = YOLO(os.path.join(model_dir, "veh_class.pt"))
        self.plate_detector = YOLO(os.path.join(model_dir, "plate_detect.pt"))
        self.ocr = PaddleOCR(rec_model_dir=os.path.join(model_dir, "paddleocr"), use_angle_cls=True)
        self.log_path = init_log()

    def process_frame(self, frame):
        detections = self.detector(frame)
        for det in detections[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            vehicle_crop = frame[y1:y2, x1:x2]

            cls_result = self.classifier(vehicle_crop)
            if not cls_result[0].probs:
                continue
            vehicle_type = cls_result[0].names[cls_result[0].probs.top1]

            plate_result = self.plate_detector(vehicle_crop)
            if not plate_result[0].boxes:
                continue

            for box in plate_result[0].boxes:
                px1, py1, px2, py2 = map(int, box.xyxy[0])
                plate_crop = vehicle_crop[py1:py2, px1:px2]

                ocr_result = self.ocr.ocr(plate_crop, cls=True)
                if not ocr_result or not ocr_result[0]:
                    continue
                raw_text = ocr_result[0][0][1][0].strip()
                plate_text = clean_plate_text(raw_text)
                if not plate_text:
                    continue

                now = datetime.now()
                if plate_text in recent_logs and (now - recent_logs[plate_text]).total_seconds() < 60:
                    continue  

                recent_logs[plate_text] = now
                append_log(self.log_path, vehicle_type, plate_text)

                cv2.rectangle(frame, (x1 + px1, y1 + py1), (x1 + px2, y1 + py2), (0, 255, 0), 2)
                cv2.putText(frame, f"{vehicle_type}: {plate_text}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return frame
