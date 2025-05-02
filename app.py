# app.py
import gradio as gr
import cv2, os, pandas as pd
from inference_utils import load_models, process_frame

vehicle_detector, vehicle_classifier, plate_detector, ocr_reader = load_models()
log_path = "vehicle_log.csv"
if not os.path.exists(log_path):
    pd.DataFrame(columns=["timestamp", "vehicle_type", "plate_text"]).to_csv(log_path, index=False)

def run_pipeline(video):
    cap = cv2.VideoCapture(video)
    logs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (640, 480))
        res = process_frame(frame, vehicle_detector, vehicle_classifier, plate_detector, ocr_reader)
        logs.extend(res)

    df = pd.DataFrame(logs, columns=["timestamp", "vehicle_type", "plate_text"])
    df.to_csv(log_path, mode='a', header=False, index=False)
    return f"Processed {len(logs)} vehicles. Log saved to {log_path}."

iface = gr.Interface(fn=run_pipeline,
                     inputs=gr.Video(label="Upload Video"),
                     outputs="text",
                     title="Vehicle ANPR Pipeline",
                     description="Uploads a video, detects vehicles, classifies type, reads plate and logs to CSV.")

iface.launch()
