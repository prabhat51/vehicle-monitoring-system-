import gradio as gr
import cv2
from inference import VehiclePipeline

pipeline = VehiclePipeline()

def process_video(video_file):
    cap = cv2.VideoCapture(video_file)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output = pipeline.process_frame(frame)
        frames.append(output)
    cap.release()
    return frames[-1]  # Show final frame

gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Vehicle Video"),
    outputs=gr.Image(type="numpy", label="Last Processed Frame"),
    title="Vehicle Monitoring System"
).launch()
