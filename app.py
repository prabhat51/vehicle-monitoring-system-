import gradio as gr
import cv2
from inference import VehiclePipeline

pipeline = VehiclePipeline()

def process_video(video_file):
    if hasattr(video_file, "name"):
        video_path = video_file.name
    else:
        video_path = video_file

    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        output = pipeline.process_frame(frame)
        if output is None:
            continue
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        frames.append(output_rgb)
    cap.release()
    
    return frames[-1] if frames else None

gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Vehicle Video", format="mp4"),
    outputs=gr.Image(type="numpy", label="Last Processed Frame"),
    title="Vehicle Monitoring System"
).launch(share=True)
