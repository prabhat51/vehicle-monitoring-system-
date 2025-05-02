import gradio as gr
from utils.inference_utils import run_pipeline

def inference(img):
    output_img, plate_text = run_pipeline(img)
    return output_img, plate_text

app = gr.Interface(
    fn=inference,
    inputs=gr.Image(type="numpy", label="Upload Vehicle Vedio"),
    outputs=[gr.Image(label="Detected Image"), gr.Textbox(label="Plate Number")],
    title="Vehicle Monitoring System",
    description="Vehicle Detection, Classification, Number Plate Recognition using PaddleOCR"
)

if __name__ == "__main__":
    app.launch()
