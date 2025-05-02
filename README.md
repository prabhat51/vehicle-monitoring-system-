# Vehicle Monitoring System

This project detects vehicles in video, classifies type, detects license plates, and recognizes the number using PaddleOCR.

## Features

- Vehicle detection (YOLO)
- Type classification
- Plate detection
- OCR + cleaning for valid Indian plate formats
- CSV logging

## Run Locally

```bash
git clone https://github.com/prabhat51/vehicle-monitoring-system.git
cd vehicle-monitoring-system
pip install -r requirements.txt
python app.py
