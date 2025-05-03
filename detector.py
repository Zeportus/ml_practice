import cv2
import numpy as np
from roboflow import Roboflow
import supervision as sv
from PIL import Image
import io
import json
import os
from datetime import datetime
import csv


rf = Roboflow(api_key="roboflow_api_key")
project = rf.workspace().project("class-monitoring")
model = project.version(4).model

label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoxAnnotator()

def detect_phones(image_bytes):
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    temp_path = "temp_input.jpg"
    cv2.imwrite(temp_path, image_bgr)

    result = model.predict(temp_path, confidence=40, overlap=30).json()
    filtered_predictions = [item for item in result["predictions"] if item["class"] == 'Using_phone']
    result["predictions"] = filtered_predictions
            
    detections = sv.Detections.from_inference(result)
    labels = [item["class"] for item in result["predictions"]]

    annotated_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    return Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)), len(result["predictions"])

def save_history(file_name, detected, count):
    record = {
        "timestamp": datetime.now().isoformat(),
        "file_name": file_name,
        "detected": detected,
        "num_phones": count
    }

    if os.path.exists("requests_log.json"):
        with open("requests_log.json", "r") as f:
            history = json.load(f)
    else:
        history = []

    history.append(record)

    with open("requests_log.json", "w") as f:
        json.dump(history, f, indent=4)

def export_csv(path="report.csv"):
    if not os.path.exists("requests_log.json"):
        return False

    with open("requests_log.json", "r") as f:
        history = json.load(f)

    with open(path, "w", newline="") as csvfile:
        fieldnames = ["timestamp", "file_name", "detected", "num_phones"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            writer.writerow(record)
    return True
