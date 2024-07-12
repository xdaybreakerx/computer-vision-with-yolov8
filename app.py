import numpy as np
import supervision as sv
from ultralytics import YOLO

# YOLO V8 models in order from small/quick/less good - large/slow/more good 

# yolov8n - nano
# yolov8s - small
# yolov8m - medium
# yolov8l - large
# yolov8x - extra large

# models will download on first use if not already installed locally

model = YOLO("./models/yolov8m.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    return box_annotator.annotate(frame.copy(), detections=detections)

sv.process_video(
    source_path="./video/source_trimmed.mp4",
    target_path="./video/result.mp4",
    callback=callback
)
