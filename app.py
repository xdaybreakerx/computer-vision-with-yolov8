import numpy as np
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm
import cv2

# YOLO V8 models in order from small/quick/less good - large/slow/more good 
# yolov8n - nano
# yolov8s - small
# yolov8m - medium
# yolov8l - large
# yolov8x - extra large

# models will download on first use if not already installed locally

model = YOLO("yolov8m.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()

def filter_vehicle_detections(detections):
    # Vehicle class IDs (2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck')
    selected_classes = [2, 3, 5, 7]
    detections = detections[detections.confidence > 0.6]
    vehicle_detections = detections[np.isin(detections.class_id, selected_classes)]
    return vehicle_detections

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    # Filter only vehicle detections
    detections = filter_vehicle_detections(detections)
    # Update tracker with vehicle detections
    detections = tracker.update_with_detections(detections)
    # Annotate frame with vehicle detections only
    return box_annotator.annotate(frame.copy(), detections=detections)

def process_video_with_progress(source_path: str, target_path: str, callback, progress_desc: str = "Processing"):
    cap = cv2.VideoCapture(source_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames, desc=progress_desc) as pbar:
        def wrapped_callback(frame: np.ndarray, frame_idx: int) -> np.ndarray:
            result_frame = callback(frame, frame_idx)
            pbar.update(1)
            return result_frame
        
        sv.process_video(
            source_path=source_path,
            target_path=target_path,
            callback=wrapped_callback
        )

process_video_with_progress(
    source_path="./video/source_trimmed.mp4",
    target_path="./video/result.mp4",
    callback=callback
)