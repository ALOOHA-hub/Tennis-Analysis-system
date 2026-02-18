from ultralytics import YOLO
from constants import DETECTION_BATCH_SIZE, DETECTION_CONFIDENCE_THRESHOLD

class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, batch_size=DETECTION_BATCH_SIZE, conf=DETECTION_CONFIDENCE_THRESHOLD):
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=conf)
            detections += detections_batch
        return detections