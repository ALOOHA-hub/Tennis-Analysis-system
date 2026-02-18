from ultralytics import YOLO
from constants import DETECTION_BATCH_SIZE, DETECTION_CONFIDENCE_THRESHOLD
from core.utils.logger import logger

class Detector:
    def __init__(self, model_path):
        logger.info(f"Loading YOLO Detector from {model_path}")
        self.model = YOLO(model_path)

    def detect_frames(self, frames, batch_size=DETECTION_BATCH_SIZE, conf=DETECTION_CONFIDENCE_THRESHOLD):
        logger.info(f"Running detection on {len(frames)} frames with batch size {batch_size}")
        detections = []
        
        for i in range(0, len(frames), batch_size):
            # Log progress for large videos
            if i % (batch_size * 5) == 0 and i > 0:
                logger.info(f"Processed {i}/{len(frames)} frames...")
                
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=conf, verbose=False)
            detections += detections_batch
            
        logger.info("Detection phase complete.")
        return detections