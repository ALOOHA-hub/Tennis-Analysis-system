from core.utils.video_utils import read_video, save_video
from core.utils.logger import logger

class TennisAnalysisPipeline:
    def __init__(self, input_video_path: str, output_video_path: str):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        
        # We will load our YOLO trackers and ResNet court detectors here in Phase 2
        logger.info("Tennis Analysis Pipeline initialized.")

    def run(self):
        logger.info("Starting pipeline execution...")
        
        # Step 1: Read Video
        frames = read_video(self.input_video_path)
        if not frames:
            logger.error("Pipeline aborted: No frames loaded.")
            return
            
        # TODO: Phase 2 - Inject ball and player tracking here
        # TODO: Phase 3 - Inject court detection and perspective transformation here
        # TODO: Phase 4 - Draw bounding boxes and UI stats

        # Step 5: Save Output
        logger.info("Pipeline processing complete. Moving to save step.")
        save_video(frames, self.output_video_path)
        
        logger.info("Pipeline execution completed successfully.")