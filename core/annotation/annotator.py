import cv2
from .entity_annotator import EntityAnnotator
from utils.logger import logger
from constants.visual_consts import PLAYER_COLOR, BALL_COLOR

class Annotator:
    def __init__(self):
        logger.info("Initializing Video Annotator...")
        self.entity_annotator = EntityAnnotator()
        # self.stats_annotator = StatsAnnotator() # To be implemented in Phase 4 (speeds/distances)
    
    def draw_annotations(self, video_frames, tracks):
        logger.info("Drawing visual annotations onto video frames...")
        output_video_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            # Use .get() safely in case a frame is missing tracking data
            player_dict = tracks.get("players", [])[frame_num]
            ball_dict = tracks.get("ball", [])[frame_num]
            
            # 1. Draw Players (Ellipse)
            for track_id, player in player_dict.items():
                frame = self.entity_annotator.draw_ellipse(frame, player["bbox"], PLAYER_COLOR, track_id)
            
            # 2. Draw Ball (Triangle)
            for track_id, ball in ball_dict.items():
                frame = self.entity_annotator.draw_triangle(frame, ball["bbox"], BALL_COLOR)

            # TODO: Phase 4 - Draw Court Keypoints and Player Speeds Overlay

            output_video_frames.append(frame)
            
        logger.info("Annotation processing complete.")
        return output_video_frames