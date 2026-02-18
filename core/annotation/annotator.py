import cv2
from .entity_annotator import EntityAnnotator
from utils.logger import logger
from constants.visual_consts import PLAYER_COLOR, BALL_COLOR
from .mini_court import MiniCourt

class Annotator:
    def __init__(self):
        logger.info("Initializing Video Annotator...")
        self.entity_annotator = EntityAnnotator()
        self.mini_court = None
        # self.stats_annotator = StatsAnnotator() # To be implemented in Phase 4 (speeds/distances)
    
    def draw_annotations(self, video_frames, tracks, court_keypoints=None):
        logger.info("Drawing visual annotations onto video frames...")
        output_video_frames = []

        if court_keypoints is not None:
            self.mini_court = MiniCourt(video_frames[0])
            # Process ALL frames mathematically before the drawing loop starts
            tracks = self.mini_court.convert_bounding_boxes_to_mini_court_coordinates(tracks, court_keypoints)
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            if court_keypoints is not None:
                frame = self.entity_annotator.draw_court_keypoints(frame, court_keypoints)
            
            # Use .get() safely in case a frame is missing tracking data
            player_dict = tracks.get("players", [])[frame_num]
            ball_dict = tracks.get("ball", [])[frame_num]
            
            # 1. Draw Players (Ellipse)
            for track_id, player in player_dict.items():
                frame = self.entity_annotator.draw_ellipse(frame, player["bbox"], PLAYER_COLOR, track_id)
            
            # 2. Draw Ball (Triangle)
            for track_id, ball in ball_dict.items():
                frame = self.entity_annotator.draw_triangle(frame, ball["bbox"], BALL_COLOR)

            # --- DRAW THE MINI COURT ---
            if self.mini_court is not None:
                frame = self.mini_court.draw_background_rectangle(frame)
                frame = self.mini_court.draw_court(frame)
                
                # Draw Players on radar
                for track_id, player in player_dict.items():
                    if "mini_court_position" in player:
                        pos = player["mini_court_position"]
                        cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, PLAYER_COLOR, -1)
                        
                # Draw Ball on radar
                for track_id, ball in ball_dict.items():
                    if "mini_court_position" in ball:
                        pos = ball["mini_court_position"]
                        cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, BALL_COLOR, -1)

            output_video_frames.append(frame)
            
            # TODO: Phase 5 - Draw Player Speeds Overlay

            output_video_frames.append(frame)
            
        logger.info("Annotation processing complete.")
        return output_video_frames