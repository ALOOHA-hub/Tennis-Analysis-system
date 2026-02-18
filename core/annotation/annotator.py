import cv2
from .entity_annotator import EntityAnnotator
from utils.logger import logger
from constants.visual_consts import PLAYER_COLOR, BALL_COLOR

class Annotator:
    def __init__(self):
        logger.info("Initializing Video Annotator...")
        self.entity_annotator = EntityAnnotator()
        
    def draw_annotations(self, video_frames, tracks, court_keypoints=None, mini_court=None):
        logger.info("Drawing visual annotations onto video frames...")
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks.get("players", [])[frame_num]
            ball_dict = tracks.get("ball", [])[frame_num]
            
            # 1. Draw Court Keypoints
            if court_keypoints is not None:
                frame = self.entity_annotator.draw_court_keypoints(frame, court_keypoints)

            # 2. Draw Players and Physics Stats (Speed & Distance)
            for track_id, player in player_dict.items():
                frame = self.entity_annotator.draw_ellipse(frame, player["bbox"], PLAYER_COLOR, track_id)
                
                speed = player.get('speed')
                distance = player.get('distance')
                if speed is not None or distance is not None:
                    frame = self.entity_annotator.draw_player_speed_and_distance(frame, player["bbox"], speed, distance)
            
            # 3. Draw Ball
            for track_id, ball in ball_dict.items():
                frame = self.entity_annotator.draw_triangle(frame, ball["bbox"], BALL_COLOR)

            # 4. Draw the Mini Court Radar
            if mini_court is not None:
                frame = mini_court.draw_background_rectangle(frame)
                frame = mini_court.draw_court(frame)
                
                for track_id, player in player_dict.items():
                    if "mini_court_position" in player:
                        pos = player["mini_court_position"]
                        cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, PLAYER_COLOR, -1)
                        
                for track_id, ball in ball_dict.items():
                    if "mini_court_position" in ball:
                        pos = ball["mini_court_position"]
                        cv2.circle(frame, (int(pos[0]), int(pos[1])), 5, BALL_COLOR, -1)

            output_video_frames.append(frame)
            
        logger.info("Annotation processing complete.")
        return output_video_frames