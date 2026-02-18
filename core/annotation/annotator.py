import cv2
from .entity_annotator import EntityAnnotator
from .stats_annotator import StatsAnnotator
from constants import (
    PLAYER_DEFAULT_COLOR,
    BALL_POSSESSION_COLOR,
    REFEREE_COLOR,
    BALL_COLOR
)

class Annotator:

    def __init__(self):
        self.entity_annotator = EntityAnnotator()
        self.stats_annotator = StatsAnnotator()
    
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            
            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", PLAYER_DEFAULT_COLOR)
                frame = self.entity_annotator.draw_ellipse(frame, player["bbox"], color, track_id)
                
                if player.get('has_ball', False):
                    frame = self.entity_annotator.draw_triangle(frame, player["bbox"], BALL_POSSESSION_COLOR)
                
                # Draw Speed and Distance
                speed = player.get('speed')
                distance = player.get('distance')
                if speed is not None or distance is not None:
                    frame = self.entity_annotator.draw_player_speed_and_distance(frame, player["bbox"], speed, distance)
            
            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.entity_annotator.draw_ellipse(frame, referee["bbox"], REFEREE_COLOR)
            
            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.entity_annotator.draw_triangle(frame, ball["bbox"], BALL_COLOR)

            # Draw Team Ball Control
            frame = self.stats_annotator.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)
        return output_video_frames