import cv2
import numpy as np
from constants import (
    TEAM_ID_1,
    TEAM_ID_2,
    OVERLAY_RECT_START,
    OVERLAY_RECT_END,
    OVERLAY_COLOR,
    OVERLAY_ALPHA,
    OVERLAY_TEXT_POS_1,
    OVERLAY_TEXT_POS_2,
    OVERLAY_TEXT_COLOR,
    OVERLAY_FONT_SCALE,
    OVERLAY_FONT_THICKNESS
)

class StatsAnnotator:
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectangle 
        overlay = frame.copy()
        cv2.rectangle(overlay, OVERLAY_RECT_START, OVERLAY_RECT_END, OVERLAY_COLOR, cv2.FILLED)
        alpha = OVERLAY_ALPHA
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        
        # Get the number of times each team had control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==TEAM_ID_1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==TEAM_ID_2].shape[0]
        
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Control: {team_1*100:.2f}%", OVERLAY_TEXT_POS_1, cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, OVERLAY_TEXT_COLOR, OVERLAY_FONT_THICKNESS)
        cv2.putText(frame, f"Team 2 Control: {team_2*100:.2f}%", OVERLAY_TEXT_POS_2, cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE, OVERLAY_TEXT_COLOR, OVERLAY_FONT_THICKNESS)

        return frame
