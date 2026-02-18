import cv2
import numpy as np
from utils.bbox_utils import get_center_of_bbox, get_bbox_width
from constants.visual_consts import (
    TEXT_COLOR,
    TEXT_BG_COLOR,
    ELLIPSE_HEIGHT_RATIO,
    ELLIPSE_START_ANGLE,
    ELLIPSE_END_ANGLE,
    ELLIPSE_THICKNESS,
    TRIANGLE_SIZE,
    TRIANGLE_Y_OFFSET,
    FONT_SCALE,
    FONT_THICKNESS
)

class EntityAnnotator:
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        # Draw the half-ellipse at the feet
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(ELLIPSE_HEIGHT_RATIO * width)),
            angle=0.0,
            startAngle=ELLIPSE_START_ANGLE,
            endAngle=ELLIPSE_END_ANGLE,
            color=color,
            thickness=ELLIPSE_THICKNESS,
            lineType=cv2.LINE_4
        )
        
        # Optional: Draw player ID tag
        if track_id is not None:
            text = f"{track_id}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)[0]
            
            rect_x1 = x_center - text_size[0] // 2 - 5
            rect_y1 = y2 + 5
            rect_x2 = x_center + text_size[0] // 2 + 5
            rect_y2 = y2 + 15 + text_size[1]
            
            cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), color, cv2.FILLED)
            cv2.putText(frame, text, (x_center - text_size[0] // 2, y2 + 15 + text_size[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
            
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])  # Top of the ball bounding box
        x, _ = get_center_of_bbox(bbox)
        
        triangle_points = np.array([
            [x, y],
            [x - TRIANGLE_SIZE, y - TRIANGLE_Y_OFFSET],
            [x + TRIANGLE_SIZE, y - TRIANGLE_Y_OFFSET],
        ])
        
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, TEXT_COLOR, 1) # Black border for visibility
        return frame
    
    def draw_court_keypoints(self, frame, keypoints, color=(0, 0, 255)):
        """Draws the 14 structural keypoints of the tennis court."""
        for i in range(0, len(keypoints), 2):
            x = keypoints[i]
            y = keypoints[i+1]
            
            # SAFEGUARD: Skip drawing if the model failed to find this specific point
            if np.isnan(x) or np.isnan(y):
                continue
                
            x, y = int(x), int(y)
            # Draw a red dot for the keypoint
            cv2.circle(frame, (x, y), 5, color, -1)
            # Draw the index number above the dot
            cv2.putText(frame, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return frame


    def draw_player_speed_and_distance(self, frame, bbox, speed, distance):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        
        # Draw Speed (Adjusted Y offset slightly to hover cleanly above distance)
        if speed is not None:
            cv2.putText(
                frame,
                f"{speed:.1f} km/h",
                (int(x_center) - 40, int(y2) + 25), 
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                TEXT_COLOR,
                2
            )
        
        # Draw Distance
        if distance is not None:
            cv2.putText(
                frame,
                f"{distance:.1f} m",
                (int(x_center) - 30, int(y2) + 45), 
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                TEXT_COLOR,
                2
            )
        return frame