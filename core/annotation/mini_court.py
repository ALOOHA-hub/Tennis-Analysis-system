import cv2
import numpy as np
from utils.bbox_utils import get_foot_position, get_center_of_bbox
from constants.visual_consts import (
    RADAR_WIDTH, RADAR_HEIGHT, RADAR_PADDING,
    RADAR_BG_COLOR, RADAR_LINE_COLOR,
    RADAR_PLAYER_COLOR, RADAR_BALL_COLOR
)

class MiniCourt:
    def __init__(self):
        self.width = RADAR_WIDTH
        self.height = RADAR_HEIGHT
        self.padding = RADAR_PADDING
        
        # Pre-build the background canvas to save CPU processing time
        self.background = self._create_court_background()
        
        # The exact 4 corners of our 2D radar map
        self.radar_corners = np.array([
            [self.padding, self.padding],                                   # Top-Left
            [self.padding + self.width, self.padding],                      # Top-Right
            [self.padding, self.padding + self.height],                     # Bottom-Left
            [self.padding + self.width, self.padding + self.height]         # Bottom-Right
        ], dtype=np.float32)

    def _create_court_background(self):
        """Draws the static 2D tennis court canvas."""
        canvas = np.zeros((self.height + 2 * self.padding, self.width + 2 * self.padding, 3), dtype=np.uint8)
        canvas[:] = RADAR_BG_COLOR
        
        # Outer boundary
        cv2.rectangle(canvas, (self.padding, self.padding), 
                      (self.padding + self.width, self.padding + self.height), RADAR_LINE_COLOR, 2)
        
        # Center line (Net)
        net_y = self.padding + self.height // 2
        cv2.line(canvas, (self.padding, net_y), (self.padding + self.width, net_y), RADAR_LINE_COLOR, 2)
        
        # Service lines & Singles margins (Mathematical approximations of a standard court)
        service_top = self.padding + int(self.height * 0.25)
        service_bottom = self.padding + int(self.height * 0.75)
        margin = int(self.width * 0.1)
        
        # Draw inner lines
        cv2.line(canvas, (self.padding + margin, service_top), (self.padding + self.width - margin, service_top), RADAR_LINE_COLOR, 2)
        cv2.line(canvas, (self.padding + margin, service_bottom), (self.padding + self.width - margin, service_bottom), RADAR_LINE_COLOR, 2)
        cv2.line(canvas, (self.padding + self.width // 2, service_top), (self.padding + self.width // 2, service_bottom), RADAR_LINE_COLOR, 2)
        cv2.line(canvas, (self.padding + margin, self.padding), (self.padding + margin, self.padding + self.height), RADAR_LINE_COLOR, 2)
        cv2.line(canvas, (self.padding + self.width - margin, self.padding), (self.padding + self.width - margin, self.padding + self.height), RADAR_LINE_COLOR, 2)

        return canvas

    def get_homography_matrix(self, video_keypoints):
        """Calculates the perspective transform from the camera angle to the 2D radar."""
        if video_keypoints is None: return None
            
        valid_points = []
        for i in range(0, len(video_keypoints), 2):
            x, y = video_keypoints[i], video_keypoints[i+1]
            if not np.isnan(x) and not np.isnan(y):
                valid_points.append([x, y])
                
        if len(valid_points) < 4: return None
            
        valid_pts = np.array(valid_points)
        
        # Dynamically extract the 4 extreme corners (TL, TR, BL, BR)
        sorted_y = valid_pts[valid_pts[:, 1].argsort()]
        top_pts = sorted_y[:len(valid_pts)//2]
        bot_pts = sorted_y[len(valid_pts)//2:]
        
        tl = top_pts[top_pts[:, 0].argsort()][0]
        tr = top_pts[top_pts[:, 0].argsort()][-1]
        bl = bot_pts[bot_pts[:, 0].argsort()][0]
        br = bot_pts[bot_pts[:, 0].argsort()][-1]
        
        video_corners = np.array([tl, tr, bl, br], dtype=np.float32)
        
        # Calculate the magical mapping matrix!
        return cv2.getPerspectiveTransform(video_corners, self.radar_corners)

    def project_point(self, point, matrix):
        """Applies homography matrix to project a single (x, y) video point to the 2D radar."""
        p = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(p, matrix)
        return int(transformed[0][0][0]), int(transformed[0][0][1])

    def draw_radar(self, frame, player_dict, ball_dict, court_keypoints):
        """Generates the radar map and alpha-blends it into the top right corner of the video."""
        matrix = self.get_homography_matrix(court_keypoints)
        if matrix is None: return frame
            
        radar = self.background.copy()
        
        # 1. Project & Draw Players (Using their FEET)
        for track_id, player in player_dict.items():
            foot_pos = get_foot_position(player["bbox"])
            proj_x, proj_y = self.project_point(foot_pos, matrix)
            cv2.circle(radar, (proj_x, proj_y), 15, RADAR_PLAYER_COLOR, -1)
            cv2.putText(radar, str(track_id), (proj_x - 5, proj_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
        # 2. Project & Draw Ball (Using its CENTER)
        for ball_id, ball in ball_dict.items():
            ball_center = get_center_of_bbox(ball["bbox"])
            proj_x, proj_y = self.project_point(ball_center, matrix)
            cv2.circle(radar, (proj_x, proj_y), 8, RADAR_BALL_COLOR, -1)

        # 3. Scale and overlay onto the main video frame
        scale = 0.25 # Make it 25% of its original size for the overlay
        small_radar = cv2.resize(radar, (0, 0), fx=scale, fy=scale)
        sr_h, sr_w = small_radar.shape[:2]
        
        margin = 30
        frame_h, frame_w = frame.shape[:2]
        
        # Positioning: Top Right Corner
        start_y, end_y = margin, margin + sr_h
        start_x, end_x = frame_w - sr_w - margin, frame_w - margin
        
        # Professional Alpha Blending Overlay
        alpha = 0.8
        roi = frame[start_y:end_y, start_x:end_x]
        blended = cv2.addWeighted(roi, 1 - alpha, small_radar, alpha, 0)
        frame[start_y:end_y, start_x:end_x] = blended
        
        return frame