import supervision as sv
import numpy as np
import pandas as pd
from utils.logger import logger
from utils.bbox_utils import get_center_of_bbox, get_foot_position
from constants import (
    CLASS_PLAYER, 
    CLASS_BALL, 
    TRACKER_ACTIVATION_THRESHOLD,
    TRACKER_LOST_BUFFER,
    MAX_PIXEL_MOVE_PER_FRAME,
    INTERPOLATE_LIMIT,
    ROLLING_WINDOW,
    BFILL_LIMIT
)

class Tracker:
    def __init__(self):
        logger.info("Initializing ByteTrack Tracker for Tennis.")
        self.tracker = sv.ByteTrack(
            track_activation_threshold=TRACKER_ACTIVATION_THRESHOLD, 
            lost_track_buffer=TRACKER_LOST_BUFFER
        )

    def get_object_tracks(self, player_detections, ball_detections):
        logger.info("Assigning tracking IDs to tennis players and extracting ball positions...")
        
        tracks = {"players": [], "ball": []}

        # Loop through frames based on the length of our detections
        for frame_num in range(len(player_detections)):
            # 1. Handle Players (using player_detections)
            p_det = player_detections[frame_num]
            p_inv_names = {v: k for k, v in p_det.names.items()}
            p_supervision = sv.Detections.from_ultralytics(p_det)
            p_with_tracks = self.tracker.update_with_detections(p_supervision)
            
            tracks["players"].append({})
            for frame_detection in p_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                if cls_id == p_inv_names.get(CLASS_PLAYER):
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

            # 2. Handle Ball (using ball_detections)
            b_det = ball_detections[frame_num]
            b_inv_names = {v: k for k, v in b_det.names.items()}
            b_supervision = sv.Detections.from_ultralytics(b_det)
            
            tracks["ball"].append({})
            for frame_detection in b_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == b_inv_names.get(CLASS_BALL):
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        logger.info("Tracking complete. Interpolating ball positions...")
        tracks["ball"] = self.interpolate_ball_positions(tracks["ball"])
        return tracks

    def interpolate_ball_positions(self, ball_positions):
        # 1. Convert to DataFrame
        processed_positions = []
        for x in ball_positions:
            bbox = x.get(1, {}).get('bbox', [])
            if not bbox:
                processed_positions.append([np.nan, np.nan, np.nan, np.nan])
            else:
                processed_positions.append(bbox)
        
        df_ball_positions = pd.DataFrame(processed_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # 2. Filter False Positives (Sudden jumps)
        df_ball_positions['center_x'] = (df_ball_positions['x1'] + df_ball_positions['x2']) / 2
        df_ball_positions['center_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        
        df_ball_positions['dist'] = np.sqrt(
            df_ball_positions['center_x'].diff()**2 + df_ball_positions['center_y'].diff()**2
        )
        
        outliers = df_ball_positions['dist'] > MAX_PIXEL_MOVE_PER_FRAME
        df_ball_positions.loc[outliers, ['x1', 'y1', 'x2', 'y2']] = np.nan

        # 3. Interpolate
        df_ball_positions = df_ball_positions.interpolate(method='linear', limit=INTERPOLATE_LIMIT, limit_direction='both')

        # 4. Smooth the path
        df_ball_positions[['x1', 'y1', 'x2', 'y2']] = (
            df_ball_positions[['x1', 'y1', 'x2', 'y2']]
            .rolling(window=ROLLING_WINDOW, min_periods=1, center=True)
            .mean()
        )

        # 5. Fill remaining edges
        df_ball_positions = df_ball_positions.bfill(limit=BFILL_LIMIT)

        # Reconstruct format
        final_positions = []
        for row in df_ball_positions.to_numpy():
            if np.isnan(row[0]):
                final_positions.append({}) 
            else:
                final_positions.append({1: {"bbox": row[:4].tolist()}})

        return final_positions

    def choose_and_filter_players(self, court_keypoints, tracks):
        """Filters out the audience/umpires, keeping only the 2 actual players."""
        logger.info("Filtering audience/umpires based on spatial distance to court lines...")
        
        # Use the first frame to identify the two players
        player_detections_first_frame = tracks["players"][0]
        chosen_players = self._choose_players(court_keypoints, player_detections_first_frame)
        
        # Rebuild the tracks dictionary keeping only the chosen IDs
        filtered_player_tracks = []
        for player_dict in tracks["players"]:
            filtered_player_dict = {
                track_id: bbox_data 
                for track_id, bbox_data in player_dict.items() 
                if track_id in chosen_players
            }
            filtered_player_tracks.append(filtered_player_dict)
            
        tracks["players"] = filtered_player_tracks
        return tracks

    def _choose_players(self, court_keypoints, player_dict):
        """Calculates distance from people to court lines to find the 2 players."""
        import math
        from utils.bbox_utils import get_center_of_bbox
        
        distances = []
        for track_id, track_info in player_dict.items():
            player_center = get_center_of_bbox(track_info['bbox'])

            min_distance = float('inf')
            # Loop through all 14 court keypoints (stored as x,y,x,y...)
            for i in range(0, len(court_keypoints), 2):
                court_point = (court_keypoints[i], court_keypoints[i+1])
                distance = math.dist(player_center, court_point)
                if distance < min_distance:
                    min_distance = distance
                    
            distances.append((track_id, min_distance))
        
        # Sort by shortest distance to the court
        distances.sort(key=lambda x: x[1])
        
        # Safeguard: Ensure we have at least 2 detections before selecting
        if len(distances) >= 2:
            return [distances[0][0], distances[1][0]]
        else:
            return [d[0] for d in distances]

    @staticmethod
    def add_position_to_tracks(tracks):
        logger.info("Calculating real-world spatial positions (feet/center) for players and ball...")
        for obj, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if obj == "tennis ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[obj][frame_num][track_id]['position'] = position
        return tracks