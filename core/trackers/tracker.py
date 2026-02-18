import supervision as sv
import numpy as np
import pandas as pd
from core.utils.logger import logger
from core.utils.bbox_utils import get_center_of_bbox, get_foot_position
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

    def get_object_tracks(self, detections):
        logger.info("Assigning tracking IDs to tennis players and extracting ball positions...")
        
        tracks = {
            "players": [],
            "tennis ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Update ByteTrack (assigns persistent IDs to players)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["tennis ball"].append({})
            
            # Process tracked players
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv.get(CLASS_PLAYER):
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

            # Process the ball (No tracking ID needed, just grab the bbox for interpolation)
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv.get(CLASS_BALL):
                    tracks["tennis ball"][frame_num][1] = {"bbox": bbox}

        logger.info("Tracking complete. Interpolating ball positions...")
        tracks["tennis ball"] = self.interpolate_ball_positions(tracks["tennis ball"])

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