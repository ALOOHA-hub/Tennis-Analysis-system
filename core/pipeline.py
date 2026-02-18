import os
import pickle
from utils.video_utils import read_video, save_video
from utils.logger import logger
from utils.config_loader import cfg
from core.trackers import Tracker
from core.annotation import Annotator
from core.detection import CourtDetector
from core.analysis import PhysicsEngine
from core.annotation import MiniCourt

class Pipeline:
    def __init__(self, input_video_path: str, output_video_path: str):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path        
        self.tracker = Tracker()
        self.annotator = Annotator()
        self.court_detector = CourtDetector(cfg['models']['court_detector']['model_path'])
        # We will load our YOLO trackers and ResNet court detectors here in Phase 2
        logger.info("Tennis Analysis Pipeline initialized.")

    def run(self):
        logger.info("--- Starting Tennis Analysis Pipeline ---")
        
        video_frames = read_video(self.input_video_path)
        if not video_frames: return

        # 1. Base Tracking
        tracks = self._get_tracks(video_frames)
        
        # 2. Court Detection & Filtering
        logger.info("Detecting court lines...")
        court_keypoints = self.court_detector.predict(video_frames[0])
        tracks = self.tracker.choose_and_filter_players(court_keypoints, tracks)
        tracks = self.tracker.add_position_to_tracks(tracks)
        
        # 3. Phase 4: Mini-Court Projection
        logger.info("Projecting tracking coordinates to 2D Mini-Court...")
        mini_court = MiniCourt(video_frames[0])
        tracks = mini_court.convert_bounding_boxes_to_mini_court_coordinates(tracks, court_keypoints)

       # 4. Phase 5: Physics & Analytics
        logger.info("Calculating player real-world speeds and distances...")
        
        # FIX: Use the global 'cfg' we imported at the top of the file
        fps = cfg.get('video', {}).get('fps', 24.0) 
        
        physics = PhysicsEngine(fps, mini_court.court_drawing_width)
        tracks = physics.add_speed_and_distance_to_tracks(tracks)
        # 5. Draw Everything
        annotated_frames = self.annotator.draw_annotations(
            video_frames, 
            tracks, 
            court_keypoints=court_keypoints,
            mini_court=mini_court
        )
        
        logger.info("Pipeline processing complete. Moving to save step.")
        save_video(annotated_frames, self.output_video_path, fps=fps)
        logger.info("---Pipeline Completed Successfully---")

    def _get_tracks(self, video_frames):
        """Runs tracking, loads unified stub, or migrates old legacy stubs."""
        
        # Load exactly what is in the config, no magic strings
        tracks_stub_file = cfg['paths'].get('unified_stub')
        legacy_player_stub = cfg['paths'].get('legacy_player_stub')
        legacy_ball_stub = cfg['paths'].get('legacy_ball_stub')
        
        # Ensure the directory for the unified stub exists
        if tracks_stub_file:
            os.makedirs(os.path.dirname(tracks_stub_file), exist_ok=True)
        else:
            logger.error("unified_stub path is missing from config.yaml!")
            return None

        # 1. Primary Check: Load unified stub if it exists
        if os.path.exists(tracks_stub_file):
            logger.info(f"Loading unified tracking data from: {tracks_stub_file}")
            with open(tracks_stub_file, 'rb') as f:
                return pickle.load(f)

        # 2. Fallback Check: Migrate legacy stubs if they exist
        if legacy_player_stub and legacy_ball_stub and os.path.exists(legacy_player_stub) and os.path.exists(legacy_ball_stub):
            logger.info("Legacy stubs detected. Triggering data migration...")
            
            with open(legacy_player_stub, 'rb') as f:
                old_players = pickle.load(f)
            with open(legacy_ball_stub, 'rb') as f:
                old_ball = pickle.load(f)
                
            tracks = {"players": [], "ball": []}
            
            # Transform Player Data
            for frame_dict in old_players:
                frame_players = {track_id: {"bbox": bbox} for track_id, bbox in frame_dict.items()}
                tracks["players"].append(frame_players)
                
            # Transform Ball Data
            for frame_dict in old_ball:
                frame_ball = {ball_id: {"bbox": bbox} for ball_id, bbox in frame_dict.items()}
                tracks["ball"].append(frame_ball)
                
            # Apply new interpolation logic
            logger.info("Applying updated interpolation logic to migrated ball data...")
            tracks["ball"] = self.tracker.interpolate_ball_positions(tracks["ball"])
            
            # Save the new unified stub
            logger.info(f"Migration successful. Saving unified stub to: {tracks_stub_file}")
            with open(tracks_stub_file, 'wb') as f:
                pickle.dump(tracks, f)
                
            return tracks

        # 3. Execution: Run Models if NO stubs exist
        logger.info("No stubs found. Running AI inference (this may take a few minutes)...")
        
        logger.info("[1/2] Detecting Players...")
        player_detections = self.player_detector.detect_frames(
            video_frames, 
            conf=cfg['models']['player_tracker']['confidence_threshold']
        )
        
        logger.info("[2/2] Detecting Ball...")
        ball_detections = self.ball_detector.detect_frames(
            video_frames, 
            conf=cfg['models']['ball_tracker']['confidence_threshold']
        )
        
        tracks = self.tracker.get_object_tracks(player_detections, ball_detections)
        
        logger.info(f"Saving new tracking data to stub: {tracks_stub_file}")
        with open(tracks_stub_file, 'wb') as f:
            pickle.dump(tracks, f)
            
        return tracks