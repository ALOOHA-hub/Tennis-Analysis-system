from utils.bbox_utils import measure_distance
from constants.visual_consts import DOUBLE_LINE_WIDTH

class PhysicsEngine:
    def __init__(self, fps, mini_court_width):
        self.fps = fps
        # 10.97 meters / radar_width_pixels = a constant uniform conversion ratio
        self.meters_per_pixel = DOUBLE_LINE_WIDTH / mini_court_width

    def add_speed_and_distance_to_tracks(self, tracks, frame_window=12):
        """Calculates distance and speed. Default window is 12 frames (0.5 seconds at 24fps) to smooth jitter."""
        total_distance = {}
        last_speed = {}

        for frame_num in range(len(tracks["players"])):
            for track_id, player in tracks["players"][frame_num].items():
                if track_id not in total_distance:
                    total_distance[track_id] = 0.0
                    last_speed[track_id] = 0.0

                # 1. Calculate Frame-by-Frame Distance (Total Distance)
                if frame_num > 0 and track_id in tracks["players"][frame_num - 1]:
                    prev_pos = tracks["players"][frame_num - 1][track_id].get("mini_court_position")
                    curr_pos = player.get("mini_court_position")
                    
                    if prev_pos and curr_pos:
                        dist_pixels = measure_distance(prev_pos, curr_pos)
                        dist_meters = dist_pixels * self.meters_per_pixel
                        total_distance[track_id] += dist_meters

                # 2. Calculate Windowed Speed (km/h)
                if frame_num >= frame_window:
                    prev_window_player = tracks["players"][frame_num - frame_window].get(track_id)
                    if prev_window_player:
                        prev_window_pos = prev_window_player.get("mini_court_position")
                        curr_pos = player.get("mini_court_position")
                        
                        if prev_window_pos and curr_pos:
                            dist_pixels = measure_distance(prev_window_pos, curr_pos)
                            dist_meters = dist_pixels * self.meters_per_pixel
                            
                            time_elapsed = frame_window / self.fps
                            speed_mps = dist_meters / time_elapsed
                            speed_kmh = speed_mps * 3.6  # Convert m/s to km/h
                            last_speed[track_id] = speed_kmh

                # 3. Store the physics data natively inside the tracks dictionary
                player['distance'] = total_distance[track_id]
                player['speed'] = last_speed[track_id]

        return tracks