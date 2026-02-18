import cv2
import numpy as np
from utils.bbox_utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance
)
from constants.visual_consts import (
    DOUBLE_LINE_WIDTH, HALF_COURT_LINE_HEIGHT, DOUBLE_ALLY_DIFFERENCE,
    NO_MANS_LAND_HEIGHT, SINGLE_LINE_WIDTH, 
    PLAYER_1_HEIGHT_METERS, PLAYER_2_HEIGHT_METERS
)

class MiniCourt:
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters, DOUBLE_LINE_WIDTH, self.court_drawing_width)

    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*28

        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(HALF_COURT_LINE_HEIGHT*2)
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 
        drawing_key_points[8] = drawing_key_points[0] + self.convert_meters_to_pixels(DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(NO_MANS_LAND_HEIGHT)
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(NO_MANS_LAND_HEIGHT)
        drawing_key_points[22] = drawing_key_points[20] + self.convert_meters_to_pixels(SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points = drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2), (4, 5), (6, 7), (1, 3),
            (0, 1), (8, 9), (10, 11), (2, 3)
        ]

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court(self, frame):
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x,y), 5, (0,0,255), -1)

        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out

    def get_mini_court_coordinates(self, object_position, closest_key_point, closest_key_point_index, player_height_in_pixels, player_height_in_meters):
        dist_x_pixels, dist_y_pixels = measure_xy_distance(object_position, closest_key_point)

        dist_x_meters = convert_pixel_distance_to_meters(dist_x_pixels, player_height_in_meters, player_height_in_pixels)
        dist_y_meters = convert_pixel_distance_to_meters(dist_y_pixels, player_height_in_meters, player_height_in_pixels)
        
        mini_court_x_dist = self.convert_meters_to_pixels(dist_x_meters)
        mini_court_y_dist = self.convert_meters_to_pixels(dist_y_meters)
        
        closest_mini_court_kp = (self.drawing_key_points[closest_key_point_index*2],
                                 self.drawing_key_points[closest_key_point_index*2+1])
        
        return (closest_mini_court_kp[0] + mini_court_x_dist,
                closest_mini_court_kp[1] + mini_court_y_dist)

    def convert_bounding_boxes_to_mini_court_coordinates(self, tracks, original_court_key_points):
        player_heights = {1: PLAYER_1_HEIGHT_METERS, 2: PLAYER_2_HEIGHT_METERS}

        # Dynamically map Tracker IDs to 1 and 2
        unique_ids = set()
        for frame_dict in tracks["players"]:
            unique_ids.update(frame_dict.keys())
        unique_ids = list(unique_ids)
        id_map = {tid: (1 if idx == 0 else 2) for idx, tid in enumerate(unique_ids)}

        for frame_num, player_dict in enumerate(tracks["players"]):
            ball_dict = tracks["ball"][frame_num]
            
            ball_position = None
            if 1 in ball_dict and 'bbox' in ball_dict[1]:
                ball_position = get_center_of_bbox(ball_dict[1]['bbox'])

            closest_player_id_to_ball = None
            if ball_position and player_dict:
                closest_player_id_to_ball = min(
                    player_dict.keys(), 
                    key=lambda x: measure_distance(ball_position, get_center_of_bbox(player_dict[x]['bbox']))
                )

            for player_id, player in player_dict.items():
                bbox = player['bbox']
                foot_position = get_foot_position(bbox)

                closest_key_point_index = get_closest_keypoint_index(foot_position, original_court_key_points, [0, 2, 12, 13])
                closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                     original_court_key_points[closest_key_point_index*2+1])

                frame_index_min = max(0, frame_num - 20)
                frame_index_max = min(len(tracks["players"]), frame_num + 50)
                bboxes_heights_in_pixels = [
                    get_height_of_bbox(tracks["players"][i][player_id]['bbox'])
                    for i in range(frame_index_min, frame_index_max)
                    if player_id in tracks["players"][i]
                ]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels) if bboxes_heights_in_pixels else get_height_of_bbox(bbox)

                mapped_id = id_map.get(player_id, 1)

                mini_court_player_position = self.get_mini_court_coordinates(
                    foot_position, closest_key_point, closest_key_point_index, 
                    max_player_height_in_pixels, player_heights.get(mapped_id, 1.88)
                )
                player["mini_court_position"] = mini_court_player_position

                if closest_player_id_to_ball == player_id and ball_position is not None:
                    closest_kp_idx_ball = get_closest_keypoint_index(ball_position, original_court_key_points, [0, 2, 12, 13])
                    closest_kp_ball = (original_court_key_points[closest_kp_idx_ball*2], 
                                       original_court_key_points[closest_kp_idx_ball*2+1])
                    
                    mini_court_ball_position = self.get_mini_court_coordinates(
                        ball_position, closest_kp_ball, closest_kp_idx_ball, 
                        max_player_height_in_pixels, player_heights.get(mapped_id, 1.88)
                    )
                    tracks["ball"][frame_num][1]["mini_court_position"] = mini_court_ball_position

        return tracks