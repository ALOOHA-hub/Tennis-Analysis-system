def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2]-bbox[0]

def measure_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1, p2):
    return p1[0]-p2[0], p1[1]-p2[1]

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int(y2)

def get_height_of_bbox(bbox):
    return bbox[3]-bbox[1]

def get_closest_keypoint_index(point, keypoints, valid_indices):
    closest_distance = float('inf')
    key_point_ind = valid_indices[0]
    for keypoint_indix in valid_indices:
        keypoint = keypoints[keypoint_indix*2], keypoints[keypoint_indix*2+1]
        distance = abs(point[1]-keypoint[1])
        if distance < closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_indix
    return key_point_ind

def convert_pixel_distance_to_meters(pixel_distance, reference_height_in_meters, reference_height_in_pixels):
    return (pixel_distance * reference_height_in_meters) / reference_height_in_pixels

def convert_meters_to_pixel_distance(meters, reference_width_in_meters, reference_width_in_pixels):
    return (meters * reference_width_in_pixels) / reference_width_in_meters