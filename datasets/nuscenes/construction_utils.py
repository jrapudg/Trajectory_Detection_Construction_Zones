import numpy as np
from nuscenes.prediction import convert_local_coords_to_global
from nuscenes.prediction.helper import convert_global_coords_to_local


def convert_coordinates_to_ego(points, ego_rotation, ego_translation):
    rotated_points = convert_global_coords_to_local(points, 
                                                    ego_translation, 
                                                    ego_rotation)
    return rotated_points

def get_box_points(width, length, instance_translation, instance_rotation):
    center = np.array([0, 0])
    upper_right_corner = np.array([width/2, length/2])
    upper_left_corner = np.array([-width/2, length/2])
    lower_right_corner = np.array([-width/2, -length/2])
    lower_left_corner = np.array([width/2, -length/2])
    points = np.vstack([center, upper_right_corner, upper_left_corner, 
                      lower_right_corner, lower_left_corner])
    return convert_local_coords_to_global(points, 
                                          instance_translation, 
                                          instance_rotation)