from skimage import img_as_bool, morphology
import matplotlib.pyplot as plt
import numpy as np


def get_skeleton(mask):
    # Convert to a binary image
    binary_mask = img_as_bool(mask)
    skeleton = morphology.skeletonize(binary_mask)
    skeleton = (skeleton * 255).astype(np.uint8)

    return skeleton


def find_starting_point(binary_mask):
    # Find most top left point
    y, x = np.where(binary_mask == 255)
    coordinates = sorted(zip(x, y), key=lambda coord: (coord[0], coord[1]))
    return coordinates[0]


def get_kp_first_hit_x(binary_mask, steps=10):
    height, width = binary_mask.shape
    starting_point = find_starting_point(binary_mask)

    kps = []

    for x in range(starting_point[0], width-1, steps):
        for y in range(0, height-1):
            if binary_mask[y, x] == 255:
                kps.append({
                    'x': x,
                    'y': y
                })
                break

    return kps


def get_kp_first_hit_y(binary_mask, steps=10):
    height, width = binary_mask.shape
    starting_point = find_starting_point(binary_mask)

    kps = []

    for y in range(starting_point[1], width, steps):
        for x in range(width-1, 0, -1):
            if binary_mask[y, x] == 255:
                kps.append({
                    'x': x,
                    'y': y
                })
                break

    return kps