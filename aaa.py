import cv2 
import line_utils
import infer

from skimage import io, img_as_bool, morphology
import matplotlib.pyplot as plt
import numpy as np
from post_process_prediction._post_process_binary_mask_old import load_binary_mask
from collections import Counter

from post_process_prediction._post_process_binary_mask_old import get_kaplan_meier_data_from_events


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
                


def process_groups(my_list, max_consecutive_increase_count=4, axis='y'):
    """
    Process groups of entries in `my_list` based on consecutive increases in the x or y values.

    Parameters:
        my_list (list of dicts): A list of dictionaries containing 'x' and 'y' values.
        max_consecutive_increase_count (int): The number of consecutive increases that trigger a new group.
        axis (str): Either 'x' or 'y' to specify which axis to use for processing ('x' or 'y').
        
    Returns:
        list: A list of processed groups, where values in the axis are adjusted based on most common value.
    """
    groups = []
    groups_info = []
    current_group = []
    consecutive_increase_count = 0
    prev_value = None  # This will hold the previous value of the axis (either x or y)

    # Choose the key based on the axis ('x' or 'y')
    axis_key = axis

    for i, entry in enumerate(my_list):
        if not current_group:
            # Start a new group
            current_group.append(entry)
            prev_value = entry[axis_key]  # Initialize the value as the first one
            continue

        # Check if the current value on the chosen axis is greater than the previous value
        if entry[axis_key] > prev_value:
            consecutive_increase_count += 1
        else:
            consecutive_increase_count = 0

        # Add the current entry to the current group
        current_group.append(entry)

        # If there are 'max_consecutive_increase_count' consecutive increases, close the current group
        if consecutive_increase_count >= max_consecutive_increase_count:
            # Get the most common value of the axis in the current group
            axis_values = [item[axis_key] for item in current_group]
            most_common_value = Counter(axis_values).most_common(1)[0][0]

            # Set all axis values in the current group to the most common value
            for entry in current_group:
                entry[axis_key] = most_common_value

            # Add the group to the list of groups
            groups.append(current_group)


            start_pos = (current_group[0]['x'], current_group[0]['y'])

            # Start a new group with the current item
            current_group = [entry]
            prev_value = entry[axis_key]  # Update the previous value for the new group
            consecutive_increase_count = 0

            groups_info.append({
                'group_id': len(groups),
                'start_pos': start_pos,
                'end_pos': (entry['x'], entry['y']),
                'most_common_value': prev_value
            })

        # Update the previous value to the most common value in the current group
        prev_value = Counter([item[axis_key] for item in current_group]).most_common(1)[0][0]

    # Process the last group
    if current_group:
        axis_values = [item[axis_key] for item in current_group]
        most_common_value = Counter(axis_values).most_common(1)[0][0]
        for entry in current_group:
            entry[axis_key] = most_common_value

        groups.append(current_group)

        start_pos = (current_group[0]['x'], current_group[0]['y'])
        groups_info.append({
            'group_id': len(groups),
            'start_pos': start_pos,
            'end_pos': (entry['x'], entry['y']),
            'most_common_value': prev_value
        })

    return my_list, groups_info


img_path = "sample_result_mask_3.png"
# img_path = "inst_mask_3.png"
binary_mask = load_binary_mask(img_path)

new_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
binary_mask = get_skeleton(binary_mask)

cv2.imwrite("skeleton.png", binary_mask)

x_range = line_utils.get_xrange(binary_mask)



# key_points, num_lines = line_utils.get_kp(binary_mask, interval=2, x_range=x_range, get_num_lines=True, get_center=True)
# new_image = line_utils.draw_kps(new_image, key_points, color=(0,127,255))



# key_points_x = line_utils.get_kp(binary_mask, interval=1, x_range=None, get_num_lines=False, get_center=True)

key_points_x = get_kp_first_hit_x(binary_mask, steps=1)
# new_image = line_utils.draw_kps(new_image, key_points_x, color=(200,127,127))

key_points_y = get_kp_first_hit_y(binary_mask, steps=1)
# new_image = line_utils.draw_kps(new_image, key_points_y, color=(127,127,200))

# key_points_y = line_utils.get_kp_y(binary_mask, interval=1, y_range=None, get_num_lines=False, get_center=True)
# new_image = line_utils.draw_kps(new_image, key_points_y, color=(0,127,255))


def extend_lines_x(binary_mask, group_dict):
    new_list = []
    height, width = binary_mask.shape
    for item in group_dict:
        _item = item.copy()
        x, y = item['start_pos']
        x = int(x)
        y = int(y)

        _item['start_pos'] = (x, y)
        _item['end_pos'] = (width-1, y)

        new_list.append(_item)

    return new_list


def extend_lines_y(binary_mask, group_dict):
    new_list = []
    height, width = binary_mask.shape
    for item in group_dict:
        _item = item.copy()
        x, y = item['end_pos']
        x = int(x)
        y = int(y)

        _item['start_pos'] = (x, y)
        _item['end_pos'] = (x, 0)

        new_list.append(_item)

    return new_list


def draw_lines(img, group_dict, color=(0,255,0)):
    for item in group_dict:
        x, y = item['start_pos']
        x = int(x)
        y = int(y)

        x_end, y_end = item['end_pos']
        x_end = int(x_end)
        y_end = int(y_end)

        img = cv2.line(img, (x, y), (x_end, y_end), color, 1)
    return img


def draw_ponints(img, pnt_list, radius=2, color=(0, 255, 255)):
    for item in pnt_list:
        x, y = item
        img = cv2.circle(img, (x, y), radius=radius, color=color, thickness=-1)
    return img


def keep_first_point_found_on_x(points, group_x_dict):
    points_to_keep = []

    for item in group_x_dict:
        line_x, line_y = item['start_pos']
        for pnt in points: 
            pnt_x, pnt_y = pnt
            if pnt_y == line_y:
                points_to_keep.append(pnt)
                break
    
    return points_to_keep



key_points_x, group_x_dict = process_groups(key_points_x, axis='y')
group_x_dict = extend_lines_x(binary_mask, group_x_dict)
# new_image = draw_lines(new_image, group_x_dict, color=(0, 255, 0))
# new_image = line_utils.draw_kps(new_image, key_points_x, color=(0,255,0))


key_points_y, group_y_dict = process_groups(key_points_y, axis='x')
group_y_dict = extend_lines_y(binary_mask, group_y_dict)
# new_image = draw_lines(new_image, group_y_dict, color=(0, 0, 255))
# new_image = line_utils.draw_kps(new_image, key_points_y, color=(0,0,255))


def find_intersections(horizontal_lines, vertical_lines):
    intersections = []
    
    # Loop through each horizontal line
    for h_line in horizontal_lines:
        h_y = h_line['start_pos'][1]  # y-coordinate of the horizontal line
        h_x_start = min(h_line['start_pos'][0], h_line['end_pos'][0])
        h_x_end = max(h_line['start_pos'][0], h_line['end_pos'][0])
        
        # Loop through each vertical line
        for v_line in vertical_lines:
            v_x = v_line['start_pos'][0]  # x-coordinate of the vertical line
            v_y_start = min(v_line['start_pos'][1], v_line['end_pos'][1])
            v_y_end = max(v_line['start_pos'][1], v_line['end_pos'][1])
            
            # Check if the lines intersect
            if h_x_start <= v_x <= h_x_end and v_y_start <= h_y <= v_y_end:
                # Intersection point is (v_x, h_y)
                intersections.append((v_x, h_y))
    
    return intersections



intersection_points = find_intersections(group_x_dict, group_y_dict)
intersection_points = keep_first_point_found_on_x(intersection_points, group_x_dict)


new_image = cv2.imread('plt_0.png')
new_image = draw_ponints(new_image, intersection_points, color=(127, 0, 127), radius=5)

df = get_kaplan_meier_data_from_events(intersection_points, 1)
print(df)



cv2.imwrite("sample_result_mask.png", new_image)
