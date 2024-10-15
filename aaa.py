import cv2 
import line_utils
import infer

from skimage import io, img_as_bool, morphology
import matplotlib.pyplot as plt
import numpy as np
from post_process_binary_mask import load_binary_mask
from collections import Counter


def get_skeleton(mask):
    # Convert to a binary image
    binary_mask = img_as_bool(mask)
    skeleton = morphology.skeletonize(binary_mask)
    skeleton = (skeleton * 255).astype(np.uint8)

    return skeleton

def get_most_common_value(group, axis_key):
    """Get the most common value for the given axis in a group."""
    axis_values = [item[axis_key] for item in group]
    most_common_value = Counter(axis_values).most_common(1)[0][0]
    return most_common_value

def update_group_axis_values(group, axis_key, value):
    """Update the axis values of all entries in the group to the given value."""
    for entry in group:
        entry[axis_key] = value

def close_and_reset_group(groups, current_group, axis_key):
    """Close the current group, update axis values, and add it to groups."""
    most_common_value = get_most_common_value(current_group, axis_key)
    update_group_axis_values(current_group, axis_key, most_common_value)
    groups.append(current_group)
    return most_common_value  # Return the most common value for the next group's initialization

def process_groups(my_list, max_consecutive_increase_count=2, axis='y'):
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
            prev_value = close_and_reset_group(groups, current_group, axis_key)
            # Start a new group with the current item

            start_pos = (current_group[0]['x'], current_group[0]['y'])
            current_group = [entry]
            consecutive_increase_count = 0

            groups_info.append({
                'group_id': len(groups),
                'start_pos': start_pos,
                'end_pos': (entry['x'], entry['y']),
                'most_common_value': prev_value
            })

        # Update the previous value to the most common value in the current group
        prev_value = get_most_common_value(current_group, axis_key)

    # Process the last group
    if current_group:
        close_and_reset_group(groups, current_group, axis_key)

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




x_range = line_utils.get_xrange(binary_mask)
print(x_range)



# key_points, num_lines = line_utils.get_kp(binary_mask, interval=2, x_range=x_range, get_num_lines=True, get_center=True)
# new_image = line_utils.draw_kps(new_image, key_points, color=(0,127,255))



key_points_x = line_utils.get_kp(binary_mask, interval=1, x_range=None, get_num_lines=False, get_center=True)
new_image = line_utils.draw_kps(new_image, key_points_x, color=(200,127,127))

key_points_y = line_utils.get_kp_y(binary_mask, interval=1, y_range=None, get_num_lines=False, get_center=True)
new_image = line_utils.draw_kps(new_image, key_points_y, color=(0,127,255))



def draw_lines_x(img, group_dict):
    for item in group_dict:
        x, y = item['start_pos']
        x = int(x)
        y = int(y)

        img = cv2.line(img, (x, y), (512-1, y), (0,255,0), 1)
    return img

def draw_lines_y(img, group_dict):
    for item in group_dict:
        x, y = item['end_pos']
        x = int(x)
        y = int(y)

        img = cv2.line(img, (x, y), (x, 0), (0,0,255), 1)
    return img


key_points_x, group_x_dict = process_groups(key_points_x, axis='y')
# new_image = line_utils.draw_kps(new_image, key_points_x, color=(0,255,0))
new_image = draw_lines_x(new_image, group_x_dict)



key_points_y, group_y_dict = process_groups(key_points_y, axis='x')
new_image = draw_lines_y(new_image, group_y_dict)
# new_image = line_utils.draw_kps(new_image, key_points_y, color=(0,0,255))

print(group_x_dict)



cv2.imwrite("sample_result_mask.png", new_image)
assert False
all_key_points = key_points + key_points_y

# Sort by 'x' first, then by 'y'
sorted_kp = sorted(all_key_points, key=lambda d: (d['x'], d['y']))

# Remove duplicates by converting to a set of tuples, then back to a list of dictionaries
unique_kp = list({(d['x'], d['y']): d for d in sorted_kp}.values())


# kp = [{'x': int(p['x']), 'y': int(p['y'])} for p in unique_kp]


cv2.imwrite("sample_result_mask.png", new_image)

kp = infer.interpolate(unique_kp, inter_type='linear')
kp = [kp]


img = cv2.imread('plt_0.png')
prediction_image = line_utils.draw_lines(img, line_utils.points_to_array(kp))
cv2.imwrite("sample_result_mask_line.png", prediction_image)

