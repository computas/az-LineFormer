import cv2 
import line_utils
import infer

from post_process_binary_mask import load_binary_mask


from collections import Counter

def process_groups(my_list, max_consecutive_increase_count = 4):
    groups = []
    current_group = []
    consecutive_increase_count = 0
    prev_voted_y = None  # This will hold the previous voted 'y'

    for i, entry in enumerate(my_list):
        if not current_group:
            # Start a new group
            current_group.append(entry)
            prev_voted_y = entry['y']  # Initialize the voted y as the first 'y' value
            continue

        # Check if the current 'y' is lower than the previous voted y
        if entry['y'] > prev_voted_y:
            consecutive_increase_count += 1
        else:
            consecutive_increase_count = 0

        # Add the current entry to the current group
        current_group.append(entry)

        # If there are 4 consecutive decreases, close the current group and start a new one
        if consecutive_increase_count >= max_consecutive_increase_count:
            # Vote for the most common 'y' in the current group
            y_values = [item['y'] for item in current_group]
            most_common_y = Counter(y_values).most_common(1)[0][0]  # Get the most frequent 'y'
            
            # Set all 'y' values in the current group to the most common 'y'
            for entry in current_group:
                entry['y'] = most_common_y

            # Add the group to the list of groups
            groups.append(current_group)

            # Start a new group with the current item
            current_group = [entry]
            prev_voted_y = entry['y']  # Update previous voted y for the new group
            consecutive_increase_count = 0

        # Update the previous voted y to the most common y so far in the current group
        prev_voted_y = Counter([item['y'] for item in current_group]).most_common(1)[0][0]

    # Process the last group
    if current_group:
        y_values = [item['y'] for item in current_group]
        most_common_y = Counter(y_values).most_common(1)[0][0]
        for entry in current_group:
            entry['y'] = most_common_y

        groups.append(current_group)

    return my_list




# img_path = "sample_result_mask_3.png"
img_path = "inst_mask_3.png"
binary_mask = load_binary_mask(img_path)

new_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

x_range = line_utils.get_xrange(binary_mask)
print(x_range)



# key_points, num_lines = line_utils.get_kp(binary_mask, interval=2, x_range=x_range, get_num_lines=True, get_center=True)
# new_image = line_utils.draw_kps(new_image, key_points, color=(0,127,255))



# Track Kp and vote over them




def cnt_kps_in_line(kps, current_y):
    filter_kps = [kp for kp in kps if kp['y'] == current_y]
    return len(filter_kps)


def clean_kps(kps):
    new_kps = []

    for kp in kps:
        x, y = kp['x'], kp['y']

        inline_cnt = cnt_kps_in_line(kps, current_y=y)

        _dict = {
            'x': x,
            'y': y,
            'inline_cnt': inline_cnt
        }
        new_kps.append(_dict)

    return new_kps


        

    # pass

key_points = line_utils.get_kp(binary_mask, interval=1, x_range=None, get_num_lines=False, get_center=True)
new_image = line_utils.draw_kps(new_image, key_points, color=(200,127,127))

key_points_y = line_utils.get_kp_y(binary_mask, interval=1, y_range=None, get_num_lines=False, get_center=True)
new_image = line_utils.draw_kps(new_image, key_points_y, color=(0,127,255))


adjusted_key_points = clean_kps(key_points)
adjusted_key_points = process_groups(adjusted_key_points)
adjusted_key_points = [{'x': point['x'], 'y': point['y']} for point in adjusted_key_points]

new_image = line_utils.draw_kps(new_image, adjusted_key_points, color=(0,0,255))
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

