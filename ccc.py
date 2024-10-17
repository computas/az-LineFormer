import cv2 
import line_utils

from post_process_binary_mask import load_binary_mask
import post_process_prediction.post_process_groups as post_process_groups
import post_process_prediction.post_process_utils as post_process_utils

from post_process_binary_mask import get_kaplan_meier_data_from_events

                



# img_path = "sample_result_mask_3.png"
img_path = "inst_mask_3.png"
img_path = "inst_mask_0.png"
binary_mask = load_binary_mask(img_path)

new_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
# new_image = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
binary_mask = post_process_utils.get_skeleton(binary_mask)


cv2.imwrite("skeleton.png", binary_mask)

x_range = line_utils.get_xrange(binary_mask)



# key_points, num_lines = line_utils.get_kp(binary_mask, interval=2, x_range=x_range, get_num_lines=True, get_center=True)
# new_image = line_utils.draw_kps(new_image, key_points, color=(0,127,255))



# key_points_x = line_utils.get_kp(binary_mask, interval=1, x_range=None, get_num_lines=False, get_center=True)

key_points_x = post_process_utils.get_kp_first_hit_x(binary_mask, steps=1)
# new_image = line_utils.draw_kps(new_image, key_points_x, color=(200,127,127))

key_points_y = post_process_utils.get_kp_first_hit_y(binary_mask, steps=1)
# new_image = line_utils.draw_kps(new_image, key_points_y, color=(255,127,0))

# key_points_y = line_utils.get_kp_y(binary_mask, interval=1, y_range=None, get_num_lines=False, get_center=True)
# new_image = line_utils.draw_kps(new_image, key_points_y, color=(0,127,255))


def extend_lines_x(binary_mask, group_dict):
    new_list = []
    height, width = binary_mask.shape
    for item in group_dict:
        _item = item.copy()
        x, y = (int(coord) for coord in item['start_pos'])
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
        x, y = (int(coord) for coord in item['end_pos'])

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



# key_points_x, group_x_dict = post_process_groups.process_groups(key_points_x, axis='y')

key_points_x = [(pnt['x'], pnt['y']) for pnt in key_points_x]
group_x_dict = post_process_groups.get_inline_groups(key_points_x, max_consecutive_increase_count=4, axis_to_group='x')
group_x_dict = extend_lines_x(binary_mask, group_x_dict)
new_image = draw_lines(new_image, group_x_dict, color=(0, 255, 0))
# new_image = line_utils.draw_kps(new_image, key_points_x, color=(0,255,0))


# key_points_y, group_y_dict = post_process_groups.process_groups(key_points_y, axis='x')
key_points_y = [(pnt['x'], pnt['y']) for pnt in key_points_y]
group_y_dict = post_process_groups.get_inline_groups(key_points_y, max_consecutive_increase_count=4, axis_to_group='y')
group_y_dict = extend_lines_y(binary_mask, group_y_dict)
new_image = draw_lines(new_image, group_y_dict, color=(255, 127, 0))
# new_image = line_utils.draw_kps(new_image, key_points_y, color=(255,127,))


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




def draw_km_lines(img, intersection_points, start_point, end_point, color=(0, 0, 255)):
    intersection_points = interpolate_points(intersection_points)

    intersection_points.insert(0, start_point)
    intersection_points.append(end_point)
    prev_point = None
    for point in intersection_points:
        if prev_point is None:
            prev_point = point
            continue
        
        
        img = cv2.line(img, prev_point, point, color, 1)
        prev_point = point

    return img


def interpolate_points(points):
    interpolated = []
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        interpolated.append((x1, y1))
        interpolated.append((x1, y2))
    interpolated.append(points[-1])  # Adding the last point (no interpolation needed)
    return interpolated



def rescale_coordinates(intersection_points, plot_start_coord, plot_end_coord):

    x_pixel_min, y_pixel_max = intersection_points[0]
    x_pixel_max, y_pixel_min = intersection_points[-1]

    x_coord_min, y_coord_max = plot_start_coord
    x_coord_max, y_coord_min = plot_end_coord

    # Rescaling function
    def rescale(value, old_min, old_max, new_min, new_max):
        return (value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    # Empty list to store the results
    mapped_coordinates = []

    # Loop over the pixel values
    for x, y in intersection_points:
        new_x = rescale(x, x_pixel_min, x_pixel_max, x_coord_min, x_coord_max)
        new_y = rescale(y, y_pixel_min, y_pixel_max, y_coord_min, y_coord_max)
        mapped_coordinates.append((new_x, new_y))

    return mapped_coordinates



intersection_points = find_intersections(group_x_dict, group_y_dict)
intersection_points = keep_first_point_found_on_x(intersection_points, group_x_dict)


# new_image = cv2.imread('plt_0.png')
new_image = draw_ponints(new_image, intersection_points, color=(0, 238, 220), radius=5)


start_point = group_x_dict[0]['start_pos']
end_point = group_y_dict[-1]['start_pos']

real_start_point = (0, 1)
real_end_point = (890, 0)
# new_image = draw_km_lines(new_image, intersection_points, start_point, end_point)
df = get_kaplan_meier_data_from_events(intersection_points, 1)
print(df)

intersection_points.insert(0, start_point)
intersection_points.append(end_point)

intersection_points = rescale_coordinates(intersection_points, real_start_point, real_end_point)
df = get_kaplan_meier_data_from_events(intersection_points, 1)



print(df)


print(group_x_dict)


cv2.imwrite("sample_result_mask.png", new_image)
