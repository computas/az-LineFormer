import cv2 

from post_process_prediction._post_process_binary_mask_old import load_binary_mask
import post_process_prediction.post_process_groups as post_process_groups
import post_process_prediction.post_process_utils as post_process_utils
import post_process_prediction.post_process_debug_draw as post_process_debug_draw
from post_process_prediction.extract_data_from_coordinates import get_kaplan_meier_data_from_events, rescale_coordinates_to_plot_values

                
def extract_event_coordinates(binary_mask):
    debug_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Skeletonize
    binary_mask = post_process_utils.get_skeleton(binary_mask)

    # Find x-groups
    key_points_x = post_process_utils.get_kp_first_hit_x(binary_mask, steps=1)
    group_x_dict = post_process_groups.get_inline_groups(key_points_x, max_consecutive_increase_count=4, axis_to_group='x')
    group_x_dict = post_process_utils.extend_lines_x(binary_mask, group_x_dict)
    
    debug_image = post_process_debug_draw.draw_lines(debug_image, group_x_dict, color=(0, 255, 0))


    # Find y-groups
    key_points_y = post_process_utils.get_kp_first_hit_y(binary_mask, steps=1)
    group_y_dict = post_process_groups.get_inline_groups(key_points_y, max_consecutive_increase_count=4, axis_to_group='y')
    group_y_dict = post_process_utils.extend_lines_y(binary_mask, group_y_dict)

    debug_image = post_process_debug_draw.draw_lines(debug_image, group_y_dict, color=(255, 127, 0))


    # Find intersections
    intersection_points = post_process_utils.find_intersections(group_x_dict, group_y_dict)
    intersection_points = post_process_utils.keep_first_intersection_point_found_on_x_group(intersection_points, group_x_dict)

    debug_image = post_process_debug_draw.draw_ponints(debug_image, intersection_points, color=(0, 238, 220), radius=5)


    # Add start and end coordinates
    start_point = group_x_dict[0]['start_pos']
    end_point = group_y_dict[-1]['start_pos']
    intersection_points.insert(0, start_point)
    intersection_points.append(end_point)
    
    # debug_image = post_process_debug_draw.draw_ponints(debug_image, [start_point, end_point], color=(217, 156, 177), radius=5)
    # debug_image = post_process_debug_draw.draw_km_lines(debug_image, intersection_points)


    return intersection_points, debug_image




if __name__ == '__main__':
    # img_path = "sample_result_mask_3.png"
    # img_path = "inst_mask_3.png"
    img_path = "inst_mask_0.png"
    binary_mask = load_binary_mask(img_path)
    event_coordinates, debug_image = extract_event_coordinates(binary_mask)

    real_start_point = (0, 1)
    real_end_point = (890, 0)

    event_coordinates = rescale_coordinates_to_plot_values(event_coordinates, real_start_point, real_end_point)
    df = get_kaplan_meier_data_from_events(event_coordinates, group=1)

    print(df)

    cv2.imwrite("sample_result_mask.png", debug_image)

