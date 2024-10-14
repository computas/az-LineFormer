import cv2 
import line_utils
import infer_playground

from post_process_binary_mask import load_binary_mask


img_path = "sample_result_mask_3.png"
binary_mask = load_binary_mask(img_path)


line_dataseries, inst_masks = infer_playground.get_dataseries(binary_mask, to_clean=False, return_masks=True)

new_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
# Visualize extracted line keypoints
prediction_image = line_utils.draw_lines(new_image, line_utils.points_to_array(line_dataseries))

cv2.imwrite("sample_result_mask.png", prediction_image)