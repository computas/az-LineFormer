import post_process_prediction.post_process_utils as post_process_utils
from post_process_prediction.extract_events import extract_events

if __name__ == '__main__':
    # img_path = "sample_result_mask_3.png"
    # img_path = "inst_mask_3.png"
    img_path = "inst_mask_0.png"

    binary_mask = post_process_utils.load_binary_mask(img_path)
    df = extract_events(binary_mask, write_debug=True, map_to_plot_coordinates=True)
    print(df)