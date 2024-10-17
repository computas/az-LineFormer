from collections import Counter


def create_inline_groups(key_points, ):
    inline_groups = []
    inline_groups_info = []
    current_inline_group = []
    consecutive_increase_count = 0






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