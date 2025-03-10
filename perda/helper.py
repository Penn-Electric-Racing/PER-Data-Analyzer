import numpy as np

def align_nparr(np_list: list[np.ndarray]):
    num_arr = len(np_list)
    sorted_arr = merge_and_sort_with_source(np_list)
    final_sorted_arr = []
    sorted_hv_imp = []
    curr_idx = -1
    this_tsp = 0
    last_tsp = -1
    for arr in sorted_arr:
        this_tsp = arr[0]
        arr_input_num = int(arr[-1])
        if this_tsp != last_tsp:
            final_sorted_arr.append([None] * (num_arr+1))
            sorted_hv_imp.append([])
            curr_idx += 1
            final_sorted_arr[curr_idx][0] = this_tsp
            sorted_hv_imp[curr_idx] = [this_tsp, arr[2], arr[3]]
        final_sorted_arr[curr_idx][arr_input_num+1] = arr[1]
        last_tsp = this_tsp
    return np.array(final_sorted_arr), np.array(sorted_hv_imp)

def merge_and_sort_with_source(arrays: list[np.ndarray]):
    """
    Merge a list of NumPy arrays (each with a timestamp in the first column) into one array.
    An extra column is added to mark the source (input order) of each row. The merged array is
    then sorted first by the timestamp and, for rows with identical timestamps, by the input sequence.
    
    :param arrays: List of NumPy arrays, each assumed to have a timestamp in column 0.
    :return: A single merged and sorted NumPy array with an extra column indicating the source.
    """
    new_arrays = []
    
    # Add an extra column to each array that indicates the input order (source index)
    for source_index, arr in enumerate(arrays):
        # Create a column with the source index repeated for each row in the array
        source_col = np.full((arr.shape[0], 1), source_index)
        # Append the source column to the right of the original array
        new_arr = np.hstack((arr, source_col))
        new_arrays.append(new_arr)
    
    # Merge all arrays vertically
    merged_array = np.vstack(new_arrays)
    
    # Sort the merged array using lexsort.
    # np.lexsort takes a tuple of keys, where the last key is the primary key.
    # Here, we want to sort primarily by the timestamp (column 0) and then by source index (last column).
    sorted_indices = np.lexsort((merged_array[:, -1], merged_array[:, 0]))
    sorted_array = merged_array[sorted_indices]
    
    return sorted_array

def fill_missing_values(arr, math_type="connect"):
    """
    Fills missing values in a 2D array (each row is [timestamp, val]) where some val entries may be None.
    
    Parameters:
    arr (np.ndarray): 2D array with shape (m, 2), where arr[i,0] is the timestamp and arr[i,1] is the value.
                        The array's dtype should be object if it contains None.
    math_type (str): The method used for filling missing values.
                    "connect" - linear interpolation between nearest valid neighbors.
                    "extend_forward" - use the last valid value.
                    "extend_back" - use the next valid value.
                    
    Returns:
    np.ndarray: A new array with missing values filled.
    """
    # Make a copy so we don't modify the original array.
    filled = arr.copy()
    m = len(filled)
    
    # Loop over each row in the array.
    for i in range(m):
        if filled[i, 1] is None:
            # Find the previous valid index.
            prev_i = i - 1
            while prev_i >= 0 and filled[prev_i, 1] is None:
                prev_i -= 1
            # Find the next valid index.
            next_i = i + 1
            while next_i < m and filled[next_i, 1] is None:
                next_i += 1

            if math_type == "extend_forward":
                # Use the previous valid value.
                if prev_i >= 0:
                    filled[i, 1] = filled[prev_i, 1]
                # If no previous value exists, use the next valid one.
                elif next_i < m:
                    filled[i, 1] = filled[next_i, 1]

            elif math_type == "extend_back":
                # Use the next valid value.
                if next_i < m:
                    filled[i, 1] = filled[next_i, 1]
                # If no next value exists, use the previous valid one.
                elif prev_i >= 0:
                    filled[i, 1] = filled[prev_i, 1]

            else:
                if math_type != "connect":
                    print("Invalid math_type. Please input 'connect', 'extend_forward', or 'extend_back'. Default to 'connect'")
                # Both neighbors found: interpolate.
                if prev_i >= 0 and next_i < m:
                    t_prev, v_prev = filled[prev_i, 0], filled[prev_i, 1]
                    t_next, v_next = filled[next_i, 0], filled[next_i, 1]
                    t_current = filled[i, 0]
                    # Avoid division by zero.
                    if t_next != t_prev:
                        interpolated_val = v_prev + (v_next - v_prev) * (t_current - t_prev) / (t_next - t_prev)
                        filled[i, 1] = interpolated_val
                    else:
                        filled[i, 1] = v_prev
                # If only one neighbor is available, fallback to that value.
                elif prev_i >= 0:
                    filled[i, 1] = filled[prev_i, 1]
                elif next_i < m:
                    filled[i, 1] = filled[next_i, 1]
                
    return filled

def name_matches(short_name, full_name):
    return f'({short_name})' in full_name