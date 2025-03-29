import numpy as np

def name_matches(short_name, full_name):
    return f'({short_name})' in full_name

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
            sorted_hv_imp[curr_idx] = [arr[2], arr[3]]
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

            elif math_type == "extend_backward":
                # Use the next valid value.
                if next_i < m:
                    filled[i, 1] = filled[next_i, 1]
                # If no next value exists, use the previous valid one.
                elif prev_i >= 0:
                    filled[i, 1] = filled[prev_i, 1]

            else:
                if math_type != "connect":
                    print("Invalid math_type. Please input 'connect', 'extend_forward', or 'extend_backward'. Default to 'connect'")
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

def sample_missing_values(empty_tsp, match_np, math_type="connect"):
    length = len(match_np)

    prev_tsp = match_np[0][0]
    prev_val = match_np[0][1]
    prev_hv = match_np[0][2]
    prev_imp = match_np[0][3]

    next_tsp = match_np[1][0]
    next_val = match_np[1][1]
    next_hv = match_np[1][2]
    next_imp = match_np[1][3]

    curr_tsp_index = 1

    sample_data = []

    for tsp in empty_tsp:
        this_val = None
        this_hv = None
        this_imp = None
        while (next_tsp < tsp):
            prev_tsp = next_tsp
            prev_val = next_val
            prev_hv = next_hv
            prev_imp = next_imp
            curr_tsp_index += 1
            
            if (curr_tsp_index == length):
                raise IndexError("Index Out Of Bounds")

            next_tsp = match_np[curr_tsp_index][0]
            next_val = match_np[curr_tsp_index][1]
            next_hv = match_np[curr_tsp_index][2]
            next_imp = match_np[curr_tsp_index][3]
        
        if math_type == "extend_forward":
            this_val = next_val
            this_hv = next_hv
            this_imp = next_imp

        elif math_type == "extend_backward":
            this_val = prev_val
            this_hv = prev_hv
            this_imp = prev_imp

        else:
            if math_type != "connect":
                print("Invalid math_type. Please input 'connect', 'extend_forward', or 'extend_backward'. Default to 'connect'")
            # Both neighbors found: interpolate.

            this_hv = prev_hv
            this_imp = prev_imp

            if prev_tsp != next_tsp:
                this_val = prev_val + (next_val - prev_val) * (tsp - prev_tsp) / (next_tsp - prev_tsp)
            else:
                this_val = prev_val
        sample_data.append([tsp, this_val, this_hv, this_imp])

    return np.array(sample_data)

def sample_data(nparr, fs, start_time = -1, end_time = -1, time_unit = "s", match_type = "connect"):
    data_start_time = nparr[0][0]
    data_end_time = nparr[-1][0]
    if time_unit == "s":
        if start_time != -1:
            start_time *= 1e3
        else:
            start_time = data_start_time
        if end_time != -1:
            end_time *= 1e3
        else:
            end_time = data_end_time
    if start_time > data_end_time or end_time < data_start_time:
        raise AttributeError("Time Range Incorrect")
    
    interval = 1000 / fs  # 1000 ms per second divided by samples per second

    # Generate the array of timestamps
    timestamps = np.arange(start_time, end_time, interval)
    sampled_data = sample_missing_values(timestamps, nparr, match_type)

    return sampled_data

def compute_fft(x, fs: int, center_frequencies=True):
    """
    Compute the DFT of x.

    Args:
        x: Signal of length N.
        fs: Sampling frequency.
        center_frequencies: If true then returns frequencies on [-f/2,f/2]. If false then returns frequencies between [0,f].
    Returns:
        X: DFT of x.
        f: Frequencies
    """
    N = len(x)
    X = np.fft.fft(x, norm="ortho")
    f = np.fft.fftfreq(N, 1/fs)
    if center_frequencies:
        X = np.fft.fftshift(X)
        f = np.fft.fftshift(f)
    return X, f

def compute_ifft(X, fs: int, center_frequencies=True):
    """
    Compute the iDFT of X.

    Args:
        X: Spectrum  of length N.
        fs: Sampling frequency.
        center_frequencies: If true then X is defined over the frequency range [-f/2,f/2]. If false then [0,f].
    Returns:
        x: iDFT of X and it should be real.
        t: real time instants in the range [0, T]
    """
    N = len(X)
    if center_frequencies:
      X = np.fft.ifftshift(X)
    x = np.fft.ifft(X, norm="ortho")
    t = np.arange(N)/fs
    return x.real, t

def bandlimitter(X, f, bl, bu):
    """
    Generate a band-limited signal by nulling the frequency components within the range [bl, bu].
    If bu is set to -1, the function nulls all frequency components above bl.

    Args:
        X: Spectrum (1D numpy array)
        f: Frequency range corresponding to the spectrum X (1D numpy array)
        bl: Lower bound of the frequency range to eliminate
        bu: Upper bound of the frequency range to eliminate; if bu = -1, eliminates all frequencies above bl

    Returns:
        X_BL: Spectrum of the band-limited signal
    """
    # Create a mask to null frequencies within the range [bl, bu]
    if bu == -1:
        mask = np.abs(f) <= bl
    else:
        mask = (np.abs(f) < bl) | (np.abs(f) > bu)

    # Apply the mask to the spectrum
    X_BL = X * mask

    return X_BL