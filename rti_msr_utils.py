import os
import numpy as np
import math
import matplotlib.pylab as plt
import pandas as pd
from typing import List
from collections import defaultdict
from bisect import bisect_left

maximum_distance = 10

def cal_distance(left: List[float], right: List[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Both points must have the same number of dimensions")
    if len(left) == 0 or len(right) == 0:
        raise ValueError("Points must not be empty")
        
    return np.linalg.norm(np.array(left) - np.array(right))

def generate_mirror_pos(pos, room_length, room_width):
    """
    Generates a list of mirrored positions based on the original positions.
    
    Parameters:
    - pos: List of tuples containing original positions as (x, y).
    - room_length: The length of the room.
    - room_width: The width of the room.
    
    Returns:
    - mirror_pos: List of mirrored positions.
    """
    mirror_pos = []
    for point in pos:
        x, y = point
        mirrors = [
            (x, y),
            (x, 2 * room_width - y),
            (x, -y),
            (-x, y),
            (2 * room_length - x, y)
        ]
        mirror_pos.append(mirrors)
    return mirror_pos

def generate_mirror_pos_rel(mirror_pos, rel_x, rel_y):
    """
    Shifts each mirrored position by the relative x and y coordinates.
    
    Parameters:
    - mirror_pos: List of lists of tuples containing mirrored points.
    - rel_x: The relative x-coordinate.
    - rel_y: The relative y-coordinate.
    
    Returns:
    - shifted_pos: List of shifted mirrored positions.
    """
    shifted_pos = [[(point[0] - rel_x, point[1] - rel_y) for point in points] for points in mirror_pos]
    return shifted_pos

def visualize_room(room_length, room_width, original_points, mirrored_points, rel_x, rel_y):
    """
    Visualize the room, original points, mirrored points, and relative point.
    
    Parameters:
    - room_length: The length of the room.
    - room_width: The width of the room.
    - original_points: List of tuples containing original points.
    - mirrored_points: List of lists of tuples containing mirrored points.
    - rel_x: The x-coordinate of the relative point.
    - rel_y: The y-coordinate of the relative point.
    """
    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    # Draw room boundaries
    ax.set_xlim([-room_length, 2 * room_length])
    ax.set_ylim([-room_width, 2 * room_width])
    
    # Draw the room
    ax.add_patch(plt.Rectangle((0, 0), room_length, room_width, fill=False, edgecolor='black', linewidth=2))

    # Plot original points
    for point in original_points:
        plt.scatter(*point, c='blue', label='Original Point')

    # Plot mirrored points
    for points in mirrored_points:
        for point in points:
            plt.scatter(*point, c='red', marker='x', label='Mirrored Point')

    # Plot the relative point
    plt.scatter(rel_x, rel_y, c='green', marker='s', label='Relative Point')
    
    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Room and Points Visualization')
    
    # Show legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # Show the plot
    plt.show()

def generate_mpc_position(anchor_pair_list, mirror_pos):
    """
    Generates multi-path component (MPC) positions based on a list of anchor pairs and mirror positions.
    
    Parameters:
    - anchor_pair_list: List of tuples each containing two indices representing the anchor pairs.
    - mirror_pos: List of lists of tuples containing mirrored positions for each anchor.
    
    Returns:
    - mpc_position: Dictionary mapping each anchor pair to a NumPy array of their MPC positions.
    """
    mpc_position = {}
    
    for anchor_pair in anchor_pair_list:
        i, j = anchor_pair
        distances = np.zeros(5)

        # Calculate distances
        for k in range(5):
            distances[k] = cal_distance(mirror_pos[i][0], mirror_pos[j][k])

        # Remove duplicates and adjust distances to be relative to the base distance
        base_distance = distances[0]
        unique_distances = np.unique(distances)
        adjusted_distances = unique_distances - base_distance
        
        mpc_position[anchor_pair] = adjusted_distances

    return mpc_position

def update_weights(weight_matrix, small_position, large_position, base_distance, romm_x_len, romm_y_len, pixel_width, prm_lambda):
    anchor_distance = cal_distance(small_position, large_position)
    anchor_distance_base = round(anchor_distance - base_distance, 2)
    if (anchor_distance_base > maximum_distance):
        return
    for j in range(romm_x_len):
        for k in range(romm_y_len):
            x_position = j * pixel_width + pixel_width / 2
            y_position = k * pixel_width + pixel_width / 2
            pixel_position = (x_position, y_position)
            
            pixel_2_anchor_distance = cal_distance(pixel_position, small_position) + cal_distance(pixel_position, large_position)
            
            if pixel_2_anchor_distance < anchor_distance + prm_lambda:
                weight_matrix[anchor_distance_base][k * romm_x_len + j] = 1 / np.sqrt(anchor_distance)

def generate_weight_matrix(anchor_pair_list, mpc_position, mirror_pos, romm_x_len, romm_y_len, pixel_width, num_of_mirror, prm_lambda):
    """
    Generates a weight matrix based on various parameters.
    """
    pixel_num_j = romm_x_len * romm_y_len
    weight_matrix = {}
    
    for item in anchor_pair_list:
        small, large = item
        weight_matrix[item] = {}
        
        # Initialize with zeros
        for distance in mpc_position[item]:
            distance = round(distance, 2)
            weight_matrix[item][distance] = np.zeros(pixel_num_j)
        
        base_distance = cal_distance(mirror_pos[small][0], mirror_pos[large][0])

        small_position = mirror_pos[small][0]
        for i in range(num_of_mirror):
            large_position = mirror_pos[large][i]
            update_weights(weight_matrix[item], small_position, large_position, base_distance, romm_x_len, romm_y_len, pixel_width, prm_lambda)
        
        large_position = mirror_pos[large][0]
        for i in range(1, num_of_mirror):
            small_position = mirror_pos[small][i]
            update_weights(weight_matrix[item], small_position, large_position, base_distance, romm_x_len, romm_y_len, pixel_width, prm_lambda)
    
    return weight_matrix

def plot_weight_matrix(weight_matrix, anchor_pair, distance, romm_x_len, romm_y_len):
    # Get the specific weight matrix slice for the given anchor_pair and distance
    weight_slice = weight_matrix[anchor_pair][distance]
    
    # Reshape the weight slice into a 2D array if needed (assuming romm_x_len and romm_y_len are known)
    weight_slice_2d = weight_slice.reshape((romm_y_len, romm_x_len))
    
    # Create the heatmap
    plt.imshow(weight_slice_2d, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar()
    
    plt.title(f'Weight Matrix for Anchor Pair {anchor_pair} and Distance {distance}')
    plt.xlabel('X-coordinate (in pixels)')
    plt.ylabel('Y-coordinate (in pixels)')
    
    plt.show()

def index_to_xy(index, room_x_len):
    """Convert a 1D index to a 2D (x, y) coordinate."""
    y = index // room_x_len
    x = index % room_x_len
    return x, y

def calculate_position(x, y, pixel_width):
    """Calculate the actual position based on x, y, and pixel_width."""
    return (x * pixel_width + pixel_width / 2, y * pixel_width + pixel_width / 2)

def generate_regularization_matrix(room_x_len, room_y_len, pixel_width):
    """
    Generate a regularization matrix C based on room dimensions and pixel width.
    """
    num_elements = room_x_len * room_y_len
    regularization_matrix_c = np.zeros((num_elements, num_elements))
    
    # Cache for distance calculations
    position_cache = {}
    
    for k in range(num_elements):
        x_k, y_k = index_to_xy(k, room_x_len)
        k_position = calculate_position(x_k, y_k, pixel_width)
        
        for l in range(num_elements):
            x_l, y_l = index_to_xy(l, room_x_len)
            
            # Use cached position if available, otherwise calculate and cache
            if l in position_cache:
                l_position = position_cache[l]
            else:
                l_position = calculate_position(x_l, y_l, pixel_width)
                position_cache[l] = l_position
            
            distance = cal_distance(k_position, l_position)
            regularization_matrix_c[k, l] = 0.5 * math.exp(-distance / 0.5)
    
    return regularization_matrix_c

def plot_regularization_matrix(regularization_matrix):
    plt.imshow(regularization_matrix, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.title('Regularization Matrix C')
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.show()

def convert_weight_matrix_to_numpy(weight_matrix, anchor_pair_list, mpc_position, pixel_num_j, missing_mpc):
    """Convert weight_matrix dictionary to NumPy array"""
    # Filter anchor pairs
    filtered_anchor_pair_list = [anchor_pair for anchor_pair in anchor_pair_list if anchor_pair not in missing_mpc]

    # Calculate the total number of rows in the resulting NumPy array
    num_rows = sum(len(mpc_position[anchor_pair]) for anchor_pair in filtered_anchor_pair_list)
    
    # Initialize the NumPy array
    weight_matrix_np = np.zeros((num_rows, pixel_num_j))
    
    index = 0
    for anchor_pair in filtered_anchor_pair_list:
        for distance in mpc_position[anchor_pair]:
            rounded_distance = round(distance, 2)
            weight_matrix_np[index] = weight_matrix[anchor_pair][rounded_distance]
            index += 1
            
    return weight_matrix_np

def calculate_rti_matrix(weight_matrix_np, regularization_matrix_c):
    """Calculate the RTI matrix based on the weight_matrix_np and regularization_matrix_c"""
    # Precompute reusable terms
    weight_transpose_mult_weight = np.matmul(weight_matrix_np.T, weight_matrix_np)
    inv_regularization = np.linalg.inv(regularization_matrix_c)
    
    # Compute the RTI matrix
    term1 = np.linalg.inv(weight_transpose_mult_weight + inv_regularization * 0.5)
    term2 = weight_matrix_np.T
    rti_matrix = np.matmul(term1, term2)
    
    return rti_matrix

def convert_to_seconds(hmsms_string):
    """Convert a timestamp string to seconds."""
    h, m, s, ms = map(int, hmsms_string.replace(':', '.').split('.'))
    return h * 3600 + m * 60 + s + ms / 1e6

def to_complex(x):
    try:
        return complex(x.replace("i", "j"))
    except ValueError:
        print(f"Warning: Malformed string {x}. Returning 0.")
        return 0

def load_csv_data(file_path, num_cir_samples, tag_name_left=None, tag_name_right=None):
    """Load CSV data and return relevant data arrays."""
    uwb_data = pd.read_csv(file_path)
    print(os.path.splitext(os.path.basename(file_path))[0])

    to_numpy_fields = [
        "tx_id", "rx_id", "rx_pream_count", "timestamp", "fp_index", "rx_pow_dbm"
    ]
    tx_id, rx_id, rx_pream_count, timestamp, fp_index, rx_pow_dbm = [
        uwb_data[field].to_numpy() for field in to_numpy_fields
    ]

    # Time-stamp conversion
    time_stamp_sec = np.array([convert_to_seconds(ts) for ts in timestamp])

    # CIR magnitude
    cir = uwb_data.iloc[:, -num_cir_samples:].applymap(to_complex).to_numpy()

    # Calculations related to FP index and time offset
    fractional_fp = np.abs(fp_index) % 1
    cir_sample_period = 0  # Initialize or fetch this value
    time_offset_ns = fractional_fp * cir_sample_period * 1e9

    # Ground Position
    ground_x, ground_y = None, None
    if tag_name_left and tag_name_right:
        left_arm_x, left_arm_y = uwb_data[f"{tag_name_left}x"].to_numpy(), uwb_data[f"{tag_name_left}y"].to_numpy()
        right_arm_x, right_arm_y = uwb_data[f"{tag_name_right}x"].to_numpy(), uwb_data[f"{tag_name_right}y"].to_numpy()
        ground_x, ground_y = (left_arm_x + right_arm_x) / 2, (left_arm_y + right_arm_y) / 2

    return uwb_data, cir, tx_id, rx_id, rx_pream_count, fractional_fp, rx_pow_dbm, time_stamp_sec, ground_x, ground_y

def group_data_by_tx_rx(tx_id_uwb, rx_id_uwb, cir, cir_up, rx_pream_count, fp, rx_lvl, time_stamp, ground_x=None, ground_y=None):
    grouped_data = defaultdict(lambda: defaultdict(list))
    data_length = len(tx_id_uwb)

    for i in range(data_length):
        # Check if the value is infinite or negative infinite
        if np.isinf(rx_lvl[i]) or np.isinf(cir[i]).any() or np.isinf(cir_up[i]).any() or np.isinf(rx_pream_count[i]) or np.isinf(fp[i]) or np.isinf(time_stamp[i]):
            continue
        
        tx = tx_id_uwb[i]
        rx = rx_id_uwb[i]
        key = tuple(sorted((tx, rx)))

        grouped_data[key]['cir'].append(cir[i])
        grouped_data[key]['cir_up'].append(cir_up[i])
        grouped_data[key]['rx_pream_count'].append(rx_pream_count[i])
        grouped_data[key]['fp'].append(fp[i])
        grouped_data[key]['rx_lvl'].append(rx_lvl[i])
        grouped_data[key]['time_stamp'].append(time_stamp[i])
        grouped_data[key]['ground_x'].append(ground_x[i] if ground_x is not None else None)
        grouped_data[key]['ground_y'].append(ground_y[i] if ground_y is not None else None)

    for key, metrics in grouped_data.items():
        for metric, value_list in metrics.items():
            np_array = np.array(value_list)
            grouped_data[key][metric] = np_array
    
    return grouped_data

def upsample_and_align_cir(cir, fp, freq_s_ratio=64):
    N = len(cir[0])  # Assuming `cir` is a 2D array, get the length of the first row

    # Initialize as a complex-valued array
    cir_up = np.zeros((len(cir), N * freq_s_ratio), dtype=complex)

    # Upsampling
    for i in range(len(cir)):
        y = cir[i]
        Y = np.fft.fft(y)
        
        # Pad the frequency-domain signal with zeros
        Y_pad = np.concatenate((Y[:N // 2], np.zeros(N * (freq_s_ratio - 1)), Y[N // 2:]))
        
        # Transform back to time domain with higher sample rate
        y_pad = np.fft.ifft(Y_pad)
        
        # Assign the upsampled array to cir_up
        cir_up[i] = y_pad[:N * freq_s_ratio]

    offset = np.round(fp*64).astype(int)

    aligned_cirs = []
    # index alignment using fp
    for i in range(len(cir_up)):
        aligned_cirs.append(cir_up[i][offset[i]:]) 

    # Find the shortest length among the aligned CIRs
    min_length = min([len(cir) for cir in aligned_cirs])
    
    # Trim all aligned CIRs to this minimum length
    aligned_cirs = [cir[:min_length] for cir in aligned_cirs]
    
    # Convert to numpy array for easier manipulation
    aligned_cirs = np.array(aligned_cirs)
    
    # Phase alignment
    index = 4*freq_s_ratio
    for i in range(len(aligned_cirs)):
        angle = np.angle(aligned_cirs[i][index])
        aligned_cirs[i] =  aligned_cirs[i] * np.exp(-1j * angle)
    
    return aligned_cirs

def plot_cir_lines_with_mpc(cir_up, anchor_pair, mpc_position, start_index, end_index):
    """
    Plots CIR lines with MPC positions as vertical lines.

    Parameters:
    - cir_up: The CIR data array
    - anchor_pair: The anchor pair for which MPC positions are available
    - start_index: The starting index for the range of CIR lines to plot
    - end_index: The ending index for the range of CIR lines to plot
    """
    
    # Create a distance array
    sampling_points = 64  # Number of sampling points for 0.15m
    distance_per_sample = 0.3 / sampling_points  # Distance per sample
    zero_point = 4 * 64  # Zero point position

    # Assuming the length of your cir_up[i] is 'n'
    n = len(cir_up[start_index])  # Replace this with the actual length

    # Create distance array
    distance_array = np.arange(-zero_point, n - zero_point) * distance_per_sample

    # Plotting
    for i in range(start_index, end_index):
        plt.plot(distance_array, abs(cir_up[i]))

    # Plot vertical lines for mpc_position[anchor_pair]
    for pos in mpc_position[anchor_pair]:
        plt.axvline(x=pos, color='r', linestyle='--')

    plt.xlabel('Distance (m)')
    plt.ylabel('Amplitude')
    plt.title('CIR Lines with MPC Positions')
    plt.show()

def plot_cir_lines_with_mpc_mean(cir_up, anchor_pair, mpc_position, start_index, end_index):
    """
    Plots CIR lines with MPC positions as vertical lines.

    Parameters:
    - cir_up: The CIR data array
    - anchor_pair: The anchor pair for which MPC positions are available
    - start_index: The starting index for the range of CIR lines to plot
    - end_index: The ending index for the range of CIR lines to plot
    """
    
    # Create a distance array
    sampling_points = 64  # Number of sampling points for 0.15m
    distance_per_sample = 0.3 / sampling_points  # Distance per sample
    zero_point = 4 * 64  # Zero point position

    # Assuming the length of your cir_up[i] is 'n'
    n = len(cir_up[start_index])  # Replace this with the actual length

    # Create distance array
    distance_array = np.arange(-zero_point, n - zero_point) * distance_per_sample

    # Plotting
    plt.plot(distance_array, np.mean(np.abs(cir_up[start_index:end_index]),axis=0))

    # Plot vertical lines for mpc_position[anchor_pair]
    for pos in mpc_position[anchor_pair]:
        plt.axvline(x=pos, color='r', linestyle='--')

    plt.xlabel('Distance (m)')
    plt.ylabel('Amplitude')
    plt.title('CIR Lines with MPC Positions')
    plt.show()

def plot_overlayed_weight_matrices(weight_matrix, anchor_pair, room_x_len, room_y_len, x_pos, y_pos):
    # Get the specific weight matrices for the given anchor_pair
    anchor_data = weight_matrix[anchor_pair]
    
    # Initialize the figure
    plt.figure(figsize=(10, 10))
    
    # Use different color maps to differentiate between the matrices
    color_maps = ['hot', 'cool', 'plasma', 'viridis', 'cividis']
    
    # Iterate over the distances to overlay the weight matrices
    for i, (distance, weight_slice) in enumerate(sorted(anchor_data.items())):
        # Get and reshape the weight slice
        weight_slice_2d = weight_slice.reshape((room_y_len, room_x_len))
        
        # Create the heatmap
        plt.imshow(weight_slice_2d, cmap=color_maps[i % len(color_maps)], alpha=0.5, origin='lower')
    
    # Plot the point in meters
    plt.scatter(x_pos[0] / 0.1, y_pos[0] / 0.1, c='red', marker='o', s=100, label=f'Point ({x_pos[0]}m, {y_pos[0]}m)')
    for i in range(1,len(x_pos)):
        plt.scatter(x_pos[i] / 0.1, y_pos[i] / 0.1, c='red', marker='o', s=100)
    
    # Add a color bar for demonstration, may not be strictly accurate due to overlaying
    plt.colorbar()
    
    # Add labels and title
    plt.title(f'Overlayed Weight Matrices for Anchor Pair {anchor_pair}')
    
    # Adjust tick labels to show meters
    x_ticks = np.arange(0, room_x_len, int(room_x_len / 10))
    y_ticks = np.arange(0, room_y_len, int(room_y_len / 10))
    plt.xticks(x_ticks, (x_ticks * 0.1))
    plt.yticks(y_ticks, (y_ticks * 0.1))
    
    plt.xlabel('X-coordinate (in meters)')
    plt.ylabel('Y-coordinate (in meters)')
    
    # Show legend
    plt.legend()
    
    plt.show()

def plot_2_cir_lines_with_mpc_mean(cir_up1, cir_up2, anchor_pair, mpc_position, start_index1, end_index1, start_index2, end_index2):
    """
    Plots two CIR lines with MPC positions as vertical lines.

    Parameters:
    - cir_up1, cir_up2: The CIR data arrays for the two datasets
    - anchor_pair: The anchor pair for which MPC positions are available
    - start_index1, end_index1: The starting and ending indices for the range of CIR lines to plot for the first dataset
    - start_index2, end_index2: The starting and ending indices for the range of CIR lines to plot for the second dataset
    """
    
    # Create a distance array
    sampling_points = 64  # Number of sampling points for 0.15m
    distance_per_sample = 0.3 / sampling_points  # Distance per sample
    zero_point = 4 * 64  # Zero point position

    # Assuming the length of your cir_up[i] is 'n'
    n = len(cir_up1[start_index1])  # Replace this with the actual length

    # Create distance array
    distance_array = np.arange(-zero_point, n - zero_point) * distance_per_sample

    # Plotting for the first dataset
    plt.plot(distance_array, np.mean(np.abs(cir_up1[start_index1:end_index1]), axis=0), color='b', label='Statistics')

    # Plotting for the second dataset
    plt.plot(distance_array, np.mean(np.abs(cir_up2[start_index2:end_index2]), axis=0), color='g', label='Real Time')

    # Plot vertical lines for mpc_position[anchor_pair]
    for pos in mpc_position[anchor_pair]:
        plt.axvline(x=pos, color='r', linestyle='--')

    plt.xlabel('Distance (m)')
    plt.ylabel('Amplitude')
    plt.title('CIR Lines with MPC Positions')
    plt.legend()
    plt.show()

def cal_mag_multipath_component(anchor_pair, mpc_position, grouped_data, start, end, ratio):
    positions = mpc_position[anchor_pair]

    cir_up = grouped_data[anchor_pair]["cir_up"]
    cir_up_mean = np.mean(np.abs(cir_up[start:end]),axis=0)
    pream_count = np.mean(grouped_data[anchor_pair]["rx_pream_count"][start:end])
    recei_power_level = np.mean(grouped_data[anchor_pair]["rx_lvl"][start:end])

    # if cir_up are empty, continue
    if len(cir_up) == 0:
        return []


    mpc_mag = []
    for pos in positions:
        pos = int(pos / 0.3 * ratio) + 4*ratio
        power_of_path = 10*math.log10((cir_up_mean[pos] ** 2 + 
                            cir_up_mean[pos - ratio] ** 2 + 
                            cir_up_mean[pos + ratio] ** 2)/(pream_count ** 2)) - 121.74
        power_of_path = power_of_path - recei_power_level
        mpc_mag.append(power_of_path)
    return mpc_mag

def find_timestamp_indices(ts, start_time, end_time):
    start_idx = bisect_left(ts, start_time)
    end_idx = bisect_left(ts, end_time, lo=start_idx)

    return start_idx, end_idx

def cal_z(anchor_pair_list, mpc_mag_idle, grouped_data, mpc_position, start_time, end_time):
    z = []
    missing_mpc = []  # List to store anchor_pair with missing mpc_mag_pair

    for anchor_pair in anchor_pair_list:
        mpc_mag_pair = mpc_mag_idle.get(anchor_pair, [])
        
        if not mpc_mag_pair:
            missing_mpc.append(anchor_pair)  # Save the anchor_pair for later processing
            continue

        data_pair = grouped_data.get(anchor_pair, {})
        ts = data_pair.get("time_stamp", [])

        start_idx, end_idx = find_timestamp_indices(ts, start_time, end_time)
        mpc_map = cal_mag_multipath_component(anchor_pair, mpc_position, grouped_data, start_idx, end_idx, 64)
        z.append(np.array(mpc_map) - np.array(mpc_mag_pair))

    return z, missing_mpc  # Return both lists
















