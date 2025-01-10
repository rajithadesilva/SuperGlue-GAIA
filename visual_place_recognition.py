import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define variables for reference and months
REF = "march"
MONTHS = ["march", "april", "may", "june", "september"]

# Set the reference point as specified
x_ref, y_ref, z_ref = -6.67979613163444, 0.3024452780728452, 0.0

# Define the starting index for analysis
START_INDEX = 500  # Set your desired starting index here

# Define paths for ground truth poses
tum_file_path = f'assets/long/{REF}/gt_poses_{REF}.tum'  # Adjust as needed

# Step 1: Load Ground Truth Poses
def load_ground_truth_poses(tum_file_path):
    gt_poses = {}
    with open(tum_file_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            data = line.strip().split()
            timestamp = int(float(data[0]))  # Convert timestamp to integer index
            x = float(data[1])
            y = float(data[2])
            z = float(data[3])
            gt_poses[timestamp] = {'x': x, 'y': y, 'z': z}
    return gt_poses

# Step 2: Process Evaluation Files
def process_evaluation_files(eval_folder, gt_poses):
    pattern = re.compile(r'(\d+)_(\d+)_evaluation\.npz')
    data = []

    if not os.path.exists(eval_folder):
        print(f"Warning: Evaluation folder '{eval_folder}' does not exist.")
        return pd.DataFrame(data)  # Return empty DataFrame

    for filename in sorted(os.listdir(eval_folder)):
        match = pattern.match(filename)
        if match:
            idx0 = int(match.group(1))
            idx1 = int(match.group(2))
            if idx1 < START_INDEX:
                continue  # Skip indices before START_INDEX

            eval_file_path = os.path.join(eval_folder, filename)

            # Load evaluation data
            eval_data = np.load(eval_file_path)
            confidence = eval_data['confidence']
            num_keypoints = len(confidence)

            # Compute average matching score
            matching_score = np.mean(confidence) if num_keypoints > 0 else 0.0

            # Compute keypoints_score_product
            keypoints_score_product = num_keypoints * matching_score

            # Get ground truth pose for idx1
            if idx1 in gt_poses:
                pose1 = gt_poses[idx1]

                # Compute Euclidean distance from reference point to pose1
                distance = np.sqrt(
                    (pose1['x'] - x_ref)**2 +
                    (pose1['y'] - y_ref)**2 +
                    (pose1['z'] - z_ref)**2
                )

                # Append to data list
                data.append({
                    'image_idx': idx1,
                    'distance': distance,
                    'matching_score': matching_score,
                    'num_keypoints': num_keypoints,
                    'keypoints_score_product': keypoints_score_product
                })
            else:
                print(f"Warning: Missing ground truth pose for index {idx1}")
    return pd.DataFrame(data)

# Step 3: Analyze and Save Results
def analyze_and_save_results(df, output_metrics_file, output_plot_file, month):
    if df.empty:
        print(f"No data to process for {month}. Skipping.")
        return

    # Filter data from START_INDEX onwards
    df = df[df['image_idx'] >= START_INDEX]

    if df.empty:
        print(f"No data after index {START_INDEX} for {month}. Skipping.")
        return

    # Sort data by image index
    df = df.sort_values(by='image_idx')

    # Normalize distance and keypoints_score_product to [0,1]
    df['distance_norm'] = (df['distance'] - df['distance'].min()) / (df['distance'].max() - df['distance'].min())
    df['keypoints_score_product_norm'] = (df['keypoints_score_product'] - df['keypoints_score_product'].min()) / (df['keypoints_score_product'].max() - df['keypoints_score_product'].min())

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_metrics_file), exist_ok=True)

    # Save metrics to CSV
    df.to_csv(output_metrics_file, index=False)

    # Find the index where keypoints_score_product is maximized
    max_idx = df.loc[df['keypoints_score_product'].idxmax(), 'image_idx']
    max_product = df['keypoints_score_product'].max()

    # Plot distance and keypoints_score_product
    plt.figure(figsize=(12, 6))
    #plt.plot(df['image_idx'], df['distance_norm'], label='Normalized Distance from Reference Point')
    plt.plot(df['image_idx'], df['keypoints_score_product_norm'], label='Normalized Keypoints Score Product')

    # Draw vertical line at max keypoints_score_product index
    plt.axvline(x=max_idx, color='red', linestyle='--', label=f'Max Product at Index {int(max_idx)}')

    # Add text label near the line
    plt.text(max_idx, 0.05, f'Index {int(max_idx)}', rotation=90, verticalalignment='bottom', color='red')

    plt.title(f'Distance from Reference Point and Keypoints Score Product over Images ({month.capitalize()} vs {REF.capitalize()})\nStarting from Index {START_INDEX}')
    plt.xlabel('Image Index')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_plot_file)
    plt.close()  # Close the plot to free memory
    print(f"Results saved to {output_metrics_file} and {output_plot_file}.")

# Main Execution
if __name__ == '__main__':
    # Load ground truth poses
    gt_poses = load_ground_truth_poses(tum_file_path)
    print(f"Loaded {len(gt_poses)} ground truth poses from '{tum_file_path}'.")

    # Reference point is already set explicitly
    print(f"Reference point set to: ({x_ref}, {y_ref}, {z_ref})")
    print(f"Starting analysis from index {START_INDEX}.")

    # Iterate over the months
    for MONTH in MONTHS:
        if MONTH == REF:
            print(f"Skipping reference month '{REF}'.")
            continue

        print(f"\nProcessing month: {MONTH}")

        # Define paths using REF and MONTH
        eval_folder = f'dump_demo_sequence/{REF}_ref/{MONTH}/'
        output_metrics_file = f'vps/{MONTH}_distance_matching_score_ref_{REF}_from_{START_INDEX}.csv'
        output_plot_file = f'vps/{MONTH}_distance_matching_score_plot_ref_{REF}_from_{START_INDEX}.png'

        # Process evaluation files
        df = process_evaluation_files(eval_folder, gt_poses)
        print(f"Processed {len(df)} evaluation files for month '{MONTH}' after index {START_INDEX}.")

        # Analyze results and save outputs
        analyze_and_save_results(df, output_metrics_file, output_plot_file, MONTH)
