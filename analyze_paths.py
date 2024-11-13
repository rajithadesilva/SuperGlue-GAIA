import numpy as np
from pathlib import Path
import evo.core.trajectory as et
import evo.core.metrics as em
#'''
scale = {
    "march": [9.21786078, 817.54724485, 1.0],
    "april": [7.41406952, 4367.62265586, 1.0],
    "may": [8.40812939, -186.88633132, 1.0],
    "june": [7.92170162, 142.66150795, 1.0],
    "september": [39.41186153, -31.20859617, 1.0]
}
'''
scale = {
    "march": [9.21786078, 1.0, 1.0],
    "april": [7.41406952, 1.0, 1.0],
    "may": [8.40812939, 1.0, 1.0],
    "june": [7.92170162, 1.0, 1.0],
    "september": [39.41186153, 1.0, 1.0]
}
'''
# Declare descriptor and month variables
desc = 'U-64U-196U-FN-SPBG'
month = 'september'

def read_poses(npz_files):
    """
    Reads the ground truth and recovered poses from npz files.
    """
    gt_poses = []
    recovered_poses = []

    for data_path in npz_files:
        # Load the saved pose data
        data = np.load(data_path)
        gt_poses.append(data['ground_truth_pose'])
        if 'recovered_pose' in data:
            recovered_poses.append(data['recovered_pose'])

    return gt_poses, recovered_poses

def convert_to_absolute(poses, scale_factors=None):
    """
    Converts a list of relative poses to absolute poses.
    """
    cumulative_translation = np.zeros(3)
    cumulative_rotation = np.eye(3)
    absolute_poses = []

    for i, pose in enumerate(poses):
        R = pose[:3, :3]
        t = pose[:3, 3]

        # Apply scaling if provided
        if scale_factors is not None:
            t = np.divide(t, scale_factors)

        # Update cumulative rotation and translation
        cumulative_rotation = cumulative_rotation @ R
        cumulative_translation += t

        # Construct absolute pose
        absolute_pose = np.eye(4)
        absolute_pose[:3, :3] = cumulative_rotation
        absolute_pose[:3, 3] = cumulative_translation

        # Append to absolute poses list
        absolute_poses.append(absolute_pose)

    return absolute_poses

def analize_transformations():
    """
    Analyzes the ground truth and recovered transformations.
    Computes and prints basic pose analysis using evo.
    """
    # Construct the data path
    data_dir = Path(f'dump_match_pairs/{desc}/{month}/')
    npz_files = list(data_dir.glob('*_poses.npz'))

    if not npz_files:
        print('No pose files found in the specified directory.')
        return

    # Read poses from npz files
    gt_poses, recovered_poses = read_poses(npz_files)

    # Convert relative poses to absolute poses
    absolute_gt_poses = convert_to_absolute(gt_poses)
    absolute_recovered_poses = convert_to_absolute(recovered_poses, scale_factors=scale[month])

    # Create synthetic timestamps
    timestamps = np.arange(len(absolute_gt_poses))

    # Create PoseTrajectory3D objects with absolute poses
    gt_traj = et.PoseTrajectory3D(poses_se3=absolute_gt_poses, timestamps=timestamps)
    rec_traj = et.PoseTrajectory3D(poses_se3=absolute_recovered_poses, timestamps=timestamps)

    # Compute Absolute Trajectory Error (ATE)
    ape_metric = em.APE(em.PoseRelation.translation_part)
    ape_metric.process_data((gt_traj, rec_traj))
    ape_stats = ape_metric.get_all_statistics()

    # Print the ATE statistics
    print("Absolute Trajectory Error (ATE) statistics:")
    for k, v in ape_stats.items():
        print(f"{k}: {v}")

    # Compute Relative Pose Error (RPE)
    rpe_metric = em.RPE(em.PoseRelation.translation_part, delta=1, all_pairs=False)
    rpe_metric.process_data((gt_traj, rec_traj))
    rpe_stats = rpe_metric.get_all_statistics()

    # Print the RPE statistics
    print("\nRelative Pose Error (RPE) statistics:")
    for k, v in rpe_stats.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    analize_transformations()

