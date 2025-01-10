import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

'''
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


rot = [[0,0,1],[-1,0,0],[0,-1,0]]
# Declare descriptor and month variables
desc = 'U-64U-196U-FN-SPBG'
month = 'march'

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

def plot_3d_frame(ax, R, t, frame_name, color):
    """
    Plots the coordinate frame in 3D using the rotation matrix R and translation vector t.
    """
    # Create origin
    origin = t

    # Define unit vectors along the coordinate axes
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]

    # Plot the coordinate frame
    ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color=color[0], label=f'{frame_name}_x')
    ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color=color[1], label=f'{frame_name}_y')
    ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color=color[2], label=f'{frame_name}_z')

def visualize_transformations():
    """
    Visualizes the ground truth and recovered transformations.
    """
    # Construct the data path
    data_dir = Path(f'dump_match_pairs/{desc}/{month}/')
    npz_files = list(data_dir.glob('*_poses.npz'))

    if not npz_files:
        print('No pose files found in the specified directory.')
        return

    # Read poses from npz files
    gt_poses, recovered_poses = read_poses(npz_files)

    # Setup 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Set equal scaling
    ax.set_box_aspect([1, 1, 1])

    # Set plot limits (adjust based on your data scale)
    #ax.set_xlim([-3, 3])
    ax.set_ylim([-10, 20])
    ax.set_zlim([-10, 10])

    # Initialize cumulative translation vector and rotation matrix
    cumulative_translation_gt = np.zeros(3)
    cumulative_rotation_gt = np.eye(3)
    #cumulative_translation_rec = np.zeros(3)
    #cumulative_rotation_rec = np.eye(3)

    # Plot ground truth frames
    for i, gt_pose in enumerate(gt_poses):
        R_gt = gt_pose[:3, :3]
        t_gt = gt_pose[:3, 3]
        if i == 0:
            cumulative_translation_rec = gt_pose[:3, 3]
            cumulative_rotation_rec = gt_pose[:3, :3]
        print(f'###################  Pair {i}')
        cumulative_rotation_gt = cumulative_rotation_gt @ R_gt
        cumulative_translation_gt += t_gt
        #print(cumulative_translation_gt)
        plot_3d_frame(ax, cumulative_rotation_gt, cumulative_translation_gt, f'GT_Frame_{i}', color=['r', 'g', 'b'])

    # Plot recovered frames
    for i, rec_pose in enumerate(recovered_poses):
        R_rec = rec_pose[:3, :3]
        t_rec = rec_pose[:3, 3]
        t_rec[0] = -rec_pose[2, 3]
        t_rec[1] = rec_pose[1, 3]
        t_rec[2] = 0.0

        cumulative_rotation_rec = cumulative_rotation_rec @ R_rec
        cumulative_translation_rec += np.divide(t_rec,scale[month])
        #print(cumulative_translation_rec)
        plot_3d_frame(ax, cumulative_rotation_rec, cumulative_translation_rec, f'Recovered_Frame_{i}', color=['c', 'm', 'y'])

    # plt.legend()
    plt.show()

if __name__ == "__main__":
    visualize_transformations()

