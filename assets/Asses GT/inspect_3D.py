import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

month = "may"

def read_file(file_path):
    """
    Reads the transformation matrix from the file and returns rotation and translation components.
    """
    with open(file_path, 'r') as file:
        for line in file:
            elements = line.split()

            # Extract the 4x4 transformation matrix from the last 16 values
            T = np.array([
                [float(elements[22]), float(elements[23]), float(elements[24]), float(elements[25])],
                [float(elements[26]), float(elements[27]), float(elements[28]), float(elements[29])],
                [float(elements[30]), float(elements[31]), float(elements[32]), float(elements[33])],
                [float(elements[34]), float(elements[35]), float(elements[36]), float(elements[37])]
            ])

            # Extract rotation matrix (top-left 3x3) and translation vector (top-right 3x1)
            R = T[:3, :3]
            t = T[:3, 3]

            yield R, t

def plot_3d_frame(ax, R, t, frame_name):
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
    ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color='r', label=f'{frame_name}_x')
    ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color='g', label=f'{frame_name}_y')
    ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color='b', label=f'{frame_name}_z')

def visualize_transformations(file_path):
    """
    Visualizes the transformations extracted from the file.
    """
    # Setup 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Set equal scaling
    ax.set_box_aspect([1, 1, 1])

    # Set plot limits (adjust based on your data scale)
    ax.set_xlim([-10, 50])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])

    # Initialize cumulative translation vector
    cumulative_translation = np.zeros(3)

    # Read and plot each frame's transformation
    for i, (R, t) in enumerate(read_file(file_path)):
        # Accumulate translation (cumulative sum of translations)
        cumulative_translation += t

        # Plot the current frame using cumulative translation
        plot_3d_frame(ax, R, cumulative_translation, f'Frame_{i}')

    #plt.legend()
    plt.show()

if __name__ == "__main__":
    # File path to the text file with transformations
    file_path = f"{month}_test_pairs_gt.txt"

    # Visualize transformations with cumulative translation
    visualize_transformations(file_path)

