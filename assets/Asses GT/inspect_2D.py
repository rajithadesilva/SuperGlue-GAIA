import numpy as np
import matplotlib.pyplot as plt

# Define the axis to plot ('x', 'y', or 'z')
axis_to_plot = 'x'

# Mapping axis to index in the 4x4 transformation matrix
axis_indices = {'x': 25, 'y': 29, 'z': 33}
month = "june"
def read_file(file_path, axis_to_plot):
    """
    Reads the transformation matrix from the file and extracts the specified axis component of the translation vector.
    """
    delta_components = []
    with open(file_path, 'r') as file:
        for line in file:
            elements = line.split()

            # Extract the translation component based on the specified axis
            component_index = axis_indices[axis_to_plot]
            component = float(elements[component_index])

            # Store the delta translation component
            delta_components.append(component)

    return delta_components

def calculate_cumulative(delta_components):
    """
    Calculates the cumulative translation by summing delta values.
    """
    cumulative_translation = np.cumsum(delta_components)
    return cumulative_translation

def plot_component(cumulative_components, axis_to_plot):
    """
    Plots the cumulative axis components of the translation vectors.
    """
    plt.figure()
    plt.plot(cumulative_components, marker='o', linestyle='-', color='r')
    plt.title(f'Cumulative {axis_to_plot.upper()} Translation Component')
    plt.xlabel('Index')
    plt.ylabel(f'Cumulative {axis_to_plot.upper()} Translation Value')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # File path to the text file with transformations
    file_path = f"{month}_test_pairs_gt.txt"

    # Read the delta axis components from the file
    delta_components = read_file(file_path, axis_to_plot)

    # Calculate the cumulative translation for the specified axis
    cumulative_components = calculate_cumulative(delta_components)

    # Plot the cumulative axis components
    plot_component(cumulative_components, axis_to_plot)

