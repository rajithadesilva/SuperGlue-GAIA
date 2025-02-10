import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_exg(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or unable to load.")
        return None, None, None

    # Convert the image from BGR to RGB (for consistent color space)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Split into R, G, B channels
    R, G, B = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]

    # Compute Excess Green Index (ExG)
    exg = 2 * G - R - B

    # Normalize ExG for display purposes
    exg_normalized = cv2.normalize(exg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    exg_normalized = exg_normalized.astype(np.uint8)

    # Calculate the average ExG value
    avg_exg = np.mean(exg)

    # Convert the original image to grayscale and calculate its average pixel value
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_grayscale = np.mean(grayscale_image)

    return exg_normalized, avg_exg, avg_grayscale, image_rgb

def process_multiple_images(base_path, months, image_ids, output_path):
    exg_images = []
    avg_exg_values = []
    avg_grayscale_values = []
    original_images = []

    for month, image_id in zip(months, image_ids):
        image_name = f"{month}_color_image_{image_id}.png"
        image_path = f"{base_path}/{month}/rgb/{image_name}"
        exg_image, avg_exg, avg_grayscale, original_image = compute_exg(image_path)
        if exg_image is not None:
            exg_images.append(exg_image)
            avg_exg_values.append(avg_exg)
            avg_grayscale_values.append(avg_grayscale)
            original_images.append(original_image)

    # Tile images vertically with original and ExG images side by side
    tiled_image = None
    for idx, (original_image, exg_image, avg_exg, avg_grayscale) in enumerate(zip(original_images, exg_images, avg_exg_values, avg_grayscale_values)):
        # Calculate the ratio of ExG to grayscale average
        ratio = avg_exg / avg_grayscale if avg_grayscale != 0 else 0

        # Add text overlay to RGB image
        original_with_text = original_image.copy()
        cv2.putText(original_with_text, f"ExG/Gray Ratio: {ratio:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(original_with_text, f"Avg ExG: {avg_exg:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(original_with_text, f"Avg Gray: {avg_grayscale:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2, cv2.LINE_AA)

        # Stack original with text and ExG images horizontally
        combined = np.hstack((cv2.cvtColor(original_with_text, cv2.COLOR_RGB2BGR), cv2.cvtColor(exg_image, cv2.COLOR_GRAY2BGR)))

        if tiled_image is None:
            tiled_image = combined
        else:
            tiled_image = np.vstack((tiled_image, combined))

    # Resize the final image to have a height of 3000 pixels while preserving aspect ratio
    if tiled_image is not None:
        height, width = tiled_image.shape[:2]
        new_height = 3000
        new_width = int((new_height / height) * width)
        resized_image = cv2.resize(tiled_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Save the resized image
        cv2.imwrite(output_path, resized_image)
        print(f"Tiled image saved at: {output_path}")

        # Display the resized image
        plt.figure(figsize=(10, 20))
        plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Tiled Original and ExG Images with Ratio Values')
        plt.show()

# Define parameters
base_path = 'assets/long'
months = ['march', 'april', 'may', 'june', 'september']
image_ids = [562, 510, 566, 498, 580]  # Corresponding image IDs for each month
output_path = 'tiled_exg_output.png'

process_multiple_images(base_path, months, image_ids, output_path)

