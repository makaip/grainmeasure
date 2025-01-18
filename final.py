import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def save_image(image, filename):
    """Save an image to a file."""
    cv2.imwrite(filename, image)

def get_filtered_contours(binary):
    """Get filtered contours based on a binary mask."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if len(contour) >= 5]

def analyze_contours(contours, image, calibration_factor, max_length_mm):
    """Analyze contours, fit ellipses, and filter based on maximum minor axis length."""
    ellipses_image = image.copy()
    grain_lengths = []

    for contour in contours:
        try:
            ellipse = cv2.fitEllipse(contour)
            _, (major_axis, minor_axis), _ = ellipse

            minor_axis_mm = minor_axis * calibration_factor

            if minor_axis_mm <= max_length_mm:
                cv2.ellipse(ellipses_image, ellipse, (255, 0, 0), 2)
                grain_lengths.append(minor_axis_mm)
        except cv2.error:
            pass

    return ellipses_image, grain_lengths

# Ensure output directory exists
output_dir = "data/combo-output"
os.makedirs(output_dir, exist_ok=True)

# Calibration factor and max length in mm
calibration_factor = 0.0039016750486215255
max_length_mm = 5.0

# Process all images in the input directory
input_dir = "data/input"
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        # Convert to HSV and enhance saturation
        saturation_factor = 1
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 1] = cv2.multiply(hsv_image[:, :, 1], saturation_factor)
        enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        # Create masks for cyan and red colors
        lower_cyan = (30, 100, 100)
        upper_cyan = (85, 255, 255)
        lower_red = (130, 50, 50)
        upper_red = (200, 255, 255)

        mask_cyan = cv2.inRange(hsv_image, lower_cyan, upper_cyan)
        mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
        mask = cv2.bitwise_or(mask_cyan, mask_red)

        # Create color contour image
        color_contour = cv2.bitwise_and(enhanced_image, enhanced_image, mask=mask)
        color_contour[mask == 0] = [0, 0, 0]
        color_contour[mask != 0] = [255, 255, 255]

        # Convert to grayscale and binary
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128 + 32, 255, cv2.THRESH_BINARY_INV)

        # Save binary image
        base_name, _ = os.path.splitext(filename)
        binary_path = os.path.join(output_dir, f"{base_name}-binary.jpg")
        save_image(binary, binary_path)

        # Invert the color contour
        inverted_color_contour = cv2.bitwise_not(color_contour)
        sharpened_binary = cv2.bitwise_and(inverted_color_contour, inverted_color_contour, mask=binary)

        # Save contour image
        contour_path = os.path.join(output_dir, f"{base_name}-contour.jpg")
        save_image(sharpened_binary, contour_path)

        # Convert sharpened_binary to single-channel
        sharpened_binary_gray = cv2.cvtColor(sharpened_binary, cv2.COLOR_BGR2GRAY)

        # Filter contours and analyze them
        filtered_contours = get_filtered_contours(sharpened_binary_gray)
        ellipses_image, grain_lengths = analyze_contours(filtered_contours, image, calibration_factor, max_length_mm)

        # Save ellipses image
        ellipses_path = os.path.join(output_dir, f"{base_name}-ellipse.jpg")
        save_image(ellipses_image, ellipses_path)

        # Display results
        print(f"{filename} - Number of valid grains: {len(grain_lengths)}")
        print(f"{filename} - Average grain size (minor axis): {np.mean(grain_lengths):.2f} mm")

        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(grain_lengths, bins=50, color='blue', edgecolor='black', alpha=0.7)
        plt.title(f"Histogram of Grain Sizes (Minor Axis) - {filename}")
        plt.xlabel("Grain Size (mm)")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save histogram
        histogram_path = os.path.join(output_dir, f"{base_name}-histogram.jpg")
        plt.savefig(histogram_path)
        plt.close()
