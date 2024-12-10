import cv2
import os
import matplotlib.pyplot as plt

def count_grains(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_image = image.copy()
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
    contours_image_rgb = cv2.cvtColor(contours_image, cv2.COLOR_BGR2RGB)

    grain_shortest_lengths = []
    for contour in contours:
        if len(contour) >= 5:  # Minimum points needed for ellipse fitting
            ellipse = cv2.fitEllipse(contour)
            _, (major_axis, minor_axis), _ = ellipse
            grain_shortest_lengths.append(minor_axis)

    # Calibration factor (e.g., mm per pixel)
    calibration_factor = 0.0039016750486215255
    grain_lengths_real = [length * calibration_factor for length in grain_shortest_lengths]

    if grain_lengths_real:
        average_length_mm = sum(grain_lengths_real) / len(grain_lengths_real)
        print("Average shortest length (millimeters):", average_length_mm)
        return average_length_mm
    else:
        print("No grains detected.")
        return 0

def process_images_in_directory(directory):
    files = [f for f in os.listdir(directory) if f.startswith('2')]
    files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    results = []
    for filename in files:
        filepath = os.path.join(directory, filename)
        print(f"Processing {filepath}")
        result = count_grains(filepath)
        results.append(result)
    return results

# Example usage
results = process_images_in_directory('data')
print(results)