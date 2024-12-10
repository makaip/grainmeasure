import cv2
import os
import matplotlib.pyplot as plt

def count_grains(path, output_dir):
    # Read and process the image
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare images for saving
    contours_image = image.copy()
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
    contours_image_rgb = cv2.cvtColor(contours_image, cv2.COLOR_BGR2RGB)

    ellipses_image = image.copy()
    grain_shortest_lengths = []
    
    # Calibration factor (e.g., mm per pixel)
    calibration_factor = 0.0039016750486215255
    max_length_mm = 30  # Hard cutoff for minor axis in mm
    
    for contour in contours:
        if len(contour) >= 5:  # Minimum points needed for ellipse fitting
            ellipse = cv2.fitEllipse(contour)
            _, (major_axis, minor_axis), _ = ellipse
            minor_axis_mm = minor_axis * calibration_factor
            
            if minor_axis_mm <= max_length_mm:
                cv2.ellipse(ellipses_image, ellipse, (255, 0, 0), 2)
                grain_shortest_lengths.append(minor_axis)

    # Convert pixel lengths to real lengths
    grain_lengths_real = [length * calibration_factor for length in grain_shortest_lengths]

    # Calculate average length
    average_length_mm = sum(grain_lengths_real) / len(grain_lengths_real) if grain_lengths_real else 0
    if grain_lengths_real:
        print("Average shortest length (millimeters):", average_length_mm)
    else:
        print("No grains detected within the size cutoff.")

    # Save images to output folder
    filename = os.path.splitext(os.path.basename(path))[0]
    binary_path = os.path.join(output_dir, f"{filename}-binary.png")
    contours_path = os.path.join(output_dir, f"{filename}-contour.png")
    ellipses_path = os.path.join(output_dir, f"{filename}-ellipse.png")

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(binary_path, binary)
    cv2.imwrite(contours_path, contours_image_rgb)
    cv2.imwrite(ellipses_path, ellipses_image)

    return average_length_mm

def process_images_in_directory(directory):
    files = [f for f in os.listdir(directory) if f.startswith('2')]
    files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    output_dir = "output"
    results = []
    for filename in files:
        filepath = os.path.join(directory, filename)
        print(f"Processing {filepath}")
        result = count_grains(filepath, output_dir)
        results.append(result)
    return results

# Example usage
results = process_images_in_directory('data')
print(results)
