import cv2
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns

def process_color_image(path, output_dir, calibration_factor, max_length_mm):
    image = cv2.imread(path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = cv2.multiply(hsv_image[:, :, 1], 1.2)
    enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    lower_cyan = (30, 100, 100)
    upper_cyan = (85, 255, 255)
    lower_red = (130, 50, 50)
    upper_red = (200, 255, 255)

    mask_cyan = cv2.inRange(hsv_image, lower_cyan, upper_cyan)
    mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
    mask = cv2.bitwise_or(mask_cyan, mask_red)

    output_image = cv2.bitwise_and(enhanced_image, enhanced_image, mask=mask)
    output_image[mask == 0] = [0, 0, 0]
    output_image[mask != 0] = [255, 255, 255]

    gray_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_image = output_image.copy()
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)

    ellipses_image = image.copy()
    grain_lengths = []

    for contour in contours:
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                _, (major_axis, minor_axis), _ = ellipse
                minor_axis_mm = minor_axis * calibration_factor

                if minor_axis_mm <= max_length_mm:
                    cv2.ellipse(ellipses_image, ellipse, (0, 0, 255), 2)
                    grain_lengths.append(minor_axis_mm)
            except cv2.error:
                pass

    save_images(output_dir, os.path.splitext(os.path.basename(path))[0], gray_image, contours_image, ellipses_image)

    avg_length = sum(grain_lengths) / len(grain_lengths) if grain_lengths else 0
    return avg_length, len(grain_lengths), grain_lengths

def save_images(output_dir, filename, binary, contours_image, ellipses_image):
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"{filename}-binary.png"), binary)
    cv2.imwrite(os.path.join(output_dir, f"{filename}-contour.png"), contours_image)
    cv2.imwrite(os.path.join(output_dir, f"{filename}-ellipse.png"), ellipses_image)

def generate_histograms(grain_sizes_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename, grain_sizes in grain_sizes_dict.items():
        plt.figure(figsize=(10, 6))
        plt.hist(grain_sizes, bins=60, color='blue', edgecolor='black', alpha=0.7, density=True)
        sns.kdeplot(grain_sizes, color='red', linewidth=2)

        plt.title(f"Grain Size Distribution: {filename}")
        plt.xlabel("Grain Size (mm)")
        plt.ylabel("Density")
        plt.grid(True)

        plt.savefig(os.path.join(output_dir, f"{filename}-histogram-kde.png"))
        plt.close()

def process_directory(directory):
    calibration_factor = 0.0039016750486215255
    max_length_mm = 3

    files = sorted(
        [f for f in os.listdir(directory) if f.startswith("2")],
        key=lambda x: int("".join(filter(str.isdigit, x)))
    )

    output_dir = "data/color-output"
    results = []
    all_grain_sizes = {}

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "results.csv"), mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file", "average", "count"])

        for filename in files:
            filepath = os.path.join(directory, filename)

            avg_length, grain_count, grain_sizes = process_color_image(filepath, output_dir, calibration_factor, max_length_mm)
            writer.writerow([filename, avg_length, grain_count])
            results.append((filename, avg_length, grain_count))
            all_grain_sizes[filename] = grain_sizes

    generate_histograms(all_grain_sizes, output_dir)

    return results

results = process_directory("data/input")
print("Results:", results)
