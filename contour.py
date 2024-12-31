import cv2
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_image(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)
    return image, binary

def get_filtered_contours(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if len(contour) >= 5]

def analyze_contours(contours, image, calibration_factor, max_length_mm):
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

def save_images(output_dir, filename, binary, contours_image, ellipses_image):
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"{filename}-binary.png"), binary)
    cv2.imwrite(os.path.join(output_dir, f"{filename}-contour.png"), contours_image)
    cv2.imwrite(os.path.join(output_dir, f"{filename}-ellipse.png"), ellipses_image)

def process_image(path, output_dir, calibration_factor, max_length_mm):
    image, binary = preprocess_image(path)
    contours = get_filtered_contours(binary)

    contours_image = image.copy()
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)

    ellipses_image, grain_lengths = analyze_contours(
        contours, image, calibration_factor, max_length_mm
    )

    save_images(output_dir, os.path.splitext(os.path.basename(path))[0], binary, contours_image, ellipses_image)

    avg_length = sum(grain_lengths) / len(grain_lengths) if grain_lengths else 0
    return avg_length, len(grain_lengths), grain_lengths

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
    max_length_mm = 4

    files = sorted(
        [f for f in os.listdir(directory) if f.startswith("2")],
        key=lambda x: int("".join(filter(str.isdigit, x)))
    )

    output_dir = "data/contour-output"
    results = []
    all_grain_sizes = {}

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.csv"), mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file", "average", "count"])

        for filename in files:
            filepath = os.path.join(directory, filename)
            avg_length, grain_count, grain_sizes = process_image(filepath, output_dir, calibration_factor, max_length_mm)
            writer.writerow([filename, avg_length, grain_count])
            results.append((filename, avg_length, grain_count))
            all_grain_sizes[filename] = grain_sizes

    generate_histograms(all_grain_sizes, output_dir)
    return results

results = process_directory("data/input")
print(results)
