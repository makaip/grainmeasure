import cv2
import os
import csv


def preprocess_image(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)
    return image, binary


def find_and_filter_contours(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if len(contour) >= 5]


def process_contours(contours, image, calibration_factor, max_length_mm):
    ellipses_image = image.copy()
    grain_shortest_lengths = []
    for contour in contours:
        try:
            ellipse = cv2.fitEllipse(contour)
            _, (major_axis, minor_axis), _ = ellipse
            minor_axis_mm = minor_axis * calibration_factor
            if minor_axis_mm <= max_length_mm:
                cv2.ellipse(ellipses_image, ellipse, (255, 0, 0), 2)
                grain_shortest_lengths.append(minor_axis_mm)
        except cv2.error:
            continue
    return ellipses_image, grain_shortest_lengths


def save_images(output_dir, filename, binary, contours_image, ellipses_image):
    os.makedirs(output_dir, exist_ok=True)
    binary_path = os.path.join(output_dir, f"{filename}-binary.png")
    contours_path = os.path.join(output_dir, f"{filename}-contour.png")
    ellipses_path = os.path.join(output_dir, f"{filename}-ellipse.png")
    cv2.imwrite(binary_path, binary)
    cv2.imwrite(contours_path, contours_image)
    cv2.imwrite(ellipses_path, ellipses_image)


def count_grains(path, output_dir):
    calibration_factor = 0.0039016750486215255
    max_length_mm = 25

    image, binary = preprocess_image(path)
    contours = find_and_filter_contours(binary)
    contours_image = image.copy()
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
    ellipses_image, grain_shortest_lengths = process_contours(
        contours, image, calibration_factor, max_length_mm
    )
    average_length_mm = sum(grain_shortest_lengths) / len(grain_shortest_lengths) if grain_shortest_lengths else 0
    grain_count = len(grain_shortest_lengths)
    filename = os.path.splitext(os.path.basename(path))[0]
    save_images(output_dir, filename, binary, contours_image, ellipses_image)
    return average_length_mm, grain_count


def process_images_in_directory(directory):
    files = [f for f in os.listdir(directory) if f.startswith("2")]
    files.sort(key=lambda x: int("".join(filter(str.isdigit, x))))
    output_dir = "output"
    csv_path = os.path.join(output_dir, "results.csv")
    results = []

    os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file", "average", "count"])
        for filename in files:
            filepath = os.path.join(directory, filename)
            print(f"Processing {filepath}")
            average_length, grain_count = count_grains(filepath, output_dir)
            writer.writerow([filename, average_length, grain_count])
            results.append((filename, average_length, grain_count))
    return results


results = process_images_in_directory("data")
print(results)
