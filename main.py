import cv2
import os


def preprocess_image(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)
    return image, binary


def find_and_filter_contours(binary, calibration_factor, max_length_mm):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for contour in contours:
        if len(contour) < 5:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w > 0 and h > 0:
            valid_contours.append(contour)
    return valid_contours


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
                grain_shortest_lengths.append(minor_axis)
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
    max_length_mm = 30

    image, binary = preprocess_image(path)
    contours = find_and_filter_contours(binary, calibration_factor, max_length_mm)
    contours_image = image.copy()
    cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
    ellipses_image, grain_shortest_lengths = process_contours(
        contours, image, calibration_factor, max_length_mm
    )
    grain_lengths_real = [length * calibration_factor for length in grain_shortest_lengths]
    average_length_mm = sum(grain_lengths_real) / len(grain_lengths_real) if grain_lengths_real else 0
    filename = os.path.splitext(os.path.basename(path))[0]
    save_images(output_dir, filename, binary, contours_image, ellipses_image)
    return average_length_mm


def process_images_in_directory(directory):
    files = [f for f in os.listdir(directory) if f.startswith("2")]
    files.sort(key=lambda x: int("".join(filter(str.isdigit, x))))
    output_dir = "output"
    results = []
    for filename in files:
        filepath = os.path.join(directory, filename)
        print(f"Processing {filepath}")
        result = count_grains(filepath, output_dir)
        results.append(result)
    return results


results = process_images_in_directory("data")
print(results)
