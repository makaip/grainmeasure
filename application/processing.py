import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io

def contouralg(image, calibration_factor=0.0039016750486215255, max_length_mm=3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if len(contour) >= 5]

    contours_image = image.copy()
    cv2.drawContours(contours_image, filtered_contours, -1, (0, 255, 0), 2)

    ellipses_image = image.copy()
    grain_lengths = []

    for contour in filtered_contours:
        try:
            ellipse = cv2.fitEllipse(contour)
            _, (major_axis, minor_axis), _ = ellipse
            minor_axis_mm = minor_axis * calibration_factor

            if minor_axis_mm <= max_length_mm:
                cv2.ellipse(ellipses_image, ellipse, (255, 0, 0), 2)
                grain_lengths.append(minor_axis_mm)
        except cv2.error:
            pass

    avg_length = sum(grain_lengths) / len(grain_lengths) if grain_lengths else 0

    plt.figure(figsize=(10, 6))
    plt.hist(grain_lengths, bins=60, color='blue', edgecolor='black', alpha=0.7, density=True)
    sns.kdeplot(grain_lengths, color='red', linewidth=2)
    plt.title("Grain Size Distribution")
    plt.xlabel("Grain Size (mm)")
    plt.ylabel("Density")
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    histogram_image = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
    buf.close()
    plt.close()

    return {
        "binary_image": binary,
        "contours_image": contours_image,
        "ellipses_image": ellipses_image,
        "histogram_image": histogram_image,
        "average_length": avg_length,
        "grain_count": len(grain_lengths),
    }

def coloralg(image, saturation_factor=1.2, calibration_factor=0.0039016750486215255, max_length_mm=3):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = cv2.multiply(hsv_image[:, :, 1], saturation_factor)
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

    avg_length = sum(grain_lengths) / len(grain_lengths) if grain_lengths else 0

    plt.figure(figsize=(10, 6))
    plt.hist(grain_lengths, bins=60, color='blue', edgecolor='black', alpha=0.7, density=True)
    sns.kdeplot(grain_lengths, color='red', linewidth=2)
    plt.title("Grain Size Distribution")
    plt.xlabel("Grain Size (mm)")
    plt.ylabel("Density")
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    histogram_image = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
    buf.close()
    plt.close()

    return {
        "binary_image": gray_image,
        "contours_image": contours_image,
        "ellipses_image": ellipses_image,
        "histogram_image": histogram_image,
        "average_length": avg_length,
        "grain_count": len(grain_lengths),
    }
