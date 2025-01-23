import cv2
import numpy as np

def get_filtered_contours(binary):
    """Get filtered contours based on a binary mask."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if len(contour) >= 5]

def analyze_contours(contours, max_length_mm, calibration_factor):
    """Analyze contours, fit ellipses, and filter based on maximum minor axis length in mm."""
    ellipses = []

    for contour in contours:
        try:
            ellipse = cv2.fitEllipse(contour)
            (x_pos, y_pos), (major_axis, minor_axis), angle = ellipse

            # Convert minor axis length to mm for filtering
            minor_axis_mm = minor_axis * calibration_factor

            if minor_axis_mm <= max_length_mm:
                ellipses.append([x_pos, y_pos, angle, major_axis, minor_axis])
        except cv2.error:
            pass

    return ellipses

def get_elipses(file_path, calibration_factor, max_length_mm):
    """Process an image file to detect and return a list of filtered ellipses."""
    # Read the image
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Unable to load image from path: {file_path}")

    # Convert to HSV and enhance saturation
    saturation_factor = 1
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = cv2.multiply(hsv_image[:, :, 1], saturation_factor)

    # Create masks for cyan and red colors
    lower_cyan = (30, 100, 100)
    upper_cyan = (85, 255, 255)
    lower_red = (130, 50, 50)
    upper_red = (200, 255, 255)

    mask_cyan = cv2.inRange(hsv_image, lower_cyan, upper_cyan)
    mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
    mask = cv2.bitwise_or(mask_cyan, mask_red)

    # Create color contour image
    color_contour = cv2.bitwise_and(image, image, mask=mask)
    color_contour[mask == 0] = [0, 0, 0]
    color_contour[mask != 0] = [255, 255, 255]

    # Convert to grayscale and binary
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128 + 32, 255, cv2.THRESH_BINARY_INV)

    # Invert the color contour
    inverted_color_contour = cv2.bitwise_not(color_contour)
    sharpened_binary = cv2.bitwise_and(inverted_color_contour, inverted_color_contour, mask=binary)

    # Convert sharpened_binary to single-channel
    sharpened_binary_gray = cv2.cvtColor(sharpened_binary, cv2.COLOR_BGR2GRAY)

    # Filter contours and analyze them
    filtered_contours = get_filtered_contours(sharpened_binary_gray)
    ellipses = analyze_contours(filtered_contours, max_length_mm, calibration_factor)

    return ellipses

# Example usage (commented out, for demonstration purposes only):
if __name__ == "__main__":
    file_path = "./data/input/233800-240125051452.jpg"
    calibration_factor = 0.0039016750486215255
    max_length_mm = 0.4
    ellipses = get_elipses(file_path, calibration_factor, max_length_mm)
    print(ellipses)