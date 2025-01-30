import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import os
import csv
import scipy.stats as sts


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
output_dir = "data/combo-output/"
os.makedirs(output_dir, exist_ok=True)

# Prepare CSV for results
results_csv_path = os.path.join(output_dir, "results.csv")
with open(results_csv_path, "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["file", "average", "count"])

# Calibration factor and max length in mm
calibration_factor = 0.0039016750486215255
max_length_mm = 0.4

# Store all grain lengths for combined histogram
all_grain_lengths = []

# Process all images in the input directory
input_dir = "data/input/"
for filename in os.listdir(input_dir):
    if filename.startswith("233") and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
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
        # save_image(binary, binary_path)

        # Invert the color contour
        inverted_color_contour = cv2.bitwise_not(color_contour)
        sharpened_binary = cv2.bitwise_and(inverted_color_contour, inverted_color_contour, mask=binary)

        # Save contour image
        contour_path = os.path.join(output_dir, f"{base_name}-contour.jpg")
        # save_image(sharpened_binary, contour_path)

        # Convert sharpened_binary to single-channel
        sharpened_binary_gray = cv2.cvtColor(sharpened_binary, cv2.COLOR_BGR2GRAY)

        # Filter contours and analyze them
        filtered_contours = get_filtered_contours(sharpened_binary_gray)
        ellipses_image, grain_lengths = analyze_contours(filtered_contours, image, calibration_factor, max_length_mm)

        # Save ellipses image
        ellipses_path = os.path.join(output_dir, f"{base_name}-ellipse.jpg")
        # save_image(ellipses_image, ellipses_path)

        # Write results to CSV
        """
        with open(results_csv_path, "a", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([filename, np.mean(grain_lengths) if grain_lengths else 0, len(grain_lengths)])
        """

        # Add to combined grain lengths with color category
        if filename.startswith("233800") or filename.startswith("233801"):
            color = '#00447c'
            core_type = 'Muddy Core'
        else:
            color = '#d31145'
            core_type = 'Sandy Core'
        all_grain_lengths.append((grain_lengths, color, filename, core_type))

        """
        # Plot KDE histogram for current image
        plt.figure(figsize=(10, 6))
        # sns.histplot(grain_lengths, bins=30, kde=False, color=color, alpha=0.1, stat="density")  # Add histogram
        sns.kdeplot(grain_lengths, color=color, bw_adjust=0.5)  # Adjust bandwidth
        plt.title(f"KDE Histogram of Grain Sizes (Minor Axis) - {filename}")
        plt.xlabel("Grain Size (mm)")
        plt.ylabel("Density")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save individual histogram
        histogram_path = os.path.join(output_dir, f"{base_name}-histogram.jpg")
        plt.savefig(histogram_path)
        plt.close()
        """


plt.figure(figsize=(12, 8))

# Iterate over all grain size datasets
for grain_lengths, color, filename, core_type in all_grain_lengths:
    # Compute histogram
    num_bins = 40
    counts, bin_edges = np.histogram(grain_lengths, bins=num_bins)
    bin_width = np.diff(bin_edges)[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total_count = len(grain_lengths)

    # Compute KDE and scale it to match histogram counts
    kde = sts.gaussian_kde(grain_lengths, bw_method=0.1)
    x_vals = np.linspace(bin_edges.min(), bin_edges.max(), 500)
    kde_counts = kde.pdf(x_vals) * total_count * bin_width  # Scale KDE to match counts

    # Plot KDE
    plt.plot(x_vals, kde_counts, color=color, lw=2)

# Plot settings
plt.xlabel("Grain Size (mm)")
plt.ylabel("Count")

# Keep gridlines off but **restore y-axis ticks and labels**
plt.gca().grid(False)

plt.margins(0)
plt.gca().set_ylim(0, 800)
plt.gca().set_xlim(0, 0.3)

# Create zoomed-in inset
ax_main = plt.gca()
ax_inset = inset_axes(ax_main, width="40%", height="40%", loc="upper right")
ax_inset.set_xlim(0.05, 0.15)
ax_inset.set_ylim(0, 200)
ax_inset.margins(0)

# Add the KDE plots to the inset with count scaling
for grain_lengths, color, filename, core_type in all_grain_lengths:
    kde = sts.gaussian_kde(grain_lengths, bw_method=0.1)
    kde_counts = kde.pdf(x_vals) * len(grain_lengths) * bin_width  # Scale KDE
    ax_inset.plot(x_vals, kde_counts, color=color, lw=2)

# Restore y-axis ticks and labels in the inset
ax_inset.grid(False)

# Add legend for core types and move it to the bottom-right corner
handles = [
    plt.Line2D([0], [0], color='#00447c', lw=2, label='Muddy Core'),
    plt.Line2D([0], [0], color='#d31145', lw=2, label='Sandy Core')
]
plt.legend(handles=handles, title="Core Types", loc="upper right")


# Save combined histogram
combined_histogram_path = os.path.join(output_dir, "combined-histogram.jpg")
plt.savefig(combined_histogram_path)
plt.close()