import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry

def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Step 1: Load and preprocess the image
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Step 2: Load SAM model
def load_sam_model(model_type="vit_b", checkpoint_path="sam_vit_b.pth"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    predictor = SamPredictor(sam)
    return predictor

# Step 3: Perform segmentation with SAM
def segment_image_with_sam(predictor, image):
    predictor.set_image(image)  # Set the input image

    # Automatic mask generation
    masks_data = predictor.predict()  # Predict segmentation masks
    
    print("Debugging masks_data:", masks_data)  # Debugging to inspect structure

    # Extract segmentation masks
    masks = []
    for mask_data in masks_data:
        if isinstance(mask_data, dict) and "segmentation" in mask_data:
            masks.append(mask_data["segmentation"])  # Extract segmentation mask
        else:
            print(f"Unexpected mask format: {mask_data}")
    
    return masks

# Step 4: Analyze and measure grain sizes
def measure_grain_sizes(masks, original_image):
    grain_sizes = []
    visualized_image = original_image.copy()

    for mask in masks:
        # Binarize and clean up the mask
        binary_mask = (mask > 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the cleaned mask
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Get minimum enclosing rectangle
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            shortest_length = min(width, height)
            grain_sizes.append(shortest_length)

            # Draw contours on the visualized image
            cv2.drawContours(visualized_image, [contour], -1, (0, 255, 0), 2)

    return grain_sizes, visualized_image

# Step 5: Plot histogram
def plot_histogram(grain_sizes, output_dir, bins=20):
    plt.figure(figsize=(10, 6))
    plt.hist(grain_sizes, bins=bins, color='blue', edgecolor='black')
    plt.title("Grain Size Distribution")
    plt.xlabel("Size (pixels)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "grain_size_histogram.png"))
    plt.close()

# Main function
def main():
    # Set up paths
    image_path = "../data/input/233800-240125051621.jpg"  # Replace with your image path
    output_dir = "../data/ai-output/"
    ensure_output_dir(output_dir)

    # Load and preprocess the image
    original_image = load_image(image_path)

    # Load SAM model and segment the image
    model_type = "vit_h"  # Choose "vit_b", "vit_l", or "vit_h"
    checkpoint_path = "./sam_vit_h_4b8939.pth"  # Path to the SAM checkpoint
    predictor = load_sam_model(model_type, checkpoint_path)

    # Perform segmentation
    masks = segment_image_with_sam(predictor, original_image)

    # Measure grain sizes
    grain_sizes, visualized_image = measure_grain_sizes(masks, original_image)

    # Save the results
    for idx, mask in enumerate(masks):
        cv2.imwrite(os.path.join(output_dir, f"mask_{idx}.png"), mask * 255)
    cv2.imwrite(os.path.join(output_dir, "visualized_image.jpg"), cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR))
    plot_histogram(grain_sizes, output_dir, bins=20)

    print("Processing complete. Results saved to:", output_dir)


if __name__ == "__main__":
    main()
