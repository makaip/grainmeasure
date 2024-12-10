import cv2
import tkinter as tk

# Global variable to store points
points = []

def get_points(event, x, y, flags, param):
    """
    Mouse callback function to capture two points.
    """
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")
        # Stop after two points
        if len(points) == 2:
            cv2.destroyAllWindows()

def resize_image(image, target_width):
    """
    Resize the image to have a specified width, preserving the aspect ratio.
    """
    height, width = image.shape[:2]
    scale_ratio = target_width / width
    new_dimensions = (target_width, int(height * scale_ratio))
    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    return resized_image, scale_ratio

def get_screen_center():
    """
    Get the screen dimensions and calculate the center position for the window.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    return screen_width // 2, screen_height // 2

def main():
    global points
    
    # Step 1: Load the image
    image_path = input("Enter the path to the calibration image: ")
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image. Please check the path.")
        return
    
    original_height, original_width = image.shape[:2]
    
    # Step 2: Resize the image to have a width of n pixels
    resized_image, scale_ratio = resize_image(image, 1500)
    resized_height, resized_width = resized_image.shape[:2]
    
    # Step 3: Calculate window position to center it
    screen_center_x, screen_center_y = get_screen_center()
    window_x = screen_center_x - (resized_width // 2)
    window_y = screen_center_y - (resized_height // 2)
    
    # Step 4: Display the image and allow the user to click two points
    window_name = "Calibration Image - Click Two Points"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name, window_x, window_y)
    cv2.imshow(window_name, resized_image)
    cv2.setMouseCallback(window_name, get_points)
    cv2.waitKey(0)  # Wait for the user to close the window
    
    if len(points) < 2:
        print("Error: Less than two points selected. Restart the script.")
        return
    
    # Step 5: Map the points from the resized image back to the original image dimensions
    original_points = [(int(x / scale_ratio), int(y / scale_ratio)) for x, y in points]
    print(f"Original Points: {original_points}")
    
    # Step 6: Input the real-world distance
    real_distance = float(input("Enter the real distance between the two points (in millimeters): "))
    
    # Step 7: Compute the distance in pixels in the original image
    pixel_distance = ((original_points[0][0] - original_points[1][0])**2 +
                      (original_points[0][1] - original_points[1][1])**2)**0.5
    print(f"Pixel Distance (Original Image): {pixel_distance}")
    
    # Step 8: Calculate millimeters per pixel
    mm_per_pixel = real_distance / pixel_distance
    print(f"Millimeters per pixel: {mm_per_pixel}")

if __name__ == "__main__":
    main()
