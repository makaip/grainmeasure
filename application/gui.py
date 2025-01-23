import os
from tkinter import Tk, Canvas, Listbox, Scrollbar, Frame, VERTICAL, RIGHT, Y, BOTH
from tkinter.messagebox import showerror
from PIL import Image, ImageTk
from math import cos, sin, radians

class PanZoomCanvas(Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.original_image = None
        self.image_tk = None
        self.ellipses = []
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.image_width = 0
        self.image_height = 0
        self.selected_ellipse = None

        # Bind mouse and keyboard events
        self.bind("<ButtonPress-1>", self.start_pan_or_select)
        self.bind("<B1-Motion>", self.pan)
        self.bind("<MouseWheel>", self.zoom)
        self.bind("<Delete>", self.delete_selected_ellipse)

        self.start_x = 0
        self.start_y = 0

    def start_pan_or_select(self, event):
        self.start_x = event.x
        self.start_y = event.y

        # Check if an ellipse is clicked
        clicked_ellipse = self.find_closest(event.x, event.y)
        if clicked_ellipse and "ellipse" in self.gettags(clicked_ellipse):
            self.select_ellipse(clicked_ellipse)
        else:
            self.selected_ellipse = None  # Deselect if no ellipse is clicked

    def pan(self, event):
        dx = event.x - self.start_x
        dy = event.y - self.start_y

        self.offset_x += dx
        self.offset_y += dy

        self.start_x = event.x
        self.start_y = event.y

        self.redraw()

    def zoom(self, event):
        """Zoom in or out based on mouse wheel, centered around the cursor."""
        zoom_factor = 1.1 if event.delta > 0 else 0.9

        # Calculate cursor position relative to image
        cursor_x = (event.x - self.offset_x) / self.scale_factor
        cursor_y = (event.y - self.offset_y) / self.scale_factor

        # Adjust scale factor
        self.scale_factor *= zoom_factor

        # Adjust offsets to keep zoom centered on the cursor
        self.offset_x = event.x - cursor_x * self.scale_factor
        self.offset_y = event.y - cursor_y * self.scale_factor

        self.redraw()

    def redraw(self):
        """Redraw the canvas with the current transformations."""
        self.delete("all")
        if self.original_image:
            # Scale the image dynamically
            scaled_width = int(self.image_width * self.scale_factor)
            scaled_height = int(self.image_height * self.scale_factor)
            scaled_image = self.original_image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
            self.image_tk = ImageTk.PhotoImage(scaled_image)
            self.create_image(self.offset_x, self.offset_y, image=self.image_tk, anchor="nw")
        self.draw_ellipses(scaled_width, scaled_height)

    def set_image(self, img, ellipses, scale_factor):
        self.original_image = img
        self.ellipses = ellipses
        self.image_width, self.image_height = img.size
        self.offset_x = 0
        self.offset_y = 0
        self.scale_factor = scale_factor * 4
        self.redraw()

    def draw_ellipses(self, scaled_width, scaled_height):
        """Draw ellipses with scaling and panning transformations."""
        if not self.ellipses:
            return

        # Dynamically calculate the scaling compensation
        compensation_factor = 0.25
        image_scale_x = compensation_factor * scaled_width / self.image_width
        image_scale_y = compensation_factor * scaled_height / self.image_height

        for i, ellipse in enumerate(self.ellipses):
            x_pos, y_pos, rotation, x_scale, y_scale = ellipse

            # Scale positions and dimensions using the image's scaling factors
            x_center = x_pos * image_scale_x + self.offset_x
            y_center = y_pos * image_scale_y + self.offset_y
            x_radius = (x_scale / 2) * image_scale_x
            y_radius = (y_scale / 2) * image_scale_y
            angle_rad = radians(rotation)

            vertices = []
            for theta in range(0, 360, 10):
                theta_rad = radians(theta)
                x = x_radius * cos(theta_rad)
                y = y_radius * sin(theta_rad)

                rotated_x = x * cos(angle_rad) - y * sin(angle_rad)
                rotated_y = x * sin(angle_rad) + y * cos(angle_rad)

                vertices.append((x_center + rotated_x, y_center + rotated_y))

            color = "red" if self.selected_ellipse == i else "blue"
            tags = ("ellipse", f"ellipse_{i}")
            self.create_polygon(vertices, outline=color, width=1, fill="", tags=tags)

    def select_ellipse(self, ellipse_id):
        """Select the clicked ellipse and make it red."""
        tags = self.gettags(ellipse_id)
        if tags and "ellipse" in tags:
            self.selected_ellipse = int(tags[1].split("_")[1])  # Extract ellipse index
        self.redraw()

    def delete_selected_ellipse(self, event):
        """Delete the currently selected ellipse."""
        if self.selected_ellipse is not None:
            del self.ellipses[self.selected_ellipse]
            self.selected_ellipse = None
            self.redraw()

def browse_images(listbox):
    """Populate the Listbox with image filenames from the input directory."""
    directory = "./data/input/"
    try:
        images = [
            f
            for f in os.listdir(directory)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        ]
        if not images:
            raise FileNotFoundError("No images found in the directory.")
        for img in images:
            listbox.insert("end", img)
    except Exception as e:
        showerror("Error", str(e))

def scale_image(img, width=960):
    """Scale the image to a specified width while maintaining aspect ratio."""
    w_percent = width / float(img.size[0])
    height = int(float(img.size[1]) * w_percent)
    return img.resize((width, height), Image.Resampling.LANCZOS)

def display_image(event, listbox, canvas, get_elipses):
    """Display the selected image with ellipses drawn on it."""
    selected = listbox.curselection()
    if not selected:
        return

    file_name = listbox.get(selected[0])
    file_path = os.path.join("./data/input/", file_name)

    try:
        img = Image.open(file_path)
        original_image_width, original_image_height = img.size
        img = scale_image(img)

        calibration_factor = 0.0039016750486215255
        max_length_mm = 4
        ellipses = get_elipses(file_path, calibration_factor, max_length_mm)
        scale_factor = img.width / original_image_width

        canvas.set_image(img, ellipses, scale_factor)

    except Exception as e:
        showerror("Error", f"Could not process image: {str(e)}")
