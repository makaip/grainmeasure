from tkinter import Tk, Frame, Listbox, Scrollbar, VERTICAL, RIGHT, Y, BOTH
from gui import PanZoomCanvas, browse_images, display_image
from process import get_elipses

def setup_ui():
    """Set up the main application window and widgets."""
    app = Tk()
    app.title("Image Browser with Pan and Zoom")
    app.geometry("1300x540")

    list_frame = Frame(app)
    list_frame.pack(side="left", fill="y")

    canvas_frame = Frame(app)
    canvas_frame.pack(side="right", expand=True, fill="both")

    listbox = Listbox(list_frame, width=50, height=30)
    listbox.pack(side="left", fill="y", padx=10, pady=10)

    scrollbar = Scrollbar(list_frame, orient=VERTICAL)
    scrollbar.pack(side=RIGHT, fill=Y)

    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)

    canvas = PanZoomCanvas(canvas_frame, bg="white")
    canvas.pack(expand=True, fill=BOTH)

    listbox.bind("<<ListboxSelect>>", lambda event: display_image(event, listbox, canvas, get_elipses))

    browse_images(listbox)

    return app

if __name__ == "__main__":
    app = setup_ui()
    app.mainloop()
