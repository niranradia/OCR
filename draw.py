import tkinter as tk
from PIL import Image, ImageDraw

prev_x, prev_y = None, None

def draw():
    root = tk.Tk()
    root.title("Draw Something")

    CANVAS_SIZE = 750
    OUTPUT_SIZE = 28

    canvas = tk.Canvas(root, bg="black", width=CANVAS_SIZE, height=CANVAS_SIZE)
    canvas.pack()

    image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "black")
    pil_draw = ImageDraw.Draw(image)

    global prev_x, prev_y
    prev_x, prev_y = None, None


    def start_draw(event):
        global prev_x, prev_y
        prev_x, prev_y = event.x, event.y

    def draw_line(event):
        global prev_x, prev_y
        if prev_x is not None and prev_y is not None:
            canvas.create_line(prev_x, prev_y, event.x, event.y, fill="white", width=50, capstyle=tk.ROUND, smooth=True)
            pil_draw.line((prev_x, prev_y, event.x, event.y), fill="white", width=50)
        prev_x, prev_y = event.x, event.y

    def reset(event):
        global prev_x, prev_y
        prev_x, prev_y = None, None

    def save(event=None): 
        try:
            resized_image = image.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS)
        except AttributeError:
            resized_image = image.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)
        
        resized_image.save("drawing.png")
        print("Image saved as drawing.png")
        root.destroy()

    canvas.bind("<Button-1>", start_draw)
    canvas.bind("<B1-Motion>", draw_line)
    canvas.bind("<ButtonRelease-1>", reset)

    button = tk.Button(root, text="Save", command=save)
    button.pack()


    root.bind('<Return>', save)

    root.mainloop()

if __name__ == '__main__':
    draw()