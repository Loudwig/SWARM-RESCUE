import tkinter as tk
from typing import List, Tuple

class PathCreator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Path Creator")
        
        # Map dimensions (802x602)
        self.WIDTH = 802
        self.HEIGHT = 602
        
        # Create canvas with coordinate system centered
        self.canvas = tk.Canvas(self.root, width=self.WIDTH, height=self.HEIGHT, bg='white')
        self.canvas.pack(pady=20)
        
        # Draw coordinate axes
        self.draw_axes()
        
        # Points storage
        self.points: List[Tuple[int, int]] = [(-332, 263)]  # Starting point
        self.draw_point(-332, 263, "red")  # Mark starting point
        
        # Bind click event
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Add export button
        self.export_button = tk.Button(self.root, text="Export Path", command=self.export_path)
        self.export_button.pack(pady=10)
        
        # Add clear button
        self.clear_button = tk.Button(self.root, text="Clear Path", command=self.clear_path)
        self.clear_button.pack(pady=5)
        
        # Add coordinates label
        self.coord_label = tk.Label(self.root, text="")
        self.coord_label.pack(pady=5)

    def draw_axes(self):
        # Convert to canvas coordinates (origin at center)
        center_x = self.WIDTH // 2
        center_y = self.HEIGHT // 2
        
        # Draw axes
        self.canvas.create_line(0, center_y, self.WIDTH, center_y, fill='gray')  # X axis
        self.canvas.create_line(center_x, 0, center_x, self.HEIGHT, fill='gray')  # Y axis
        
        # Draw grid (optional)
        for x in range(0, self.WIDTH, 100):
            self.canvas.create_line(x, 0, x, self.HEIGHT, fill='lightgray')
        for y in range(0, self.HEIGHT, 100):
            self.canvas.create_line(0, y, self.WIDTH, y, fill='lightgray')

    def canvas_to_world(self, canvas_x: int, canvas_y: int) -> Tuple[int, int]:
        """Convert canvas coordinates to world coordinates"""
        world_x = canvas_x - self.WIDTH // 2
        world_y = -(canvas_y - self.HEIGHT // 2)  # Invert Y axis
        return world_x, world_y

    def world_to_canvas(self, world_x: int, world_y: int) -> Tuple[int, int]:
        """Convert world coordinates to canvas coordinates"""
        canvas_x = world_x + self.WIDTH // 2
        canvas_y = -world_y + self.HEIGHT // 2  # Invert Y axis
        return canvas_x, canvas_y

    def draw_point(self, world_x: int, world_y: int, color: str = "blue"):
        """Draw a point at the given world coordinates"""
        canvas_x, canvas_y = self.world_to_canvas(world_x, world_y)
        self.canvas.create_oval(canvas_x-3, canvas_y-3, canvas_x+3, canvas_y+3, fill=color)

    def draw_line(self, x1: int, y1: int, x2: int, y2: int):
        """Draw a line between two points in world coordinates"""
        canvas_x1, canvas_y1 = self.world_to_canvas(x1, y1)
        canvas_x2, canvas_y2 = self.world_to_canvas(x2, y2)
        self.canvas.create_line(canvas_x1, canvas_y1, canvas_x2, canvas_y2)

    def on_click(self, event):
        """Handle mouse clicks"""
        world_x, world_y = self.canvas_to_world(event.x, event.y)
        
        # Update label with coordinates
        self.coord_label.config(text=f"Last click: ({world_x}, {world_y})")
        
        # Add point to list
        self.points.append((world_x, world_y))
        
        # Draw point
        self.draw_point(world_x, world_y)
        
        # Draw line if there are at least 2 points
        if len(self.points) > 1:
            prev_point = self.points[-2]
            self.draw_line(prev_point[0], prev_point[1], world_x, world_y)

    def export_path(self):
        """Export path in Python list format"""
        path_str = "self.path = [\n"
        for i, (x, y) in enumerate(self.points):
            comment = "# Starting point" if i == 0 else ""
            path_str += f"    ({x}, {y}),{comment}\n"
        path_str += "]"
        
        # Create a new window to display the path
        export_window = tk.Toplevel(self.root)
        export_window.title("Exported Path")
        
        # Add text widget with scroll bar
        text_widget = tk.Text(export_window, height=10, width=50)
        text_widget.pack(padx=10, pady=10)
        text_widget.insert(tk.END, path_str)
        
        # Add copy button
        copy_button = tk.Button(export_window, text="Copy to Clipboard",
                              command=lambda: self.root.clipboard_append(path_str))
        copy_button.pack(pady=5)

    def clear_path(self):
        """Clear the current path"""
        self.canvas.delete("all")
        self.draw_axes()
        self.points = [(-332, 263)]  # Reset to starting point
        self.draw_point(-332, 263, "red")
        self.coord_label.config(text="")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = PathCreator()
    app.run()