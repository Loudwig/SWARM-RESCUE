# Left click : Add points
# Left drag : Delete points
# Ctrl S : Export path to a Python file



import tkinter as tk
from typing import List, Tuple
from tkinter import filedialog

class PathCreator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Path Creator (Ctrl+S to export)")
        self.starting_point = (0,0)
        
        # Map dimensions
        self.WIDTH = 1819
        self.HEIGHT = 1000
        
        # Initialize storage for canvas items
        self.point_items = {}  # Dictionary to store point canvas items
        self.line_items = []   # List to store line canvas items
        
        # Create canvas with coordinate system centered
        self.canvas = tk.Canvas(self.root, width=self.WIDTH, height=self.HEIGHT, bg='white')
        self.canvas.pack(pady=20)
        
        # Draw coordinate axes
        self.draw_axes()
        
        # Points storage
        self.points: List[Tuple[int, int]] = [self.starting_point]
        self.draw_point(*self.starting_point, "red")  # Mark starting point
        
        # Event bindings
        self.canvas.bind("<Button-1>", self.on_click)   # Left click to add points
        self.canvas.bind("<B1-Motion>", self.on_drag)   # Left drag to delete points
        self.drag_threshold = 5  # Pixels distance for deletion detection
        
        # Add buttons and labels
        # self.export_button = tk.Button(self.root, text="Export Path", command=self.export_path)
        # self.export_button.pack(pady=10)
        
        self.clear_button = tk.Button(self.root, text="Clear Path", command=self.clear_path)
        self.clear_button.pack(pady=5)
        
        self.coord_label = tk.Label(self.root, text="")
        self.coord_label.pack(pady=5)
        
        # Add Ctrl+S binding
        self.root.bind('<Control-s>', lambda e: self.export_path())

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
        """Draw a point at the given world coordinates and store its canvas item"""
        canvas_x, canvas_y = self.world_to_canvas(world_x, world_y)
        item = self.canvas.create_oval(canvas_x-3, canvas_y-3, canvas_x+3, canvas_y+3, fill=color)
        self.point_items[(world_x, world_y)] = item
        return item

    def draw_line(self, x1: int, y1: int, x2: int, y2: int):
        """Draw a line between two points and store its canvas item"""
        canvas_x1, canvas_y1 = self.world_to_canvas(x1, y1)
        canvas_x2, canvas_y2 = self.world_to_canvas(x2, y2)
        line = self.canvas.create_line(canvas_x1, canvas_y1, canvas_x2, canvas_y2)
        self.line_items.append(line)
        return line

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

    def on_drag(self, event):
        """Handle mouse drag for point deletion"""
        world_x, world_y = self.canvas_to_world(event.x, event.y)
        
        # Check each point except the starting point
        for i, point in enumerate(self.points[1:], 1):
            # Convert both points to canvas coordinates for distance check
            px, py = self.world_to_canvas(*point)
            
            # Calculate distance between drag position and point
            dist = ((event.x - px) ** 2 + (event.y - py) ** 2) ** 0.5
            
            if dist < self.drag_threshold:
                # Delete point from list
                self.points.pop(i)
                
                # Delete point visual
                if point in self.point_items:
                    self.canvas.delete(self.point_items[point])
                    del self.point_items[point]
                
                # Clear all lines and redraw
                for line in self.line_items:
                    self.canvas.delete(line)
                self.line_items.clear()
                
                # Redraw all lines
                for i in range(len(self.points) - 1):
                    self.draw_line(self.points[i][0], self.points[i][1],
                                 self.points[i+1][0], self.points[i+1][1])
                break

    def export_path(self):
        """Export path to a new Python file"""
        
        # Ask the user for the file path
        file_path = filedialog.asksaveasfilename(defaultextension=".py", filetypes=[("Python files", "*.py")])
        
        if file_path:
            path_str = "path = [\n"
            for i, (x, y) in enumerate(self.points):
                comment = "  # Starting point" if i == 0 else ""
                path_str += f"    ({x}, {y}),{comment}\n"
            path_str += "]\n"
            
            # Write the path to the file
            with open(file_path, "w") as file:
                file.write(path_str)
            
            # Notify the user
            self.coord_label.config(text=f"Path exported to {file_path}")

    def clear_path(self):
        """Clear the current path"""
        self.canvas.delete("all")
        self.draw_axes()
        self.points = [self.starting_point]  # Reset to starting point
        self.point_items.clear()
        self.line_items.clear()
        self.draw_point(*self.starting_point, "red")
        self.coord_label.config(text="")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = PathCreator()
    app.run()