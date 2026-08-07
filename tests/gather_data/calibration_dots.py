"""
Calibration dot display for Week 1 data collection.

Shows one dot at a time at a known screen coordinate. Press SPACE to log
the current dot's (target_x, target_y) and advance to the next one.
Press ESC to quit early.

Points are laid out on a grid using Chebyshev-Lobatto spacing, which
clusters points near the screen edges and spaces them out near the
center -- matching the plan's note that pose/gaze error is worst at
the edges.
"""

import csv
import math
import time
import tkinter as tk

ROWS = 5
COLS = 10
MARGIN_PX = 60
DOT_RADIUS = 12
OUTPUT_CSV = "calibration_points.csv"


def chebyshev_lobatto_fractions(n):
    """n points in [0, 1], inclusive of both ends, denser near the ends."""
    if n == 1:
        return [0.5]
    return [
        (math.cos(k * math.pi / (n - 1)) + 1) / 2
        for k in range(n)
    ]


def generate_points(width, height, rows, cols, margin):
    x_fracs = chebyshev_lobatto_fractions(cols)
    y_fracs = chebyshev_lobatto_fractions(rows)
    points = []
    for yf in y_fracs:
        for xf in x_fracs:
            x = margin + xf * (width - 2 * margin)
            y = margin + yf * (height - 2 * margin)
            points.append((round(x), round(y)))
    return points


class CalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.attributes("-fullscreen", True)
        self.root.configure(background="black")

        self.width = root.winfo_screenwidth()
        self.height = root.winfo_screenheight()

        self.points = generate_points(self.width, self.height, ROWS, COLS, MARGIN_PX)
        self.index = 0

        self.canvas = tk.Canvas(
            root, width=self.width, height=self.height,
            background="black", highlightthickness=0
        )
        self.canvas.pack()

        self.csv_file = open(OUTPUT_CSV, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["point_index", "timestamp", "target_x", "target_y"])

        self.root.bind("<space>", self.on_advance)
        self.root.bind("<Escape>", self.on_quit)

        self.draw_current_point()

    def draw_current_point(self):
        self.canvas.delete("all")

        if self.index >= len(self.points):
            self.canvas.create_text(
                self.width / 2, self.height / 2,
                text="Done. Press ESC to quit.",
                fill="white", font=("Helvetica", 32)
            )
            return

        x, y = self.points[self.index]
        self.canvas.create_oval(
            x - DOT_RADIUS, y - DOT_RADIUS, x + DOT_RADIUS, y + DOT_RADIUS,
            fill="red", outline=""
        )
        self.canvas.create_text(
            self.width / 2, 30,
            text=f"Point {self.index + 1}/{len(self.points)} "
                 f"-- look at the dot, press SPACE",
            fill="white", font=("Helvetica", 16)
        )

    def on_advance(self, event):
        if self.index >= len(self.points):
            return
        x, y = self.points[self.index]
        self.csv_writer.writerow([self.index, time.time(), x, y])
        self.csv_file.flush()
        self.index += 1
        self.draw_current_point()

    def on_quit(self, event):
        self.csv_file.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrationApp(root)
    root.mainloop()
