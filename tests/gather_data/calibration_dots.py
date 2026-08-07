"""
Calibration dot display + frame capture for Week 1/2 data collection.

Shows one dot at a time at a known screen coordinate. Press SPACE to capture
a burst of frames from the webcam, extract each frame's fused feature vector
(pitch, yaw, roll, left/right iris position) via facemesh.py, and log every
frame as a row in calibration_points.csv alongside the dot's target
coordinates. Press ESC to quit early.

Points are laid out on a grid using Chebyshev-Lobatto spacing, which
clusters points near the screen edges and spaces them out near the
center -- matching the plan's note that pose/gaze error is worst at
the edges.
"""

import csv
import math
import sys
import time
import tkinter as tk
from pathlib import Path

import cv2

# facemesh.py lives at the project root, two levels up from this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import facemesh

ROWS = 5
COLS = 10
MARGIN_PX = 60
DOT_RADIUS = 12
OUTPUT_CSV = "calibration_points.csv"
FRAMES_PER_POINT = 100


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
        self.capturing = False

        self.canvas = tk.Canvas(
            root, width=self.width, height=self.height,
            background="black", highlightthickness=0
        )
        self.canvas.pack()

        self.csv_file = open(OUTPUT_CSV, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "point_index", "frame_index", "timestamp", "target_x", "target_y",
            "pitch", "yaw", "roll",
            "left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y",
        ])

        self.cap = cv2.VideoCapture(0)
        self.face_mesh = facemesh.create_face_mesh()

        self.root.bind("<space>", self.on_advance)
        self.root.bind("<Escape>", self.on_quit)

        self.draw_current_point()

    def draw_current_point(self, status=None):
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
        label = status or f"Point {self.index + 1}/{len(self.points)} -- look at the dot, press SPACE"
        self.canvas.create_text(
            self.width / 2, 30, text=label, fill="white", font=("Helvetica", 16)
        )

    def capture_burst(self):
        """Captures FRAMES_PER_POINT frames and writes one CSV row per successful frame."""
        x, y = self.points[self.index]
        logged, missed = 0, 0

        for frame_idx in range(FRAMES_PER_POINT):
            success, frame = self.cap.read()
            if not success:
                missed += 1
                continue

            rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            height, width = rgb.shape[:2]

            if not results.multi_face_landmarks:
                missed += 1
                continue

            features = facemesh.extract_fused_features(
                results.multi_face_landmarks[0], width, height
            )
            if features is None:
                missed += 1
                continue

            self.csv_writer.writerow([
                self.index, frame_idx, time.time(), x, y,
                features["pitch"], features["yaw"], features["roll"],
                features["left_iris_x"], features["left_iris_y"],
                features["right_iris_x"], features["right_iris_y"],
            ])
            logged += 1

        self.csv_file.flush()
        return logged, missed

    def on_advance(self, event):
        if self.capturing or self.index >= len(self.points):
            return

        self.capturing = True
        self.draw_current_point(status="Capturing... hold still")
        self.root.update()  # force the "Capturing..." label to paint before the blocking burst

        logged, missed = self.capture_burst()
        print(f"Point {self.index}: logged {logged} frames, missed {missed}")

        self.capturing = False
        self.index += 1
        self.draw_current_point()

    def on_quit(self, event):
        self.csv_file.close()
        self.face_mesh.close()
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrationApp(root)
    root.mainloop()
