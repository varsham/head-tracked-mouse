"""
Calibration dot display + frame capture for Week 1/2 data collection.

On start, choose a mode:
  - Manual (M): dot waits for you to press SPACE, then a short settle delay
    lets your eyes fixate before the burst capture begins. Good for testing.
  - Automatic (A): each dot settles and captures on its own timer, then
    advances -- no keypress needed. Closer to how an actual user would sit
    through calibration.

During the burst capture (after the settle delay), move your head side to
side while keeping your eyes on the dot -- this gives the model many
different (head pose, iris position) pairs that all map to the same target,
which is what it needs to learn to fuse the two instead of only ever seeing
near-static head position per point.

Each captured frame's fused feature vector (pitch, yaw, roll, left/right
iris position) is computed via facemesh.py and logged as a row in
calibration_points.csv alongside the dot's target coordinates.
Press ESC any time to quit early.

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

ROWS = 10
COLS = 10
MARGIN_PX = 60
DOT_RADIUS = 12
OUTPUT_CSV = "calibration_points.csv"
FRAMES_PER_POINT = 100
SETTLE_MS = 800  # time to let eyes fixate on a new dot before capturing starts

MODE_MANUAL = "manual"
MODE_AUTO = "auto"

LABEL_FONT = ("Helvetica", 16)
STATUS_FONT = ("Helvetica", 48, "bold")  # large enough to catch in peripheral vision


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
        self.mode = None

        self.canvas = tk.Canvas(
            root, width=self.width, height=self.height,
            background="black", highlightthickness=0
        )
        self.canvas.pack()

        self.csv_file = open(OUTPUT_CSV, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "point_index", "frame_index", "timestamp", "target_x", "target_y",
            "pitch", "yaw", "roll", "pose_reprojection_error",
            "left_iris_rel_x", "left_iris_rel_y", "right_iris_rel_x", "right_iris_rel_y",
            "left_ear", "right_ear",
        ])

        self.cap = cv2.VideoCapture(0)
        self.face_mesh = facemesh.create_face_mesh()

        self.root.bind("<Escape>", self.on_quit)
        self.show_mode_selection()

    # -- mode selection ----------------------------------------------------

    def show_mode_selection(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            self.width / 2, self.height / 2,
            text="Press M for Manual mode\nPress A for Automatic mode",
            fill="white", font=("Helvetica", 28), justify="center"
        )
        self.root.bind("<m>", self.start_manual)
        self.root.bind("<a>", self.start_auto)

    def start_manual(self, event):
        self._start_mode(MODE_MANUAL)

    def start_auto(self, event):
        self._start_mode(MODE_AUTO)

    def _start_mode(self, mode):
        self.mode = mode
        self.root.unbind("<m>")
        self.root.unbind("<a>")
        if mode == MODE_MANUAL:
            self.root.bind("<space>", self.on_space)

        self.draw_current_point()
        if mode == MODE_AUTO:
            self.root.after(SETTLE_MS, self.begin_capture)

    # -- drawing -------------------------------------------------------

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

        if status:
            self.canvas.create_text(
                self.width / 2, 60, text=status, fill="yellow", font=STATUS_FONT
            )
        else:
            hint = "hold still" if self.mode == MODE_AUTO else "look at the dot, press SPACE"
            self.canvas.create_text(
                self.width / 2, 30,
                text=f"Point {self.index + 1}/{len(self.points)} -- {hint}",
                fill="white", font=LABEL_FONT
            )

    # -- capture flow ----------------------------------------------------

    def on_space(self, event):
        if self.capturing or self.index >= len(self.points):
            return
        self.capturing = True
        self.draw_current_point(status="Hold still, look at the dot...")
        self.root.after(SETTLE_MS, self.begin_capture)

    def begin_capture(self):
        if self.index >= len(self.points):
            return

        self.capturing = True
        self.draw_current_point(status="Capturing. Move head side to side,\nkeep eyes on the dot")
        self.root.update()  # force the status text to paint before the blocking burst

        logged, missed = self.capture_burst()
        print(f"Point {self.index}: logged {logged} frames, missed {missed}")

        self.capturing = False
        self.index += 1
        self.draw_current_point()

        if self.mode == MODE_AUTO and self.index < len(self.points):
            self.root.after(SETTLE_MS, self.begin_capture)

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
                features["pose_reprojection_error"],
                features["left_iris_rel_x"], features["left_iris_rel_y"],
                features["right_iris_rel_x"], features["right_iris_rel_y"],
                features["left_ear"], features["right_ear"],
            ])
            logged += 1

        self.csv_file.flush()
        return logged, missed

    def on_quit(self, event):
        self.csv_file.close()
        self.face_mesh.close()
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrationApp(root)
    root.mainloop()
