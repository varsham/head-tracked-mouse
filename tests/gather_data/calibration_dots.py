"""
Calibration dot display + frame capture for Week 1/2 data collection.

On start, choose a mode:
  - Manual (M): dot waits for you to press SPACE, then a short settle delay
    lets your eyes fixate before the burst capture begins. Good for testing.
  - Automatic (A): each dot settles and captures on its own timer, then
    advances -- no keypress needed. Closer to how an actual user would sit
    through calibration. Press SPACE any time between points to pause
    (and again to resume) -- useful for a break during a 150-point session.

Keep your head still throughout -- eyes are the only signal driving gaze
prediction for now, so head movement is noise, not useful diversity like it
would be if head pose were a model input.

Each captured frame's fused feature vector (pitch, yaw, roll, left/right
iris position, EAR) is computed via facemesh.py and logged as a row in
calibration_points.csv alongside the dot's target coordinates.
Press ESC any time to quit early.

Points are laid out on a grid using Chebyshev-Lobatto spacing, which
clusters points near the screen edges and spaces them out near the
center -- matching the plan's note that pose/gaze error is worst at
the edges. The grid positions themselves are fixed, but the order the
dots are presented in is shuffled each run, so the session isn't a
predictable row-by-row scan.
"""

import csv
import json
import math
import random
import statistics
import sys
import time
import tkinter as tk
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# facemesh.py lives at the project root, two levels up from this file.
sys.path.insert(0, str(PROJECT_ROOT))
import facemesh

ROWS = 10
COLS = 15
MARGIN_PX = 60
DOT_RADIUS = 12
OUTPUT_CSV = "calibration_points.csv"
FRAMES_PER_POINT = 100
SETTLE_MS = 800  # time to let eyes fixate on a new dot before capturing starts
HEAD_BASELINE_PATH = PROJECT_ROOT / "head_position_baseline.json"
DRIFT_THRESHOLD_DEG = 5.0  # max allowed pitch/yaw distance from this point's own baseline

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
        random.shuffle(self.points)  # same grid, but not scanned in a predictable order
        self.index = 0
        self.capturing = False
        self.mode = None
        self.paused = False  # Auto mode only; toggled by SPACE between points
        self.all_pitch_yaw = []  # every logged frame's (pitch, yaw), for the session head-position baseline

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
        else:
            self.root.bind("<space>", self.on_toggle_pause)

        self.draw_current_point()
        if mode == MODE_AUTO:
            self.schedule_next_capture()

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
            if self.mode == MODE_AUTO:
                hint = "hold still (SPACE to pause)"
            else:
                hint = "look at the dot, press SPACE"
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
        self.draw_current_point(status="Capturing. Keep head still,\nlook at the dot")
        drift_warning_id = self.canvas.create_text(
            self.width / 2, self.height - 60, text="",
            fill="red", font=STATUS_FONT
        )
        self.root.update()  # force the status text to paint before the blocking burst

        logged, missed, drifted = self.capture_burst(drift_warning_id)
        self.canvas.delete(drift_warning_id)
        print(f"Point {self.index}: logged {logged} frames, missed {missed}, drifted {drifted}")

        self.capturing = False
        self.index += 1

        if self.index >= len(self.points):
            self.save_head_baseline()
        self.draw_current_point()

        if self.mode == MODE_AUTO and self.index < len(self.points):
            self.schedule_next_capture()

    def schedule_next_capture(self):
        """Schedules the next begin_capture() call, unless paused -- in
        which case on_toggle_pause() schedules it instead, once resumed."""
        if self.paused:
            return
        self.root.after(SETTLE_MS, self.begin_capture)

    def on_toggle_pause(self, event):
        """Auto mode only. Only takes effect between points -- like ESC,
        it can't interrupt an in-progress capture_burst() since that runs
        synchronously and doesn't yield back to the event loop."""
        if self.mode != MODE_AUTO or self.capturing or self.index >= len(self.points):
            return

        self.paused = not self.paused
        if self.paused:
            self.draw_current_point(status="Paused -- press SPACE to resume")
        else:
            self.draw_current_point()
            self.schedule_next_capture()

    def save_head_baseline(self):
        """
        Median (pitch, yaw) across the whole session -- the head position
        this calibration is only valid for. live_gaze_test.py checks against
        this before starting normal tracking, since the model has no way to
        detect on its own that you're sitting somewhere different now.
        """
        if not self.all_pitch_yaw:
            return
        pitches, yaws = zip(*self.all_pitch_yaw)
        data = {
            "pitch": statistics.median(pitches),
            "yaw": statistics.median(yaws),
        }
        with open(HEAD_BASELINE_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved head position baseline to {HEAD_BASELINE_PATH}")

    def capture_burst(self, drift_warning_id):
        """Captures FRAMES_PER_POINT frames and writes one CSV row per successful frame."""
        x, y = self.points[self.index]
        logged, missed, drifted = 0, 0, 0
        baseline = None
        warning_active = False

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

            # First successfully-detected frame of this point sets the
            # "head still" baseline everything else is compared against.
            if baseline is None:
                baseline = (features["pitch"], features["yaw"])

            drift = math.hypot(features["pitch"] - baseline[0], features["yaw"] - baseline[1])
            is_drifting = drift > DRIFT_THRESHOLD_DEG
            if is_drifting:
                drifted += 1
            if is_drifting != warning_active:
                warning_active = is_drifting
                self.canvas.itemconfig(
                    drift_warning_id,
                    text="Head moved -- hold still!" if warning_active else ""
                )
                self.root.update_idletasks()

            self.csv_writer.writerow([
                self.index, frame_idx, time.time(), x, y,
                features["pitch"], features["yaw"], features["roll"],
                features["pose_reprojection_error"],
                features["left_iris_rel_x"], features["left_iris_rel_y"],
                features["right_iris_rel_x"], features["right_iris_rel_y"],
                features["left_ear"], features["right_ear"],
            ])
            self.all_pitch_yaw.append((features["pitch"], features["yaw"]))
            logged += 1

        self.csv_file.flush()
        return logged, missed, drifted

    def on_quit(self, event):
        self.csv_file.close()
        self.face_mesh.close()
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CalibrationApp(root)
    root.mainloop()
