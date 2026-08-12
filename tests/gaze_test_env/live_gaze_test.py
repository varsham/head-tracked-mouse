"""
Live gaze-prediction test environment (Week 5).

If head_position_baseline.json exists (saved by calibration_dots.py at the
end of a calibration session), your live pitch/yaw is checked against that
session's baseline on every single frame, not just at startup -- since the
model has no head-pose input anymore and is only valid for the head position
it was calibrated at. Whenever you drift more than ALIGN_THRESHOLD_DEG away
(at start, or any time mid-session), a blocking alignment screen takes over
until you're back within range for ALIGN_STABLE_FRAMES frames in a row, then
normal tracking resumes automatically. No baseline file -> skips the check
entirely, tracking runs continuously as before.

Tracking mode: fullscreen blank window. Click anywhere to place the
reference target dot (red). A second dot (green) tracks the model's live
gaze prediction from the webcam, updated every frame. Corner text shows the
target position, the predicted gaze position, and the distance between them
in pixels -- the same metric used in Week 4's model comparison.

Press ESC to quit.

Usage: python live_gaze_test.py [model_path]
Defaults to models/polynomial_regression.pkl (the Week 4 cross-validation winner).
"""

import json
import sys
import tkinter as tk
from pathlib import Path

import cv2
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import facemesh
from train_models import FEATURE_COLUMNS

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "polynomial_regression.pkl"
HEAD_BASELINE_PATH = PROJECT_ROOT / "head_position_baseline.json"
TICK_MS = 20
TARGET_DOT_RADIUS = 12
GAZE_DOT_RADIUS = 12
INFO_FONT = ("Helvetica", 16)
ALIGN_FONT = ("Helvetica", 40, "bold")
ALIGN_THRESHOLD_DEG = 5.0  # matches DRIFT_THRESHOLD_DEG in calibration_dots.py
ALIGN_STABLE_FRAMES = 15  # consecutive aligned frames needed before tracking starts


def load_head_baseline():
    if not HEAD_BASELINE_PATH.exists():
        return None
    with open(HEAD_BASELINE_PATH) as f:
        data = json.load(f)
    return data["pitch"], data["yaw"]


class GazeTestApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model

        self.root.attributes("-fullscreen", True)
        self.root.configure(background="black")
        self.width = root.winfo_screenwidth()
        self.height = root.winfo_screenheight()

        self.canvas = tk.Canvas(
            root, width=self.width, height=self.height,
            background="black", highlightthickness=0
        )
        self.canvas.pack()

        self.target_pos = (self.width // 2, self.height // 2)
        self.gaze_pos = None

        self.head_baseline = load_head_baseline()
        self.aligned = self.head_baseline is None
        self.aligned_streak = 0
        self.last_drift = None

        self.cap = cv2.VideoCapture(0)
        self.face_mesh = facemesh.create_face_mesh()

        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<Escape>", self.on_quit)

        self.tick()

    def on_click(self, event):
        self.target_pos = (event.x, event.y)

    def capture_features(self):
        success, frame = self.cap.read()
        if not success:
            return None

        rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        height, width = rgb.shape[:2]

        if not results.multi_face_landmarks:
            return None

        return facemesh.extract_fused_features(results.multi_face_landmarks[0], width, height)

    def predict_gaze(self, features):
        row = [[features[c] for c in FEATURE_COLUMNS]]
        pred_x, pred_y = self.model.predict(row)[0]
        return (pred_x, pred_y)

    def tick(self):
        features = self.capture_features()

        if self.head_baseline is not None:
            self.update_alignment(features)

        if self.aligned:
            if features is not None:
                self.gaze_pos = self.predict_gaze(features)
            self.draw()
        else:
            self.draw_alignment()

        self.root.after(TICK_MS, self.tick)

    def update_alignment(self, features):
        """
        Runs every tick, not just at startup -- if the user drifts away from
        the calibration head position mid-session, this drops self.aligned
        back to False so tick() blocks on the alignment screen again until
        they re-settle for ALIGN_STABLE_FRAMES in a row.
        """
        if features is None:
            self.last_drift = None
            self.aligned_streak = 0
            self.aligned = False
            return

        base_pitch, base_yaw = self.head_baseline
        drift = ((features["pitch"] - base_pitch) ** 2 + (features["yaw"] - base_yaw) ** 2) ** 0.5
        self.last_drift = drift

        if drift <= ALIGN_THRESHOLD_DEG:
            self.aligned_streak += 1
            if self.aligned_streak >= ALIGN_STABLE_FRAMES:
                self.aligned = True
        else:
            self.aligned_streak = 0
            self.aligned = False

    def draw_alignment(self):
        self.canvas.delete("all")

        if self.last_drift is None:
            text = "No face detected"
            color = "red"
        elif self.last_drift <= ALIGN_THRESHOLD_DEG:
            text = f"Aligned -- hold still  ({self.aligned_streak}/{ALIGN_STABLE_FRAMES})"
            color = "lime"
        else:
            text = f"Match your calibration head position\ndrift: {self.last_drift:.1f} deg (need < {ALIGN_THRESHOLD_DEG})"
            color = "red"

        self.canvas.create_text(
            self.width / 2, self.height / 2, text=text,
            fill=color, font=ALIGN_FONT, justify="center"
        )

    def draw(self):
        self.canvas.delete("all")

        tx, ty = self.target_pos
        self.canvas.create_oval(
            tx - TARGET_DOT_RADIUS, ty - TARGET_DOT_RADIUS,
            tx + TARGET_DOT_RADIUS, ty + TARGET_DOT_RADIUS,
            fill="red", outline=""
        )

        if self.gaze_pos is not None:
            gx, gy = self.gaze_pos
            self.canvas.create_oval(
                gx - GAZE_DOT_RADIUS, gy - GAZE_DOT_RADIUS,
                gx + GAZE_DOT_RADIUS, gy + GAZE_DOT_RADIUS,
                fill="lime", outline=""
            )
            distance = ((gx - tx) ** 2 + (gy - ty) ** 2) ** 0.5
            info = (
                "red = target, green = predicted gaze\n"
                f"Target:   ({tx}, {ty})\n"
                f"Gaze:     ({gx:.0f}, {gy:.0f})\n"
                f"Distance: {distance:.1f} px"
            )
        else:
            info = (
                "red = target, green = predicted gaze\n"
                f"Target:   ({tx}, {ty})\n"
                "Gaze:     (no face detected)\n"
                "Distance: --"
            )

        self.canvas.create_text(
            20, 20, text=info, fill="white", font=INFO_FONT, anchor="nw"
        )

    def on_quit(self, event):
        self.face_mesh.close()
        self.cap.release()
        self.root.destroy()


def main():
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    model = joblib.load(model_path)

    root = tk.Tk()
    GazeTestApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
