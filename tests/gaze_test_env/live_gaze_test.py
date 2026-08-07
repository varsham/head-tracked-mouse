"""
Live gaze-prediction test environment (Week 5).

Fullscreen blank window. Click anywhere to place the reference target dot
(red). A second dot (green) tracks the model's live gaze prediction from
the webcam, updated every frame. Corner text shows the target position, the
predicted gaze position, and the distance between them in pixels -- the
same metric used in Week 4's model comparison.

Press ESC to quit.

Usage: python live_gaze_test.py [model_path]
Defaults to models/random_forest.pkl (the Week 4 cross-validation winner).
"""

import sys
import tkinter as tk
from pathlib import Path

import cv2
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import facemesh
from train_models import FEATURE_COLUMNS

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.pkl"
TICK_MS = 20
TARGET_DOT_RADIUS = 12
GAZE_DOT_RADIUS = 12
INFO_FONT = ("Helvetica", 16)


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

        self.cap = cv2.VideoCapture(0)
        self.face_mesh = facemesh.create_face_mesh()

        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<Escape>", self.on_quit)

        self.tick()

    def on_click(self, event):
        self.target_pos = (event.x, event.y)

    def predict_gaze(self):
        success, frame = self.cap.read()
        if not success:
            return None

        rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        height, width = rgb.shape[:2]

        if not results.multi_face_landmarks:
            return None

        features = facemesh.extract_fused_features(results.multi_face_landmarks[0], width, height)
        if features is None:
            return None

        row = [[features[c] for c in FEATURE_COLUMNS]]
        pred_x, pred_y = self.model.predict(row)[0]
        return (pred_x, pred_y)

    def tick(self):
        gaze = self.predict_gaze()
        if gaze is not None:
            self.gaze_pos = gaze

        self.draw()
        self.root.after(TICK_MS, self.tick)

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
