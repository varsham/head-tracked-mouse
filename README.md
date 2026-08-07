# Head-Tracked Mouse

An assistive-tech mouse replacement that maps head pose and eye/iris position, captured from a webcam, to cursor movement and clicks — no hands required.

## Approach

1. Detect facial landmarks with MediaPipe Face Mesh (face detection + landmarks + iris tracking in one pipeline).
2. Estimate head pose (pitch/yaw/roll) via `cv2.solvePnP` on a subset of landmarks (nose tip, chin, eye corners, mouth corners).
3. Fuse head pose with iris position into a feature vector, and train a model to map that vector to on-screen gaze coordinates.
4. Drive the actual cursor with the model's predictions, smoothed to remove jitter.
5. Add blink-based clicking and a head-position scroll mode.
6. Add an Arduino Nano + buzzer for audio feedback.

## Roadmap

### Week 1 — Camera Calibration & Landmark Detection
- CV stack: MediaPipe Face Mesh (handles face detection, landmarks, and iris tracking — no separate detector needed).
- Camera calibration: start with approximate intrinsics (`fx = fy = frame_width`, principal point centered) to unblock development; do a proper OpenCV checkerboard calibration before Week 5, since intrinsic error shows up directly as pitch/yaw bias.
- Landmarks: nose tip, chin, left/right eye outer corners, left/right mouth corners → `solvePnP` → pitch/yaw/roll.
- Iris position: left/right iris center via MediaPipe's refined landmarks (indices 468-477).

### Week 2 — Collecting Frames to Train Set
- Per calibration point, log the fused feature vector: `[pitch, yaw, roll, left_iris_x, left_iris_y, right_iris_x, right_iris_y]` → target `[target_x, target_y]`.
- 100 frames per point, many points across the screen (denser near edges, where nonlinearity is worse).

### Week 3 — Data Cleaning & Feature Engineering
- Filter out frames with low landmark confidence, partial occlusion, or blinks (EAR outlier).
- Normalize features (head pose and iris position live on different scales).
- Consider deriving iris position relative to eye corner rather than raw image coordinates, for robustness to head translation.
- Train/val/test split, stratified per calibration point.

### Week 4 — Pick Model
- Small tabular regression problem (7 features → 2 targets) — start with linear/polynomial regression or a small MLP/gradient-boosted tree rather than a deep model.
- Compare candidates on held-out pixel error.

### Week 5 — Draw a Dot Based on User Gaze
- Blank-screen test environment showing a dot at predicted gaze position; measure error against known targets.
- Test suite: ~20-25 dots, generates a measurement report.
- Debug overlay mode (toggled by hotkey) for use over real applications: shows pitch, yaw, mode, and current cursor position; supports click-to-add a reference dot with live distance-to-target feedback.

### Week 6 — Replace Dot with Cursor, Fix Jitter
- Drive the OS cursor with PyAutoGUI.
- Smoothing options to evaluate: Kalman filter, dead zones, exponential moving average.
- Re-run the Week 5 test suite to measure improvement.

### Week 7 — Click Implementation
- Calibrate EAR (eye aspect ratio) thresholds: 3s sustained gaze/eyes-open baseline, normal blink, and 1s deliberate blink.
- Confirm the deliberate-blink threshold has real separation from the normal-blink baseline before relying on it for clicks.
- Freeze cursor position during a blink.

### Week 8 — Mode Switching (Cursor ↔ Scroll)
- Scroll mode uses head position only, ignoring eye/iris data.
- Mode-switch trigger: decide between dwell (stare at a fixed point for N seconds) and a blink pattern (e.g. double-blink) — these are different mechanisms and calibrated differently in Week 7.

### Week 9 — Arduino Nano + Buzzer
- Serial communication between the program and an Arduino Nano.
- Buzzer feedback patterns for clicks/mode switches, driven by `Serial.println` commands from the host.

## Tech Stack

- Python, OpenCV, MediaPipe Face Mesh
- PyAutoGUI (cursor control)
- Arduino Nano (buzzer feedback)
