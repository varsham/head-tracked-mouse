import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Landmark indices used for solvePnP, from MediaPipe's 468-point face mesh.
# These are the standard 6 points used across most mediapipe+solvePnP head
# pose tutorials. If pose looks wrong, sanity-check these against a rendered
# mesh (run this file directly and compare against the overlaid dots).
POSE_LANDMARK_IDXS = {
    "nose_tip": 1,
    "chin": 152,
    "right_eye_outer": 33,
    "left_eye_outer": 263,
    "mouth_right": 61,
    "mouth_left": 291,
}

# Generic average 3D face model (arbitrary units, not a per-user scan) paired
# with the landmarks above. This is the standard reference model used in the
# classic OpenCV head-pose-estimation tutorials -- good enough for pitch/yaw,
# not for metric distance.
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, -330.0, -65.0),     # chin
    (225.0, 170.0, -135.0),   # right eye, outer corner
    (-225.0, 170.0, -135.0),  # left eye, outer corner
    (150.0, -150.0, -125.0),  # mouth, right corner
    (-150.0, -150.0, -125.0), # mouth, left corner
], dtype=np.float64)

# Iris landmark indices, pulled from MediaPipe's own connection sets rather
# than hardcoded, so they can't drift out of sync with the library.
LEFT_IRIS_IDXS = sorted({i for pair in mp_face_mesh.FACEMESH_LEFT_IRIS for i in pair})
RIGHT_IRIS_IDXS = sorted({i for pair in mp_face_mesh.FACEMESH_RIGHT_IRIS for i in pair})


def create_face_mesh():
    """Factory so other scripts can build a FaceMesh instance identically to this one."""
    return mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,  # Includes iris tracking for 478 total points
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def get_camera_matrix(width, height):
    # Approximate intrinsics (see README Week 1 note): fx = fy = frame width,
    # principal point centered. Replace with a checkerboard calibration
    # before relying on absolute pixel accuracy.
    focal_length = width
    center = (width / 2, height / 2)
    return np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)


def _landmark_px(face_landmarks, idx, width, height):
    lm = face_landmarks.landmark[idx]
    return (lm.x * width, lm.y * height)


def estimate_head_pose(face_landmarks, width, height):
    """Returns (pitch, yaw, roll) in degrees, or None if solvePnP fails."""
    image_points = np.array([
        _landmark_px(face_landmarks, POSE_LANDMARK_IDXS["nose_tip"], width, height),
        _landmark_px(face_landmarks, POSE_LANDMARK_IDXS["chin"], width, height),
        _landmark_px(face_landmarks, POSE_LANDMARK_IDXS["right_eye_outer"], width, height),
        _landmark_px(face_landmarks, POSE_LANDMARK_IDXS["left_eye_outer"], width, height),
        _landmark_px(face_landmarks, POSE_LANDMARK_IDXS["mouth_right"], width, height),
        _landmark_px(face_landmarks, POSE_LANDMARK_IDXS["mouth_left"], width, height),
    ], dtype=np.float64)

    camera_matrix = get_camera_matrix(width, height)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    pitch, yaw, roll = angles
    return pitch, yaw, roll


def _centroid_px(face_landmarks, idxs, width, height):
    xs = [face_landmarks.landmark[i].x * width for i in idxs]
    ys = [face_landmarks.landmark[i].y * height for i in idxs]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def get_iris_centers(face_landmarks, width, height):
    """Returns ((left_iris_x, left_iris_y), (right_iris_x, right_iris_y)) in pixel coords."""
    left = _centroid_px(face_landmarks, LEFT_IRIS_IDXS, width, height)
    right = _centroid_px(face_landmarks, RIGHT_IRIS_IDXS, width, height)
    return left, right


def extract_fused_features(face_landmarks, width, height):
    """
    Returns the Week 1 fused feature dict (pitch, yaw, roll, left/right iris
    position), or None if head pose couldn't be estimated for this frame.
    """
    pose = estimate_head_pose(face_landmarks, width, height)
    if pose is None:
        return None
    pitch, yaw, roll = pose
    (left_iris_x, left_iris_y), (right_iris_x, right_iris_y) = get_iris_centers(
        face_landmarks, width, height
    )
    return {
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "left_iris_x": left_iris_x,
        "left_iris_y": left_iris_y,
        "right_iris_x": right_iris_x,
        "right_iris_y": right_iris_y,
    }


def _draw_feature_overlay(image, features):
    lines = [
        f"pitch: {features['pitch']:.1f}  yaw: {features['yaw']:.1f}  roll: {features['roll']:.1f}",
        f"L iris: ({features['left_iris_x']:.0f}, {features['left_iris_y']:.0f})"
        f"  R iris: ({features['right_iris_x']:.0f}, {features['right_iris_y']:.0f})",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            image, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )


def _run_live_preview():
    cap = cv2.VideoCapture(0)

    with create_face_mesh() as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            height, width = image.shape[:2]

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                    )

                    features = extract_fused_features(face_landmarks, width, height)
                    if features is not None:
                        _draw_feature_overlay(image, features)

            cv2.imshow('MediaPipe Face Mesh', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_live_preview()
