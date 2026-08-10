import math

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

# Eye landmark indices for EAR (eye aspect ratio), in [corner, top1, top2,
# corner, bottom2, bottom1] order to match the standard EAR formula. This is
# the widely-used 6-point set for MediaPipe's mesh; verify against the live
# preview overlay if EAR values look off for your face/camera angle.
LEFT_EYE_EAR_IDXS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR_IDXS = [33, 160, 158, 133, 153, 144]

# Outer/inner corner pair per eye, reused from the EAR landmark sets above
# (index 0 and 3 of each) to compute an eye-relative iris position.
LEFT_EYE_CORNER_IDXS = (LEFT_EYE_EAR_IDXS[0], LEFT_EYE_EAR_IDXS[3])
RIGHT_EYE_CORNER_IDXS = (RIGHT_EYE_EAR_IDXS[0], RIGHT_EYE_EAR_IDXS[3])


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
    """
    Returns (pitch, yaw, roll, reprojection_error) in degrees/pixels, or None
    if solvePnP fails. reprojection_error is the mean pixel distance between
    the detected landmarks and where the fitted rigid head model predicts
    they should be -- a high value means the landmarks don't agree with a
    rigid head (e.g. partial occlusion, bad detection), even when solvePnP
    "succeeds".
    """
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

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    pitch, yaw, roll = angles

    reprojected, _ = cv2.projectPoints(
        MODEL_POINTS_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs
    )
    reprojected = reprojected.reshape(-1, 2)
    reprojection_error = float(np.mean(np.linalg.norm(reprojected - image_points, axis=1)))

    return pitch, yaw, roll, reprojection_error


def _centroid_px(face_landmarks, idxs, width, height):
    xs = [face_landmarks.landmark[i].x * width for i in idxs]
    ys = [face_landmarks.landmark[i].y * height for i in idxs]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def get_iris_centers(face_landmarks, width, height):
    """Returns ((left_iris_x, left_iris_y), (right_iris_x, right_iris_y)) in pixel coords."""
    left = _centroid_px(face_landmarks, LEFT_IRIS_IDXS, width, height)
    right = _centroid_px(face_landmarks, RIGHT_IRIS_IDXS, width, height)
    return left, right


def _eye_reference(face_landmarks, corner_idxs, width, height):
    outer = _landmark_px(face_landmarks, corner_idxs[0], width, height)
    inner = _landmark_px(face_landmarks, corner_idxs[1], width, height)
    center = ((outer[0] + inner[0]) / 2, (outer[1] + inner[1]) / 2)
    eye_width = math.dist(outer, inner)
    return center, eye_width


def get_relative_iris_positions(face_landmarks, width, height):
    """
    Returns ((left_x, left_y), (right_x, right_y)): each iris center offset
    from that eye's own corner-midpoint, divided by that eye's corner-to-
    corner width. Dimensionless and roughly invariant to head translation,
    rotation, and distance from the camera, unlike raw iris pixel position
    -- isolates eye rotation (gaze) from head movement, which raw pixel
    coordinates conflate.
    """
    (left_iris, right_iris) = get_iris_centers(face_landmarks, width, height)

    left_center, left_width = _eye_reference(face_landmarks, LEFT_EYE_CORNER_IDXS, width, height)
    right_center, right_width = _eye_reference(face_landmarks, RIGHT_EYE_CORNER_IDXS, width, height)

    left_rel = ((left_iris[0] - left_center[0]) / left_width, (left_iris[1] - left_center[1]) / left_width)
    right_rel = ((right_iris[0] - right_center[0]) / right_width, (right_iris[1] - right_center[1]) / right_width)
    return left_rel, right_rel


def _eye_aspect_ratio(face_landmarks, idxs, width, height):
    p1, p2, p3, p4, p5, p6 = [_landmark_px(face_landmarks, i, width, height) for i in idxs]
    vertical = math.dist(p2, p6) + math.dist(p3, p5)
    horizontal = math.dist(p1, p4)
    return vertical / (2.0 * horizontal)


def get_ear(face_landmarks, width, height):
    """Returns (left_ear, right_ear). Lower values mean the eye is more closed."""
    left = _eye_aspect_ratio(face_landmarks, LEFT_EYE_EAR_IDXS, width, height)
    right = _eye_aspect_ratio(face_landmarks, RIGHT_EYE_EAR_IDXS, width, height)
    return left, right


def extract_fused_features(face_landmarks, width, height):
    """
    Returns the fused feature dict (pitch, yaw, roll, left/right eye-relative
    iris position, left/right EAR, pose reprojection error), or None if head
    pose couldn't be estimated for this frame.
    """
    pose = estimate_head_pose(face_landmarks, width, height)
    if pose is None:
        return None
    pitch, yaw, roll, reprojection_error = pose

    (left_iris_rel_x, left_iris_rel_y), (right_iris_rel_x, right_iris_rel_y) = (
        get_relative_iris_positions(face_landmarks, width, height)
    )
    left_ear, right_ear = get_ear(face_landmarks, width, height)

    return {
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "pose_reprojection_error": reprojection_error,
        "left_iris_rel_x": left_iris_rel_x,
        "left_iris_rel_y": left_iris_rel_y,
        "right_iris_rel_x": right_iris_rel_x,
        "right_iris_rel_y": right_iris_rel_y,
        "left_ear": left_ear,
        "right_ear": right_ear,
    }


def _draw_feature_overlay(image, features):
    lines = [
        f"pitch: {features['pitch']:.1f}  yaw: {features['yaw']:.1f}  roll: {features['roll']:.1f}"
        f"  reproj err: {features['pose_reprojection_error']:.1f}px",
        f"L iris rel: ({features['left_iris_rel_x']:.2f}, {features['left_iris_rel_y']:.2f})"
        f"  R iris rel: ({features['right_iris_rel_x']:.2f}, {features['right_iris_rel_y']:.2f})",
        f"L EAR: {features['left_ear']:.2f}  R EAR: {features['right_ear']:.2f}",
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
