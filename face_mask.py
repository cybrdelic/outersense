import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def create_bitcoin_laser_eyes(landmarks, frame, frame_count):
    """
    Draws animated Bitcoin laser beams using head pose estimation with MediaPipe landmarks.
    """
    model_points = np.array([
        [-100.0, 50.0, -50.0],  # Left eye outer (33)
        [100.0, 50.0, -50.0],   # Right eye outer (263)
        [0.0, 0.0, 0.0],        # Nose tip (1)
        [0.0, -150.0, -50.0],   # Chin (152)
        [-75.0, -75.0, -50.0],  # Left mouth corner (61)
        [75.0, -75.0, -50.0]    # Right mouth corner (291)
    ], dtype="double")

    image_points = np.array([
        (landmarks[33].x * frame.shape[1], landmarks[33].y * frame.shape[0]),
        (landmarks[263].x * frame.shape[1], landmarks[263].y * frame.shape[0]),
        (landmarks[1].x * frame.shape[1], landmarks[1].y * frame.shape[0]),
        (landmarks[152].x * frame.shape[1], landmarks[152].y * frame.shape[0]),
        (landmarks[61].x * frame.shape[1], landmarks[61].y * frame.shape[0]),
        (landmarks[291].x * frame.shape[1], landmarks[291].y * frame.shape[0])
    ], dtype="double")

    height, width = frame.shape[:2]
    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        print("Head pose estimation failed")
        return frame

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    gaze_direction = rotation_matrix[:, 2]

    left_eye_indices = [33, 246, 161, 160, 159, 158]
    right_eye_indices = [263, 466, 388, 387, 386, 385]
    left_center = (int(np.mean([landmarks[i].x * width for i in left_eye_indices])),
                   int(np.mean([landmarks[i].y * height for i in left_eye_indices])))
    right_center = (int(np.mean([landmarks[i].x * width for i in right_eye_indices])),
                    int(np.mean([landmarks[i].y * height for i in right_eye_indices])))

    left_eye_center_model = np.array([[-65.0, 50.0, -50.0]], dtype="double").T
    right_eye_center_model = np.array([[65.0, 50.0, -50.0]], dtype="double").T

    left_eye_center_cam = np.dot(rotation_matrix, left_eye_center_model) + translation_vector
    right_eye_center_cam = np.dot(rotation_matrix, right_eye_center_model) + translation_vector

    def project_point(point_3d, camera_matrix):
        X, Y, Z = point_3d
        if Z <= 0:
            return left_center if point_3d is left_eye_center_cam.flatten() else right_center
        u = int(camera_matrix[0, 0] * X / Z + camera_matrix[0, 2])
        v = int(camera_matrix[1, 1] * Y / Z + camera_matrix[1, 2])
        return (u, v)

    t = 200
    left_far_cam = left_eye_center_cam.flatten() + t * gaze_direction
    right_far_cam = right_eye_center_cam.flatten() + t * gaze_direction
    left_endpoint = project_point(left_far_cam, camera_matrix)
    right_endpoint = project_point(right_far_cam, camera_matrix)

    laser_color = (0, 165, 255)
    laser_thickness = 5

    for i in range(1, 5):
        intensity = 200 + int(55 * math.sin(frame_count * 0.1))
        glow_color = (0, min(255, intensity), 255)
        current_thickness = max(1, laser_thickness - i + 1)
        cv2.line(frame, left_center, left_endpoint, glow_color, current_thickness)
        cv2.line(frame, right_center, right_endpoint, glow_color, current_thickness)

    bitcoin_radius = 5
    cv2.circle(frame, left_endpoint, bitcoin_radius, (0, 215, 255), -1)
    cv2.circle(frame, right_endpoint, bitcoin_radius, (0, 215, 255), -1)

    cv2.circle(frame, left_center, 10, laser_color, -1)
    cv2.circle(frame, right_center, 10, laser_color, -1)

    return frame

def refine_landmarks(prev_frame_gray, curr_frame_gray, landmarks, landmark_history):
    """
    Refines all 468 MediaPipe landmarks using temporal smoothing and optical flow.
    """
    current_points = np.array([(l.x * curr_frame_gray.shape[1], l.y * curr_frame_gray.shape[0]) for l in landmarks], dtype=np.float32)

    if prev_frame_gray is not None and len(landmark_history) > 0:
        prev_points = np.array(landmark_history[-1], dtype=np.float32).reshape(-1, 1, 2)
        curr_points_flow, status, err = cv2.calcOpticalFlowPyrLK(
            prev_frame_gray, curr_frame_gray, prev_points, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        curr_points_flow = curr_points_flow.reshape(-1, 2)
        current_points = 0.7 * current_points + 0.3 * curr_points_flow

    landmark_history.append(current_points.tolist())
    if len(landmark_history) > 5:
        landmark_history.popleft()

    weights = np.linspace(0.5, 1.0, min(len(landmark_history), 5))
    weights /= weights.sum()
    smoothed_points = np.average(np.array(landmark_history), axis=0, weights=weights[-len(landmark_history):])

    return smoothed_points

def apply_anime_filter(frame, landmarks, frame_count, prev_frame_gray, curr_frame_gray, landmark_history, show_lasers=True, show_video=True):
    """
    Applies the Bitcoin laser eyes effect and visualizes refined MediaPipe landmarks.
    """
    refined_landmarks = refine_landmarks(prev_frame_gray, curr_frame_gray, landmarks.landmark, landmark_history)

    if not show_video:
        frame = np.zeros_like(frame)

    if show_lasers and show_video:
        frame = create_bitcoin_laser_eyes(landmarks.landmark, frame, frame_count)

    # Draw all 468 landmarks as dots
    for x, y in refined_landmarks:
        cv2.circle(frame, (int(x), int(y)), 1, (255, 255, 255), -1)

    # Use MediaPipe's FACEMESH_TESSELATION for a detailed face mesh
    for connection in mp_face_mesh.FACEMESH_TESSELATION:
        idx1, idx2 = connection
        x1, y1 = map(int, refined_landmarks[idx1])
        x2, y2 = map(int, refined_landmarks[idx2])
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    return frame

def apply_mask():
    """
    Captures video from the webcam at index 1 and applies the Bitcoin laser eyes filter.
    """
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Failed to open camera at index 1. Retrying...")
        for _ in range(3):
            cap.release()
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            if cap.isOpened():
                print("Camera opened successfully after retry.")
                break
            time.sleep(1)
        if not cap.isOpened():
            print("Failed to open camera at index 1 after retries. Exiting...")
            return

    mirror_mode = True
    show_lasers = True
    show_video = True
    print("Video is mirrored. Press 'm' (mirror), 'l' (lasers), 'v' (video), 'q' (quit).")

    frame_count = 0
    prev_frame_gray = None
    landmark_history = deque()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Attempting to reinitialize...")
            cap.release()
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame after reinitialization. Exiting...")
                break

        if mirror_mode:
            frame = cv2.flip(frame, 1)

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                frame = apply_anime_filter(frame, landmarks, frame_count, prev_frame_gray, curr_frame_gray, landmark_history, show_lasers, show_video)

        prev_frame_gray = curr_frame_gray.copy()

        mirror_text = "Mirror: ON" if mirror_mode else "Mirror: OFF"
        laser_text = "Lasers: ON" if show_lasers else "Lasers: OFF"
        video_text = "Video: ON" if show_video else "Video: OFF"
        cv2.putText(frame, mirror_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, laser_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, video_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'm' (mirror) | 'l' (lasers) | 'v' (video) | 'q' (quit)", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Bitcoin Laser Eyes Filter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mirror_mode = not mirror_mode
            print(f"Mirror mode: {'ON' if mirror_mode else 'OFF'}")
        elif key == ord('l'):
            show_lasers = not show_lasers
            print(f"Lasers: {'ON' if show_lasers else 'OFF'}")
        elif key == ord('v'):
            show_video = not show_video
            print(f"Video: {'ON' if show_video else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    try:
        apply_mask()
    except Exception as e:
        print(f"An error occurred: {e}")
