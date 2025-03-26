import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import time

# Initialize MediaPipe Face Mesh and Hand Landmarker
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def create_bitcoin_laser_eyes(landmarks, frame, frame_count, width, height, is_face=True):
    """
    Draws animated Bitcoin laser beams from face or hand landmarks.
    Args:
        landmarks: List of landmarks (MediaPipe face or hand landmarks).
        frame: The current video frame.
        frame_count: Current frame number for animation.
        width, height: Frame dimensions.
        is_face: Boolean to determine if processing face (True) or hand (False).
    """
    if is_face:
        # Face landmarks: Use eye centers
        model_points = np.array([
            [-100.0, 50.0, -50.0],  # Left eye outer (33)
            [100.0, 50.0, -50.0],   # Right eye outer (263)
            [0.0, 0.0, 0.0],        # Nose tip (1)
            [0.0, -150.0, -50.0],   # Chin (152)
            [-75.0, -75.0, -50.0],  # Left mouth corner (61)
            [75.0, -75.0, -50.0]    # Right mouth corner (291)
        ], dtype="double")

        image_points = np.array([
            (landmarks[33].x * width, landmarks[33].y * height),
            (landmarks[263].x * width, landmarks[263].y * height),
            (landmarks[1].x * width, landmarks[1].y * height),
            (landmarks[152].x * width, landmarks[152].y * height),
            (landmarks[61].x * width, landmarks[61].y * height),
            (landmarks[291].x * width, landmarks[291].y * height)
        ], dtype="double")

        left_eye_indices = [33, 246, 161, 160, 159, 158]
        right_eye_indices = [263, 466, 388, 387, 386, 385]
        left_center = (int(np.mean([landmarks[i].x * width for i in left_eye_indices])),
                       int(np.mean([landmarks[i].y * height for i in left_eye_indices])))
        right_center = (int(np.mean([landmarks[i].x * width for i in right_eye_indices])),
                        int(np.mean([landmarks[i].y * height for i in right_eye_indices])))
    else:
        # Hand landmarks: Use index finger tip (landmark 8) for laser origin
        model_points = np.array([
            [0.0, 0.0, 0.0],        # Wrist (0)
            [50.0, -50.0, -20.0],   # Index finger MCP (5)
            [50.0, -100.0, -30.0],  # Index finger tip (8)
            [-50.0, -50.0, -20.0],  # Pinky MCP (17)
            [-50.0, -100.0, -30.0], # Pinky tip (20)
            [0.0, -70.0, -10.0]     # Middle finger MCP (9)
        ], dtype="double")

        image_points = np.array([
            (landmarks[0].x * width, landmarks[0].y * height),
            (landmarks[5].x * width, landmarks[5].y * height),
            (landmarks[8].x * width, landmarks[8].y * height),
            (landmarks[17].x * width, landmarks[17].y * height),
            (landmarks[20].x * width, landmarks[20].y * height),
            (landmarks[9].x * width, landmarks[9].y * height)
        ], dtype="double")

        # Use index finger tip as the laser origin
        left_center = (int(landmarks[8].x * width), int(landmarks[8].y * height))
        right_center = left_center  # For hands, we'll use the same point for simplicity

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
        print(f"Head pose estimation failed for {'face' if is_face else 'hand'}")
        return frame

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    gaze_direction = rotation_matrix[:, 2]

    # For hands, adjust the gaze direction to point "forward" from the finger
    if not is_face:
        gaze_direction = np.array([0, -1, 0])  # Simplified: point upward in image space

    # Use the same 3D model for projection (simplified)
    laser_origin_model = np.array([[0.0, 0.0, 0.0]], dtype="double").T
    laser_origin_cam = np.dot(rotation_matrix, laser_origin_model) + translation_vector

    def project_point(point_3d, camera_matrix):
        X, Y, Z = point_3d
        if Z <= 0:
            return left_center
        u = int(camera_matrix[0, 0] * X / Z + camera_matrix[0, 2])
        v = int(camera_matrix[1, 1] * Y / Z + camera_matrix[1, 2])
        return (u, v)

    t = 200
    laser_far_cam = laser_origin_cam.flatten() + t * gaze_direction
    laser_endpoint = project_point(laser_far_cam, camera_matrix)

    laser_color = (0, 165, 255)
    laser_thickness = 5

    # Draw one laser per hand (or two for face)
    for i in range(1, 5):
        intensity = 200 + int(55 * math.sin(frame_count * 0.1))
        glow_color = (0, min(255, intensity), 255)
        current_thickness = max(1, laser_thickness - i + 1)
        cv2.line(frame, left_center, laser_endpoint, glow_color, current_thickness)
        if is_face:
            cv2.line(frame, right_center, laser_endpoint, glow_color, current_thickness)

    bitcoin_radius = 5
    cv2.circle(frame, laser_endpoint, bitcoin_radius, (0, 215, 255), -1)
    cv2.circle(frame, left_center, 10, laser_color, -1)
    if is_face:
        cv2.circle(frame, right_center, 10, laser_color, -1)

    return frame

def refine_landmarks(prev_frame_gray, curr_frame_gray, landmarks, landmark_history, num_landmarks):
    """
    Refines landmarks using temporal smoothing and optical flow.
    Args:
        num_landmarks: 468 for face, 21 for hand.
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

def apply_anime_filter(frame, face_landmarks, hand_landmarks_list, frame_count, prev_frame_gray, curr_frame_gray, face_history, hand_histories, show_lasers=True, show_video=True):
    """
    Applies the Bitcoin laser eyes effect and visualizes refined MediaPipe landmarks for face and hands.
    """
    width, height = frame.shape[1], frame.shape[0]

    if not show_video:
        frame = np.zeros_like(frame)

    # Process face landmarks
    if face_landmarks:
        refined_face_landmarks = refine_landmarks(prev_frame_gray, curr_frame_gray, face_landmarks.landmark, face_history, num_landmarks=468)

        if show_lasers and show_video:
            frame = create_bitcoin_laser_eyes(face_landmarks.landmark, frame, frame_count, width, height, is_face=True)

        for x, y in refined_face_landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (255, 255, 255), -1)

        for connection in mp_face_mesh.FACEMESH_TESSELATION:
            idx1, idx2 = connection
            x1, y1 = map(int, refined_face_landmarks[idx1])
            x2, y2 = map(int, refined_face_landmarks[idx2])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # Process hand landmarks
    for idx, hand_landmarks in enumerate(hand_landmarks_list):
        if idx >= len(hand_histories):
            hand_histories.append(deque())  # Add new history for this hand

        refined_hand_landmarks = refine_landmarks(prev_frame_gray, curr_frame_gray, hand_landmarks.landmark, hand_histories[idx], num_landmarks=21)

        if show_lasers and show_video:
            frame = create_bitcoin_laser_eyes(hand_landmarks.landmark, frame, frame_count, width, height, is_face=False)

        for x, y in refined_hand_landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (255, 255, 255), -1)

        # Draw hand connections using MediaPipe's hand connections
        for connection in mp_hands.HAND_CONNECTIONS:
            idx1, idx2 = connection
            x1, y1 = map(int, refined_hand_landmarks[idx1])
            x2, y2 = map(int, refined_hand_landmarks[idx2])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    return frame

def apply_mask():
    """
    Captures video from the webcam at index 1 and applies the Bitcoin laser effects to face and hands.
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
    face_history = deque()
    hand_histories = []  # List of deques, one per hand

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

        # Process face landmarks
        face_results = face_mesh.process(frame_rgb)
        face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None

        # Process hand landmarks
        hand_results = hands.process(frame_rgb)
        hand_landmarks_list = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else []

        frame = apply_anime_filter(frame, face_landmarks, hand_landmarks_list, frame_count, prev_frame_gray, curr_frame_gray, face_history, hand_histories, show_lasers, show_video)

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
    hands.close()

if __name__ == "__main__":
    try:
        apply_mask()
    except Exception as e:
        print(f"An error occurred: {e}")
