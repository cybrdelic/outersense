import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import time
import screeninfo
from scipy.spatial.transform import Rotation

# Initialize MediaPipe Face Mesh and Hand Landmarker
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Average human inter-eye distance in mm
AVG_EYE_DISTANCE_MM = 63

def get_display_size():
    try:
        screen = screeninfo.get_monitors()[0]
        return screen.width, screen.height
    except Exception as e:
        return 640, 480

def calculate_distance(landmarks, width, height, focal_length):
    left_eye = np.array([landmarks[33].x * width, landmarks[33].y * height])
    right_eye = np.array([landmarks[263].x * width, landmarks[263].y * height])
    pixel_distance = np.linalg.norm(left_eye - right_eye)
    distance_mm = (AVG_EYE_DISTANCE_MM * focal_length) / pixel_distance
    return distance_mm / 10  # Convert to cm

def slerp_quat(q1, q2, t):
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0 * q1) + (s1 * q2)

def get_head_orientation(landmarks, width, height, camera_matrix, dist_coeffs, angle_history, quat_history, initial_offset=None):
    model_points = np.array([
        (0.0, 0.0, 0.0),        # Nose tip
        (-225.0, 170.0, -135.0), # Left eye corner
        (225.0, 170.0, -135.0),  # Right eye corner
        (0.0, -330.0, -65.0),    # Chin
        (-150.0, -150.0, -125.0),# Left mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ], dtype="double")
    image_points = np.array([
        (landmarks[1].x * width, landmarks[1].y * height),
        (landmarks[33].x * width, landmarks[33].y * height),
        (landmarks[263].x * width, landmarks[263].y * height),
        (landmarks[152].x * width, landmarks[152].y * height),
        (landmarks[61].x * width, landmarks[61].y * height),
        (landmarks[291].x * width, landmarks[291].y * height)
    ], dtype="double")
    success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if success:
        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6
        if not singular:
            pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
            yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
            roll = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
        else:
            pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
            yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
            roll = 0

        angle_history.append([pitch, yaw, roll])
        if len(angle_history) > 10:
            angle_history.popleft()
        smoothed_angles = np.mean(angle_history, axis=0)
        pitch, yaw, roll = smoothed_angles

        if initial_offset is not None:
            pitch -= initial_offset[0]
            yaw -= initial_offset[1]
            roll -= initial_offset[2]

        quat = Rotation.from_matrix(rmat).as_quat()
        quat_history.append(quat)
        if len(quat_history) > 10:
            quat_history.popleft()
        smoothed_quat = quat_history[0]
        for i in range(1, len(quat_history)):
            smoothed_quat = slerp_quat(smoothed_quat, quat_history[i], 1.0 / (i + 1))
        smoothed_rmat = Rotation.from_quat(smoothed_quat).as_matrix()

        return pitch, yaw, roll, smoothed_rmat, tvec
    return 0.0, 0.0, 0.0, np.eye(3), np.zeros((3, 1))

def calculate_gaze_projection(landmarks, width, height, distance_cm, head_pitch, head_yaw, head_rmat, head_tvec, gaze_history, mirror_mode=True, pitch_offset=None, gaze_offset=None):
    # Compute eye centers
    left_eye_center = np.mean([(landmarks[33].x * width, landmarks[33].y * height),
                               (landmarks[133].x * width, landmarks[133].y * height)], axis=0)
    right_eye_center = np.mean([(landmarks[263].x * width, landmarks[263].y * height),
                                (landmarks[362].x * width, landmarks[362].y * height)], axis=0)
    eye_center_2d = (left_eye_center + right_eye_center) / 2

    # Compute iris centers
    left_iris_center = np.mean([(landmarks[i].x * width, landmarks[i].y * height) for i in [468, 469, 470, 471]], axis=0)
    right_iris_center = np.mean([(landmarks[i].x * width, landmarks[i].y * height) for i in [473, 474, 475, 476]], axis=0)
    iris_center_2d = (left_iris_center + right_iris_center) / 2

    eye_width = np.linalg.norm(right_eye_center - left_eye_center) / 2
    gaze_vec_2d = iris_center_2d - eye_center_2d

    # Compute gaze angles (in degrees) with a multiplier for sensitivity
    eye_yaw = math.degrees(math.atan2(gaze_vec_2d[0], eye_width)) * 1.5
    eye_pitch = math.degrees(math.atan2(gaze_vec_2d[1], eye_width)) * 1.5

    if pitch_offset is not None:
        eye_pitch -= pitch_offset

    # Adjust scaling: use a lower "gaze_range" to increase sensitivity.
    gaze_range = 15.0  # degrees corresponding to half-screen displacement
    yaw_scale = (width / 2) / gaze_range
    pitch_scale = (height / 2) / gaze_range

    gaze_x = (eye_yaw * yaw_scale) + width / 2
    gaze_y = (eye_pitch * pitch_scale) + height / 2

    if gaze_offset is not None:
        gaze_x -= gaze_offset[0]
        gaze_y -= gaze_offset[1]

    print(f"Eye Yaw: {eye_yaw:.1f}, Eye Pitch: {eye_pitch:.1f}, Gaze X: {gaze_x:.1f}, Gaze Y: {gaze_y:.1f}")

    gaze_history.append([gaze_x, gaze_y])
    if len(gaze_history) > 5:
        gaze_history.popleft()
    smoothed_gaze = np.mean(gaze_history, axis=0)
    u = max(0, min(smoothed_gaze[0], width - 1))
    v = max(0, min(smoothed_gaze[1], height - 1))

    return (int(u), int(v)), eye_yaw, eye_pitch

def create_bitcoin_laser_eyes(landmarks, frame, frame_count, width, height, is_face=True):
    if is_face:
        model_points = np.array([
            [-100.0, 50.0, -50.0], [100.0, 50.0, -50.0], [0.0, 0.0, 0.0],
            [0.0, -150.0, -50.0], [-75.0, -75.0, -50.0], [75.0, -75.0, -50.0]
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
        model_points = np.array([
            [0.0, 0.0, 0.0], [50.0, -50.0, -20.0], [50.0, -100.0, -30.0],
            [-50.0, -50.0, -20.0], [-50.0, -100.0, -30.0], [0.0, -70.0, -10.0]
        ], dtype="double")
        image_points = np.array([
            (landmarks[0].x * width, landmarks[0].y * height),
            (landmarks[5].x * width, landmarks[5].y * height),
            (landmarks[8].x * width, landmarks[8].y * height),
            (landmarks[17].x * width, landmarks[17].y * height),
            (landmarks[20].x * width, landmarks[20].y * height),
            (landmarks[9].x * width, landmarks[9].y * height)
        ], dtype="double")
        left_center = (int(landmarks[8].x * width), int(landmarks[8].y * height))
        right_center = left_center

    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        print(f"Pose estimation failed for {'face' if is_face else 'hand'}")
        return frame

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    gaze_direction = rotation_matrix[:, 2]
    if not is_face:
        gaze_direction = np.array([0, -1, 0])

    laser_origin_model = np.array([[0.0, 0.0, 0.0]], dtype="double").T
    laser_origin_cam = np.dot(rotation_matrix, laser_origin_model) + translation_vector
    t = 200
    laser_far_cam = laser_origin_cam.flatten() + t * gaze_direction

    def project_point(point_3d, camera_matrix):
        X, Y, Z = point_3d
        if Z <= 0:
            return left_center
        u = int(camera_matrix[0, 0] * X / Z + camera_matrix[0, 2])
        v = int(camera_matrix[1, 1] * Y / Z + camera_matrix[1, 2])
        return (u, v)

    laser_endpoint = project_point(laser_far_cam, camera_matrix)
    laser_color = (0, 165, 255)
    laser_thickness = 5

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
    current_points = np.array([(l.x * curr_frame_gray.shape[1], l.y * curr_frame_gray.shape[0]) for l in landmarks], dtype=np.float32)
    if prev_frame_gray is not None and len(landmark_history) > 0:
        prev_points = np.array(landmark_history[-1], dtype=np.float32).reshape(-1, 1, 2)
        curr_points_flow, status, err = cv2.calcOpticalFlowPyrLK(
            prev_frame_gray, curr_frame_gray, prev_points, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        curr_points_flow = curr_points_flow.reshape(-1, 2)
        current_points = 0.9 * current_points + 0.1 * curr_points_flow

    landmark_history.append(current_points.tolist())
    if len(landmark_history) > 5:
        landmark_history.popleft()

    weights = np.linspace(0.5, 1.0, min(len(landmark_history), 5))
    weights /= weights.sum()
    smoothed_points = np.average(np.array(landmark_history), axis=0, weights=weights[-len(landmark_history):])
    return smoothed_points

def calibration_wizard(frame, phase, instruction, timer, total_time):
    """
    Draws a semi-transparent overlay with calibration instructions.
    """
    overlay = frame.copy()
    alpha = 0.6  # transparency factor
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (50, 50, 50), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    instruction_text = f"{phase}: {instruction}"
    countdown_text = f"Hold steady for: {timer:.1f}s"

    cv2.putText(frame, instruction_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, countdown_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    progress = int((total_time - timer) / total_time * frame.shape[1])
    cv2.rectangle(frame, (0, 75), (progress, 80), (0, 255, 0), -1)

    # For Gaze Calibration, display a RED dot target at the center.
    if "Gaze" in phase:
        target = (frame.shape[1]//2, frame.shape[0]//2)
        cv2.circle(frame, target, 8, (0, 0, 255), -1)

    return frame

def apply_anime_filter(frame, landmarks, face_landmarks, hand_landmarks_list, frame_count, prev_frame_gray, curr_frame_gray, face_history, hand_histories, show_lasers=True, show_video=True, show_overlay=False, distance_cm=None, pitch=None, yaw=None, roll=None, rmat=None, gaze_point=None, eye_yaw=None, eye_pitch=None):
    width, height = frame.shape[1], frame.shape[0]
    if not show_video and not show_overlay:
        frame = np.zeros_like(frame)

    if face_landmarks:
        refined_face_landmarks = refine_landmarks(prev_frame_gray, curr_frame_gray, face_landmarks.landmark, face_history, num_landmarks=468)
        if show_lasers and show_video and not show_overlay:
            frame = create_bitcoin_laser_eyes(landmarks, frame, frame_count, width, height, is_face=True)

        if not show_overlay:
            for x, y in refined_face_landmarks:
                cv2.circle(frame, (int(x), int(y)), 1, (255, 255, 0), -1)
            for connection in mp_face_mesh.FACEMESH_TESSELATION:
                idx1, idx2 = connection
                x1, y1 = map(int, refined_face_landmarks[idx1])
                x2, y2 = map(int, refined_face_landmarks[idx2])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

        if gaze_point and show_video:
            # Draw a red dot at the gaze point for feedback.
            cv2.circle(frame, gaze_point, 10, (0, 0, 255), 2)
            cv2.putText(frame, "Gaze", (gaze_point[0] + 15, gaze_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if show_overlay and landmarks is not None:
            nose_tip = (int(landmarks[1].x * width), int(landmarks[1].y * height))
            focal_length = width
            camera_matrix = np.array([[focal_length, 0, width/2], [0, focal_length, height/2], [0, 0, 1]], dtype="double")

            axis_length = 100
            axis_points = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(-1, 3)
            axis_points_2d, _ = cv2.projectPoints(axis_points, cv2.Rodrigues(rmat)[0], np.zeros((3, 1)), camera_matrix, np.zeros((4, 1)))
            axis_points_2d = axis_points_2d.reshape(-1, 2).astype(int)

            cv2.line(frame, nose_tip, tuple(axis_points_2d[0]), (0, 0, 255), 2)  # X-axis (roll)
            cv2.line(frame, nose_tip, tuple(axis_points_2d[1]), (0, 255, 0), 2)  # Y-axis (pitch)
            cv2.line(frame, nose_tip, tuple(axis_points_2d[2]), (255, 0, 0), 2)  # Z-axis (yaw)

            left_eye_center = (int(np.mean([landmarks[i].x * width for i in [33, 133]])),
                               int(np.mean([landmarks[i].y * height for i in [33, 133]])))
            right_eye_center = (int(np.mean([landmarks[i].x * width for i in [263, 362]])),
                                int(np.mean([landmarks[i].y * height for i in [263, 362]])))
            eye_length = 150
            left_eye_end = (int(left_eye_center[0] + eye_length * math.sin(math.radians(eye_yaw))),
                            int(left_eye_center[1] - eye_length * math.sin(math.radians(eye_pitch))))
            right_eye_end = (int(right_eye_center[0] + eye_length * math.sin(math.radians(eye_yaw))),
                             int(right_eye_center[1] - eye_length * math.sin(math.radians(eye_pitch))))
            cv2.line(frame, left_eye_center, left_eye_end, (255, 255, 0), 2)
            cv2.line(frame, right_eye_center, right_eye_end, (255, 255, 0), 2)

    if not show_overlay:
        for idx, hand_landmarks in enumerate(hand_landmarks_list):
            if idx >= len(hand_histories):
                hand_histories.append(deque())
            refined_hand_landmarks = refine_landmarks(prev_frame_gray, curr_frame_gray, hand_landmarks.landmark, hand_histories[idx], num_landmarks=21)
            if show_lasers and show_video:
                frame = create_bitcoin_laser_eyes(hand_landmarks.landmark, frame, frame_count, width, height, is_face=False)

            for x, y in refined_hand_landmarks:
                cv2.circle(frame, (int(x), int(y)), 1, (255, 255, 0), -1)
            for connection in mp_hands.HAND_CONNECTIONS:
                idx1, idx2 = connection
                x1, y1 = map(int, refined_hand_landmarks[idx1])
                x2, y2 = map(int, refined_hand_landmarks[idx2])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    if distance_cm is not None:
        cv2.putText(frame, f"Distance: {distance_cm:.1f} cm", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if pitch is not None and yaw is not None and roll is not None:
        cv2.putText(frame, f"Head Pitch: {pitch:.1f} deg", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Head Yaw: {yaw:.1f} deg", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Head Roll: {roll:.1f} deg", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if eye_yaw is not None and eye_pitch is not None:
        cv2.putText(frame, f"Eye Yaw: {eye_yaw:.1f} deg", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Eye Pitch: {eye_pitch:.1f} deg", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

def apply_mask():
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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width, height = 640, 480
    focal_length = width
    display_width, display_height = get_display_size()

    mirror_mode = True
    show_lasers = True
    show_video = True
    show_overlay = False
    print("Video is mirrored. Press 'm' (mirror), 'l' (lasers), 'v' (video), 'a' (overlay), 'c' (calibration wizard), 'q' (quit).")

    frame_count = 0
    prev_frame_gray = None
    face_history = deque()
    hand_histories = []
    angle_history = deque(maxlen=10)
    quat_history = deque(maxlen=10)
    gaze_history = deque(maxlen=5)
    camera_matrix = np.array([[focal_length, 0, width/2], [0, focal_length, height/2], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    # Calibration offsets (default: not calibrated)
    initial_offset = None
    pitch_offset = None
    gaze_offset = None

    # Calibration wizard variables
    calibration_mode = False
    calibration_step_index = 0
    calibration_timer_start = 0
    # Store calibration values in a dict for easy updating
    calibration_values = {"initial_offset": None, "gaze_offset": None, "pitch_offset": None}
    # Define calibration steps with instructions and durations (in seconds)
    calibration_steps = [
        {
            "phase": "Head Calibration",
            "instruction": "Center your FACE and look straight into the camera.",
            "duration": 3.0,
            "action": lambda pitch, yaw, roll, gaze_pt, eye_pitch: calibration_values.update({"initial_offset": [pitch, yaw, roll]})
        },
        {
            "phase": "Gaze Calibration",
            "instruction": "FIX your GAZE on the RED dot at the center of the screen.",
            "duration": 3.0,
            "action": lambda pitch, yaw, roll, gaze_pt, eye_pitch: calibration_values.update({"gaze_offset": [gaze_pt[0] - width/2, gaze_pt[1] - height/2]})
        },
        {
            "phase": "Eye Pitch Calibration",
            "instruction": "Keep your eyes LEVEL and look straight ahead.",
            "duration": 3.0,
            "action": lambda pitch, yaw, roll, gaze_pt, eye_pitch: calibration_values.update({"pitch_offset": eye_pitch})
        }
    ]

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

        face_results = face_mesh.process(frame_rgb)
        face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
        hand_results = hands.process(frame_rgb)
        hand_landmarks_list = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else []

        distance_cm = None
        pitch, yaw, roll, rmat, tvec = None, None, None, None, None
        gaze_point = None
        eye_yaw, eye_pitch = None, None
        landmarks = None
        if face_landmarks:
            landmarks = face_landmarks.landmark
            distance_cm = calculate_distance(landmarks, width, height, focal_length)
            pitch, yaw, roll, rmat, tvec = get_head_orientation(landmarks, width, height, camera_matrix, dist_coeffs, angle_history, quat_history, initial_offset)
            (gaze_point, eye_yaw, eye_pitch) = calculate_gaze_projection(landmarks, width, height, distance_cm, pitch, yaw, rmat, tvec, gaze_history, mirror_mode, pitch_offset, gaze_offset)

        # --- Calibration Wizard Mode ---
        if calibration_mode:
            if face_landmarks:
                current_step = calibration_steps[calibration_step_index]
                current_time = time.time()
                time_elapsed = current_time - calibration_timer_start
                time_left = current_step["duration"] - time_elapsed
                # Overlay the calibration wizard instructions (includes RED dot for gaze calibration)
                frame = calibration_wizard(frame, current_step["phase"], current_step["instruction"], time_left, current_step["duration"])
                if time_left <= 0:
                    # Capture calibration values using current measurements
                    if pitch is not None and yaw is not None and roll is not None and gaze_point is not None and eye_pitch is not None:
                        current_step["action"](pitch, yaw, roll, gaze_point, eye_pitch)
                        print(f"{current_step['phase']} done. Values captured.")
                    else:
                        print("No valid face data for calibration. Try again.")
                    calibration_step_index += 1
                    if calibration_step_index >= len(calibration_steps):
                        # Apply calibration values
                        initial_offset = calibration_values["initial_offset"]
                        gaze_offset = calibration_values["gaze_offset"]
                        pitch_offset = calibration_values["pitch_offset"]
                        calibration_mode = False
                        print("Calibration COMPLETE! Your settings have been updated.")
                    else:
                        calibration_timer_start = time.time()
            else:
                cv2.putText(frame, "No face detected. Align your face in the frame.", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- End Calibration Wizard Mode ---

        frame = apply_anime_filter(frame, landmarks, face_landmarks, hand_landmarks_list, frame_count, prev_frame_gray, curr_frame_gray, face_history, hand_histories, show_lasers, show_video, show_overlay, distance_cm, pitch, yaw, roll, rmat, gaze_point, eye_yaw, eye_pitch)

        prev_frame_gray = curr_frame_gray.copy()

        mirror_text = "Mirror: ON" if mirror_mode else "Mirror: OFF"
        laser_text = "Lasers: ON" if show_lasers else "Lasers: OFF"
        video_text = "Video: ON" if show_video else "Video: OFF"
        overlay_text = "Overlay: ON" if show_overlay else "Overlay: OFF"
        calib_text = "Calibrated" if initial_offset is not None else "Not Calibrated"
        pitch_calib_text = "Pitch Calibrated" if pitch_offset is not None else "Pitch Not Calibrated"
        gaze_calib_text = "Gaze Calibrated" if gaze_offset is not None else "Gaze Not Calibrated"
        cv2.putText(frame, mirror_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, laser_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, video_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, overlay_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, calib_text, (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, pitch_calib_text, (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, gaze_calib_text, (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        instructions = "Press 'm' (mirror) | 'l' (lasers) | 'v' (video) | 'a' (overlay) | 'c' (calibration wizard) | 'q' (quit)"
        cv2.putText(frame, instructions, (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Bitcoin Laser Eyes with Gaze Tracking", frame)

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
        elif key == ord('a'):
            show_overlay = not show_overlay
            print(f"Overlay: {'ON' if show_overlay else 'OFF'}")
        elif key == ord('c'):
            # Start the unified calibration wizard if not already in calibration mode
            if not calibration_mode:
                calibration_mode = True
                calibration_step_index = 0
                calibration_timer_start = time.time()
                calibration_values = {"initial_offset": None, "gaze_offset": None, "pitch_offset": None}
                print("Entering Calibration Wizard Mode. Follow the on-screen instructions.")
        # When calibration_mode is active, ignore individual calibration key presses

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    hands.close()

if __name__ == "__main__":
    try:
        apply_mask()
    except Exception as e:
        print(f"An error occurred: {e}")
