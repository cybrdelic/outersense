import cv2
import dlib
import numpy as np
import math

def create_bitcoin_laser_eyes(landmarks, frame, frame_count):
    """
    Draws animated Bitcoin laser beams shooting out from the eyes using more accurate physics.
    """
    # Get the face orientation from nose and eyes position
    nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
    nose_bridge = (landmarks.part(27).x, landmarks.part(27).y)

    # Get eye landmarks
    left_eye_outer = (landmarks.part(36).x, landmarks.part(36).y)
    left_eye_inner = (landmarks.part(39).x, landmarks.part(39).y)
    right_eye_inner = (landmarks.part(42).x, landmarks.part(42).y)
    right_eye_outer = (landmarks.part(45).x, landmarks.part(45).y)

    # Calculate eye centers more precisely using all eye landmarks
    left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

    left_eye_center = (sum(p[0] for p in left_eye_points) // len(left_eye_points),
                       sum(p[1] for p in left_eye_points) // len(left_eye_points))
    right_eye_center = (sum(p[0] for p in right_eye_points) // len(right_eye_points),
                        sum(p[1] for p in right_eye_points) // len(right_eye_points))

    # Calculate face direction vector using nose position
    face_direction_x = nose_tip[0] - nose_bridge[0]
    face_direction_y = nose_tip[1] - nose_bridge[1]

    # Calculate face plane normal (perpendicular to face)
    # We'll use a simplified approach based on the nose vector
    face_normal_x = face_direction_y  # Perpendicular vector (-y, x)
    face_normal_y = -face_direction_x

    # Normalize the face normal vector
    face_normal_magnitude = math.sqrt(face_normal_x**2 + face_normal_y**2)
    if face_normal_magnitude > 0:
        face_normal_x /= face_normal_magnitude
        face_normal_y /= face_normal_magnitude

    # Calculate left and right eye direction vectors
    # Base these on both face orientation and the eye position relative to nose
    left_dir_x = face_normal_x + 0.5 * (left_eye_center[0] - nose_bridge[0]) / 100
    left_dir_y = face_normal_y + 0.5 * (left_eye_center[1] - nose_bridge[1]) / 100

    right_dir_x = face_normal_x + 0.5 * (right_eye_center[0] - nose_bridge[0]) / 100
    right_dir_y = face_normal_y + 0.5 * (right_eye_center[1] - nose_bridge[1]) / 100

    # Normalize the direction vectors
    left_dir_magnitude = math.sqrt(left_dir_x**2 + left_dir_y**2)
    if left_dir_magnitude > 0:
        left_dir_x /= left_dir_magnitude
        left_dir_y /= left_dir_magnitude

    right_dir_magnitude = math.sqrt(right_dir_x**2 + right_dir_y**2)
    if right_dir_magnitude > 0:
        right_dir_x /= right_dir_magnitude
        right_dir_y /= right_dir_magnitude

    # Calculate gaze angle from face and eye orientation
    gaze_weight = 0.7  # Weight for face orientation vs. eye direction

    # Add some eye-direction based on inner/outer eye positions
    left_eye_direction_x = left_eye_inner[0] - left_eye_outer[0]
    left_eye_direction_y = left_eye_inner[1] - left_eye_outer[1]

    right_eye_direction_x = right_eye_outer[0] - right_eye_inner[0]
    right_eye_direction_y = right_eye_outer[1] - right_eye_inner[1]

    # Normalize eye direction vectors
    left_eye_dir_magnitude = math.sqrt(left_eye_direction_x**2 + left_eye_direction_y**2)
    if left_eye_dir_magnitude > 0:
        left_eye_direction_x /= left_eye_dir_magnitude
        left_eye_direction_y /= left_eye_dir_magnitude

    right_eye_dir_magnitude = math.sqrt(right_eye_direction_x**2 + right_eye_direction_y**2)
    if right_eye_dir_magnitude > 0:
        right_eye_direction_x /= right_eye_dir_magnitude
        right_eye_direction_y /= right_eye_dir_magnitude

    # Combine face normal and eye direction for final laser direction
    left_laser_x = gaze_weight * face_normal_x + (1-gaze_weight) * left_eye_direction_x
    left_laser_y = gaze_weight * face_normal_y + (1-gaze_weight) * left_eye_direction_y

    right_laser_x = gaze_weight * face_normal_x + (1-gaze_weight) * right_eye_direction_x
    right_laser_y = gaze_weight * face_normal_y + (1-gaze_weight) * right_eye_direction_y

    # Normalize the final vectors
    left_laser_magnitude = math.sqrt(left_laser_x**2 + left_laser_y**2)
    if left_laser_magnitude > 0:
        left_laser_x /= left_laser_magnitude
        left_laser_y /= left_laser_magnitude

    right_laser_magnitude = math.sqrt(right_laser_x**2 + right_laser_y**2)
    if right_laser_magnitude > 0:
        right_laser_x /= right_laser_magnitude
        right_laser_y /= right_laser_magnitude

    # Laser beam properties
    laser_color = (0, 165, 255)  # Orange-yellow color for Bitcoin
    laser_thickness = 3
    laser_length = frame.shape[1] * 1.5  # Make beams longer than the frame width

    # Add subtle movement to the beams (less jitter, more fluid)
    phase_shift = frame_count * 0.07
    left_laser_x += 0.05 * math.sin(phase_shift)
    left_laser_y += 0.05 * math.cos(phase_shift * 0.7)
    right_laser_x += 0.05 * math.sin(phase_shift * 0.8)
    right_laser_y += 0.05 * math.cos(phase_shift * 0.9)

    # Re-normalize after adding movement
    left_laser_magnitude = math.sqrt(left_laser_x**2 + left_laser_y**2)
    left_laser_x /= left_laser_magnitude
    left_laser_y /= left_laser_magnitude

    right_laser_magnitude = math.sqrt(right_laser_x**2 + right_laser_y**2)
    right_laser_x /= right_laser_magnitude
    right_laser_y /= right_laser_magnitude

    # Calculate the final beam endpoints
    left_dx = int(left_laser_x * laser_length)
    left_dy = int(left_laser_y * laser_length)
    right_dx = int(right_laser_x * laser_length)
    right_dy = int(right_laser_y * laser_length)

    # Determine endpoints for the beams
    left_endpoint = (left_eye_center[0] + left_dx, left_eye_center[1] + left_dy)
    right_endpoint = (right_eye_center[0] + right_dx, right_eye_center[1] + right_dy)

    # Store the laser origin and endpoints
    left_center = left_eye_center
    right_center = right_eye_center

    # Draw the laser beams with gradients
    for i in range(1, 5):
        # Vary the color intensity for a glowing effect
        intensity = 200 + int(55 * math.sin(frame_count * 0.1))
        glow_color = (0, min(255, intensity), 255)

        # Ensure thickness is always between 1 and MAX_THICKNESS (usually 255)
        current_thickness = max(1, min(255, laser_thickness - i + 1))

        # Draw the main laser beams
        cv2.line(frame, left_center, left_endpoint, glow_color, current_thickness)
        cv2.line(frame, right_center, right_endpoint, glow_color, current_thickness)

    # Draw a bitcoin symbol at the end of each laser for effect
    bitcoin_radius = 5
    cv2.circle(frame, left_endpoint, bitcoin_radius, (0, 215, 255), -1)
    cv2.circle(frame, right_endpoint, bitcoin_radius, (0, 215, 255), -1)

    # Add some glow around the eyes
    cv2.circle(frame, left_center, 10, laser_color, -1)
    cv2.circle(frame, right_center, 10, laser_color, -1)

    return frame

def apply_anime_filter(frame, landmarks, frame_count):
    """
    Applies Bitcoin laser eyes effect to the detected face with subtle facial landmark visualization.
    """
    frame = create_bitcoin_laser_eyes(landmarks, frame, frame_count)

    # Draw facial landmarks as white dots
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)  # Small white dots

    # Draw subtle lines for each facial feature group
    # Jaw line
    for n in range(0, 16):
        x1 = landmarks.part(n).x
        y1 = landmarks.part(n).y
        x2 = landmarks.part(n+1).x
        y2 = landmarks.part(n+1).y
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # Eyebrows
    for n in range(17, 21):
        x1 = landmarks.part(n).x
        y1 = landmarks.part(n).y
        x2 = landmarks.part(n+1).x
        y2 = landmarks.part(n+1).y
        cv2.line(frame, (x1, y1), (x2, y2), (180, 0, 0), 1)  # Red lines for eyebrows

    for n in range(22, 26):
        x1 = landmarks.part(n).x
        y1 = landmarks.part(n).y
        x2 = landmarks.part(n+1).x
        y2 = landmarks.part(n+1).y
        cv2.line(frame, (x1, y1), (x2, y2), (180, 0, 0), 1)  # Red lines for eyebrows

    # Nose bridge
    for n in range(27, 30):
        x1 = landmarks.part(n).x
        y1 = landmarks.part(n).y
        x2 = landmarks.part(n+1).x
        y2 = landmarks.part(n+1).y
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # Nose tip
    for n in range(31, 35):
        x1 = landmarks.part(n).x
        y1 = landmarks.part(n).y
        x2 = landmarks.part(n+1).x
        y2 = landmarks.part(n+1).y
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.line(frame, (landmarks.part(35).x, landmarks.part(35).y),
             (landmarks.part(31).x, landmarks.part(31).y), (255, 255, 255), 1)

    # Eyes (left and right)
    for n in range(36, 41):
        x1 = landmarks.part(n).x
        y1 = landmarks.part(n).y
        x2 = landmarks.part(n+1).x
        y2 = landmarks.part(n+1).y
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.line(frame, (landmarks.part(41).x, landmarks.part(41).y),
             (landmarks.part(36).x, landmarks.part(36).y), (255, 255, 255), 1)

    for n in range(42, 47):
        x1 = landmarks.part(n).x
        y1 = landmarks.part(n).y
        x2 = landmarks.part(n+1).x
        y2 = landmarks.part(n+1).y
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.line(frame, (landmarks.part(47).x, landmarks.part(47).y),
             (landmarks.part(42).x, landmarks.part(42).y), (255, 255, 255), 1)

    # Lips outer and inner contour
    for n in range(48, 59):
        x1 = landmarks.part(n).x
        y1 = landmarks.part(n).y
        x2 = landmarks.part(n+1).x
        y2 = landmarks.part(n+1).y
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.line(frame, (landmarks.part(59).x, landmarks.part(59).y),
             (landmarks.part(48).x, landmarks.part(48).y), (255, 255, 255), 1)

    for n in range(60, 67):
        x1 = landmarks.part(n).x
        y1 = landmarks.part(n).y
        x2 = landmarks.part(n+1).x
        y2 = landmarks.part(n+1).y
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.line(frame, (landmarks.part(67).x, landmarks.part(67).y),
             (landmarks.part(60).x, landmarks.part(60).y), (255, 255, 255), 1)

    return frame

def apply_mask():
    """
    Captures video from the webcam and overlays the Bitcoin laser eyes filter in real time.
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Try external webcam (index 1); fallback to default (index 0)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Failed to open camera at index 1. Trying default camera...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Failed to open any camera. Exiting...")
            return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            frame = apply_anime_filter(frame, landmarks, frame_count)

        cv2.imshow("Bitcoin Laser Eyes Filter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        apply_mask()
    except Exception as e:
        print(f"An error occurred: {e}")
