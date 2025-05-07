# config.py
import cv2
import mediapipe as mp
import screeninfo

# MediaPipe configurations
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FACE_MESH_CONFIG = {
    'max_num_faces': 1,
    'refine_landmarks': True,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

HANDS_CONFIG = {
    'max_num_hands': 2,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# Camera configurations
CAMERA_INDEX = 1  # Try index 1 first, fallback to 0 if needed
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_API = cv2.CAP_DSHOW  # DirectShow for Windows
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1  # seconds

# Screen and physical constants
AVG_EYE_DISTANCE_MM = 63  # Average inter-pupillary distance
SCREEN_WIDTH_CM = 34.0
SCREEN_HEIGHT_CM = 19.0
MAX_DISTANCE_CM = 200  # Maximum allowable distance for verification

# Visualization toggles (default states)
VISUALIZATION_TOGGLES = {
    'mirror_mode': True,
    'show_lasers': True,
    'show_video': True,
    'show_overlay': False,
    'show_iris': True,
    'show_eyelids': True,
    'show_face_mesh': True,
    'show_gaze_point': True,
    'show_head_axes': True
}

# Calibration steps
CALIBRATION_STEPS = [
    {
        'phase': 'Head Calibration',
        'instruction': 'Center your FACE and look straight into the camera.',
        'duration': 3.0,
        'action': lambda values, p, y, r, gp, ep: values.update({'initial_offset': [p, y, r]})
    },
    {
        'phase': 'Point Anywhere Calibration',
        'instruction': 'Click anywhere on the screen while gazing at the cursor.',
        'duration': None,
        'action': None,
        'min_clicks': 3
    },
    {
        'phase': 'Eye Pitch Calibration',
        'instruction': 'Keep your eyes LEVEL and look straight ahead.',
        'duration': 3.0,
        'action': lambda values, p, y, r, gp, ep: values.update({'pitch_offset': ep})
    }
]

# Display size retrieval
def get_display_size():
    try:
        screen = screeninfo.get_monitors()[0]
        return screen.width, screen.height
    except Exception:
        return 1920, 1080  # Default fallback

# UDP server settings
UDP_HOST = '127.0.0.1'
UDP_PORT = 5555
UDP_TIMEOUT = 0.1

# Window settings
WINDOW_NAME = 'Bitcoin Laser Eyes with Gaze Tracking'
