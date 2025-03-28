import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import time
import screeninfo
from scipy.spatial.transform import Rotation
import threading
import socket
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Constants
AVG_EYE_DISTANCE_MM = 63
SCREEN_WIDTH = 34.0  # cm
SCREEN_HEIGHT = 19.0  # cm
MAX_DISTANCE_CM = 200  # Maximum allowable distance

# Shared variables
running = True
click_count = 0
verification_states = {
    "distance": {"valid": False, "msg": "✗ Distance not computed"},
    "head_pose": {"valid": False, "msg": "✗ Pose not computed"},
    "gaze_vector": {"valid": False, "msg": "✗ Gaze not computed"},
    "gaze_projection": {"valid": False, "msg": "✗ Projection not computed"},
    "udp_transmission": {"valid": False, "msg": "✗ UDP not sent"}
}

class GazeVisualization:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.head_position = np.array([0, 0, 60])
        self.gaze_vector = np.array([0, 0, -1])
        self.screen_width, self.screen_height = SCREEN_WIDTH, SCREEN_HEIGHT
        self.screen_distance = 60
        self._setup_visualization()
        self.server_thread = threading.Thread(target=self._udp_server)
        self.server_thread.daemon = True

    def _setup_visualization(self):
        self.ax.set_xlabel('X (cm)')
        self.ax.set_ylabel('Z (cm)')
        self.ax.set_zlabel('Y (cm)')
        self.ax.set_title('3D Gaze Tracking Visualization')
        self.ax.set_xlim([-40, 40])
        self.ax.set_ylim([0, 100])
        self.ax.set_zlim([-30, 30])
        self.laptop_body = self._create_laptop_body()
        self.laptop_screen = self._create_laptop_screen()
        self.head = self._create_head()
        gaze_xs = [self.head_position[0], self.head_position[0] + self.gaze_vector[0] * 100]
        gaze_ys = [self.head_position[2], self.head_position[2] + self.gaze_vector[2] * 100]
        gaze_zs = [self.head_position[1], self.head_position[1] + self.gaze_vector[1] * 100]
        self.gaze_line, = self.ax.plot(gaze_xs, gaze_ys, gaze_zs, 'r-', linewidth=2, label='Gaze')
        self.gaze_point, = self.ax.plot([], [], [], 'ro', markersize=8)
        self.distance_text = self.ax.text2D(0.02, 0.95, f"Distance: {self.screen_distance:.1f} cm",
                                          transform=self.ax.transAxes)
        self.gaze_text = self.ax.text2D(0.02, 0.90, "Gaze: ", transform=self.ax.transAxes)
        self.ax.view_init(elev=20, azim=-60)
        self.ax.legend()

    def _create_laptop_body(self):
        width, height = 30, 20
        depth = 1.5
        x, y, z = 0, 0, 20
        v = np.array([
            [x-width/2, y, z], [x+width/2, y, z], [x+width/2, y, z+depth],
            [x-width/2, y, z+depth], [x-width/2, y+height, z]
        ])
        laptop_body = self.ax.plot_surface(
            np.array([[v[0,0], v[1,0]], [v[3,0], v[2,0]]]),
            np.array([[v[0,1], v[1,1]], [v[3,1], v[2,1]]]),
            np.array([[v[0,2], v[1,2]], [v[3,2], v[2,2]]]),
            color='gray', alpha=0.7
        )
        return laptop_body

    def _create_laptop_screen(self):
        width, height = self.screen_width, self.screen_height
        x, base_height, z = 0, 1, 20
        y = base_height + height/2
        tilt_angle = 75
        rad_angle = math.radians(tilt_angle)
        z_offset = height/2 * math.cos(rad_angle)
        y_offset = height/2 * math.sin(rad_angle)
        corners = np.array([
            [x-width/2, y-y_offset, z-z_offset], [x+width/2, y-y_offset, z-z_offset],
            [x+width/2, y+y_offset, z+z_offset], [x-width/2, y+y_offset, z+z_offset],
            [x-width/2, y-y_offset, z-z_offset]
        ])
        self.ax.plot(corners[:,0], corners[:,2], corners[:,1], 'b-', linewidth=2)
        x_grid = np.array([[corners[0,0], corners[1,0]], [corners[3,0], corners[2,0]]])
        y_grid = np.array([[corners[0,1], corners[1,1]], [corners[3,1], corners[2,1]]])
        z_grid = np.array([[corners[0,2], corners[1,2]], [corners[3,2], corners[2,2]]])
        screen_surface = self.ax.plot_surface(x_grid, z_grid, y_grid, color='blue', alpha=0.3)
        self.screen_corners = corners
        self.screen_normal = np.array([0, math.sin(rad_angle), math.cos(rad_angle)])
        self.screen_center = np.array([x, y, z])
        return screen_surface

    def _create_head(self):
        head_radius = 10
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = self.head_position[0] + head_radius * np.outer(np.cos(u), np.sin(v))
        y = self.head_position[1] + head_radius * np.outer(np.sin(u), np.sin(v))
        z = self.head_position[2] + head_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        head = self.ax.plot_surface(x, z, y, color='tan', alpha=0.7)
        eye_offset = 3
        left_eye = self.head_position + np.array([-eye_offset, 0, 0])
        right_eye = self.head_position + np.array([eye_offset, 0, 0])
        self.left_eye, = self.ax.plot([left_eye[0]], [left_eye[2]], [left_eye[1]], 'ko', markersize=5)
        self.right_eye, = self.ax.plot([right_eye[0]], [right_eye[2]], [right_eye[1]], 'ko', markersize=5)
        return head

    def _calculate_gaze_screen_intersection(self):
        corners = self.screen_corners[:4]
        v1 = corners[1] - corners[0]
        v2 = corners[3] - corners[0]
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        ray_origin = self.head_position
        ray_direction = self.gaze_vector
        ndotu = np.dot(normal, ray_direction)
        if abs(ndotu) < 1e-6:
            return None
        w = ray_origin - corners[0]
        t = -np.dot(normal, w) / ndotu
        if t < 0:
            return None
        intersection = ray_origin + t * ray_direction
        origin = corners[0]
        basis1 = v1 / np.linalg.norm(v1)
        basis2 = v2 / np.linalg.norm(v2)
        screen_vector = intersection - origin
        u = np.dot(screen_vector, basis1)
        v = np.dot(screen_vector, basis2)
        if (0 <= u <= np.linalg.norm(v1)) and (0 <= v <= np.linalg.norm(v2)):
            return intersection
        return None

    def update(self, head_position=None, gaze_vector=None, screen_info=None):
        if head_position is not None:
            self.head_position = np.array(head_position) / 10.0
        if gaze_vector is not None:
            self.gaze_vector = np.array(gaze_vector)
            if np.linalg.norm(self.gaze_vector) > 0:
                self.gaze_vector = self.gaze_vector / np.linalg.norm(self.gaze_vector)
        if screen_info is not None:
            if 'width' in screen_info and 'height' in screen_info:
                self.screen_width = screen_info['width'] / 10.0
                self.screen_height = screen_info['height'] / 10.0
            if 'distance' in screen_info:
                self.screen_distance = screen_info['distance'] / 10.0
        self._update_head()
        self._update_gaze()
        self._update_screen()
        self._update_text()

    def _update_head(self):
        eye_offset = 3
        left_eye_pos = self.head_position + np.array([-eye_offset, 0, 0])
        right_eye_pos = self.head_position + np.array([eye_offset, 0, 0])
        self.left_eye.set_data([left_eye_pos[0]], [left_eye_pos[2]])
        self.left_eye.set_3d_properties([left_eye_pos[1]])
        self.right_eye.set_data([right_eye_pos[0]], [right_eye_pos[2]])
        self.right_eye.set_3d_properties([right_eye_pos[1]])

    def _update_gaze(self):
        gaze_length = 100
        gaze_end = self.head_position + self.gaze_vector * gaze_length
        self.gaze_line.set_data([self.head_position[0], gaze_end[0]],
                               [self.head_position[2], gaze_end[2]])
        self.gaze_line.set_3d_properties([self.head_position[1], gaze_end[1]])
        intersection = self._calculate_gaze_screen_intersection()
        if intersection is not None:
            self.gaze_point.set_data([intersection[0]], [intersection[2]])
            self.gaze_point.set_3d_properties([intersection[1]])
            corners = self.screen_corners[:4]
            origin = corners[0]
            width_vec = corners[1] - corners[0]
            height_vec = corners[3] - corners[0]
            screen_vector = intersection - origin
            u = np.dot(screen_vector, width_vec) / np.dot(width_vec, width_vec)
            v = np.dot(screen_vector, height_vec) / np.dot(height_vec, height_vec)
            self.gaze_text.set_text(f"Gaze: ({u*100:.1f}%, {v*100:.1f}%)")
        else:
            self.gaze_point.set_data([], [])
            self.gaze_point.set_3d_properties([])
            self.gaze_text.set_text("Gaze: Not on screen")

    def _update_screen(self):
        if hasattr(self, 'laptop_screen') and self.laptop_screen is not None:
            try:
                self.laptop_screen.remove()
            except:
                pass
        self.laptop_screen = self._create_laptop_screen()

    def _update_text(self):
        self.distance_text.set_text(f"Distance: {np.linalg.norm(self.head_position):.1f} cm")

    def _udp_server(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_socket.bind(('127.0.0.1', 5555))
        server_socket.settimeout(0.1)
        print("UDP server listening on 127.0.0.1:5555")
        while True:
            try:
                data, addr = server_socket.recvfrom(1024)
                try:
                    values = [float(x) for x in data.decode().split(',')]
                    head_position = values[0:3]
                    gaze_vector = values[3:6]
                    screen_info = {'width': values[6], 'height': values[7], 'distance': values[8]}
                    self.update(head_position, gaze_vector, screen_info)
                except Exception as e:
                    print(f"Error processing data: {e}")
            except socket.timeout:
                continue
            except Exception as e:
                print(f"UDP server error: {e}")
                time.sleep(1)

    def animate(self, i):
        return (self.gaze_line, self.gaze_point, self.left_eye, self.right_eye,
                self.distance_text, self.gaze_text)

    def start(self):
        self.server_thread.start()
        anim = FuncAnimation(self.fig, self.animate, interval=100, blit=True)
        plt.show()

def mouse_callback(event, x, y, flags, param):
    global click_count, calibration_values, calibration_mode, calibration_step_index
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Click detected at ({x}, {y})")
        print(f"Calibration mode: {calibration_mode}, Step: {calibration_step_index}")
        if calibration_mode and calibration_step_index == 1:
            print("Processing click in Point Anywhere Calibration")
            gaze_point = param.get('gaze_point')
            if gaze_point is not None:
                gaze_x, gaze_y = gaze_point
                display_width, display_height = param['display_size']
                frame_width, frame_height = param['frame_width'], param['frame_height']
                click_x = (x / frame_width) * display_width
                click_y = (y / frame_height) * display_height
                offset_x = gaze_x - click_x
                offset_y = gaze_y - click_y
                calibration_values['gaze_offset_points'].append([offset_x, offset_y])
                click_count += 1
                print(f"Click {click_count} recorded at ({x}, {y}) with offset ({offset_x:.1f}, {offset_y:.1f})")
                MIN_CLICKS = 3
                if click_count >= MIN_CLICKS:
                    offsets = np.mean(calibration_values['gaze_offset_points'], axis=0)
                    calibration_values['gaze_offset'] = offsets.tolist()
                    print(f"Point Anywhere Calibration completed with average offset: {offsets}")
                    calibration_step_index += 1
            else:
                print("No gaze point available - click recorded but not processed")
                click_count += 1
        else:
            print("Click outside calibration mode or wrong step")

def get_display_size():
    try:
        screen = screeninfo.get_monitors()[0]
        return screen.width, screen.height
    except Exception:
        return 1920, 1080

def verify_distance(distance_cm, pixel_distance):
    """Verify distance calculation."""
    if np.isfinite(distance_cm) and 0 < distance_cm <= MAX_DISTANCE_CM and pixel_distance > 1e-6:
        return {"valid": True, "msg": "✓ Distance valid"}
    return {"valid": False, "msg": "✗ Distance invalid or out of range"}

def calculate_distance(landmarks, width, height, focal_length):
    left_eye = np.array([landmarks[33].x * width, landmarks[33].y * height])
    right_eye = np.array([landmarks[263].x * width, landmarks[263].y * height])
    pixel_distance = np.linalg.norm(left_eye - right_eye)
    distance_mm = (AVG_EYE_DISTANCE_MM * focal_length) / pixel_distance
    distance_cm = distance_mm / 10
    verification_states["distance"] = verify_distance(distance_cm, pixel_distance)
    return distance_cm

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

def verify_head_orientation(pitch, yaw, roll, rmat, tvec):
    """Verify head orientation computation."""
    if (np.all(np.isfinite([pitch, yaw, roll])) and
        np.linalg.norm(rmat - np.eye(3)) > 1e-6 and
        np.linalg.norm(tvec) > 1e-6):
        return {"valid": True, "msg": "✓ Pose valid"}
    return {"valid": False, "msg": "✗ Pose invalid or trivial"}

def get_head_orientation(landmarks, width, height, camera_matrix, dist_coeffs, angle_history, quat_history, initial_offset=None):
    model_points = np.array([
        (0.0, 0.0, 0.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
        (0.0, -330.0, -65.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
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
        sy = math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
        singular = sy < 1e-6
        if not singular:
            pitch = math.degrees(math.atan2(rmat[2,1], rmat[2,2]))
            yaw = math.degrees(math.atan2(-rmat[2,0], sy))
            roll = math.degrees(math.atan2(rmat[1,0], rmat[0,0]))
        else:
            pitch = math.degrees(math.atan2(-rmat[1,2], rmat[1,1]))
            yaw = math.degrees(math.atan2(-rmat[2,0], sy))
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
        verification_states["head_pose"] = verify_head_orientation(pitch, yaw, roll, smoothed_rmat, tvec)
        return pitch, yaw, roll, smoothed_rmat, tvec
    verification_states["head_pose"] = {"valid": False, "msg": "✗ Pose estimation failed"}
    return 0.0, 0.0, 0.0, np.eye(3), np.zeros((3,1))

def verify_gaze_vector(eye_yaw, eye_pitch, gaze_vector):
    """Verify gaze vector computation."""
    norm = np.linalg.norm(gaze_vector)
    if np.all(np.isfinite([eye_yaw, eye_pitch])) and 0.99 <= norm <= 1.01:
        return {"valid": True, "msg": "✓ Gaze vector valid"}
    return {"valid": False, "msg": "✗ Gaze vector invalid"}

def calculate_gaze_projection(landmarks, width, height, distance_cm, head_pitch, head_yaw, head_rmat, head_tvec, gaze_history, mirror_mode=True, pitch_offset=None, gaze_offset=None, display_w=1920, display_h=1080):
    focal_length = width
    camera_matrix = np.array([[focal_length, 0, width/2], [0, focal_length, height/2], [0, 0, 1]])
    left_eye_center = np.mean([(landmarks[33].x * width, landmarks[33].y * height),
                               (landmarks[133].x * width, landmarks[133].y * height)], axis=0)
    right_eye_center = np.mean([(landmarks[263].x * width, landmarks[263].y * height),
                                (landmarks[362].x * width, landmarks[362].y * height)], axis=0)
    eye_center_2d = (left_eye_center + right_eye_center) / 2
    left_iris_center = np.mean([(landmarks[i].x * width, landmarks[i].y * height) for i in [468, 469, 470, 471]], axis=0)
    right_iris_center = np.mean([(landmarks[i].x * width, landmarks[i].y * height) for i in [473, 474, 475, 476]], axis=0)
    iris_center_2d = (left_iris_center + right_iris_center) / 2
    eye_width = np.linalg.norm(right_eye_center - left_eye_center) / 2
    gaze_vec_2d = iris_center_2d - eye_center_2d
    eye_yaw = math.degrees(math.atan2(gaze_vec_2d[0], eye_width)) * 9
    eye_pitch = math.degrees(math.atan2(gaze_vec_2d[1], eye_width)) * 9
    if pitch_offset is not None:
        eye_pitch -= pitch_offset
    gaze_range_x = 15.0
    gaze_range_y = 5.0
    yaw_scale = (display_w / 2) / gaze_range_x
    pitch_scale = (display_h / 2) / gaze_range_y
    screen_x = (eye_yaw * yaw_scale) + display_w / 2
    screen_y = (eye_pitch * pitch_scale) + display_h / 2
    if gaze_offset is not None:
        screen_x -= gaze_offset[0]
        screen_y -= gaze_offset[1]
    screen_x = max(0, min(screen_x, display_w - 1))
    screen_y = max(0, min(screen_y, display_h - 1))
    gaze_history.append([screen_x, screen_y])
    if len(gaze_history) > 5:
        gaze_history.popleft()
    smoothed_gaze = np.mean(gaze_history, axis=0)
    final_x = int(smoothed_gaze[0])
    final_y = int(smoothed_gaze[1])
    eye_yaw_rad = math.radians(eye_yaw)
    eye_pitch_rad = math.radians(eye_pitch)
    gaze_vector = [
        math.sin(eye_yaw_rad),
        math.sin(eye_pitch_rad),
        -math.cos(eye_yaw_rad) * math.cos(eye_pitch_rad)
    ]
    verification_states["gaze_vector"] = verify_gaze_vector(eye_yaw, eye_pitch, gaze_vector)
    verification_states["gaze_projection"] = {
        "valid": 0 <= final_x < display_w and 0 <= final_y < display_h,
        "msg": "✓ Projection valid" if 0 <= final_x < display_w and 0 <= final_y < display_h else "✗ Projection out of bounds"
    }
    return (final_x, final_y), eye_yaw, eye_pitch

def create_bitcoin_laser_eyes(landmarks, frame, frame_count, width, height, is_face=True):
    if is_face:
        model_points = np.array([[-100.0, 50.0, -50.0], [100.0, 50.0, -50.0], [0.0, 0.0, 0.0],
                                 [0.0, -150.0, -50.0], [-75.0, -75.0, -50.0], [75.0, -75.0, -50.0]], dtype="double")
        image_points = np.array([(landmarks[33].x * width, landmarks[33].y * height),
                                 (landmarks[263].x * width, landmarks[263].y * height),
                                 (landmarks[1].x * width, landmarks[1].y * height),
                                 (landmarks[152].x * width, landmarks[152].y * height),
                                 (landmarks[61].x * width, landmarks[61].y * height),
                                 (landmarks[291].x * width, landmarks[291].y * height)], dtype="double")
        left_eye_indices = [33, 246, 161, 160, 159, 158]
        right_eye_indices = [263, 466, 388, 387, 386, 385]
        left_center = (int(np.mean([landmarks[i].x * width for i in left_eye_indices])),
                       int(np.mean([landmarks[i].y * height for i in left_eye_indices])))
        right_center = (int(np.mean([landmarks[i].x * width for i in right_eye_indices])),
                        int(np.mean([landmarks[i].y * height for i in right_eye_indices])))
    else:
        model_points = np.array([[0.0, 0.0, 0.0], [50.0, -50.0, -20.0], [50.0, -100.0, -30.0],
                                 [-50.0, -50.0, -20.0], [-50.0, -100.0, -30.0], [0.0, -70.0, -10.0]], dtype="double")
        image_points = np.array([(landmarks[0].x * width, landmarks[0].y * height),
                                 (landmarks[5].x * width, landmarks[5].y * height),
                                 (landmarks[8].x * width, landmarks[8].y * height),
                                 (landmarks[17].x * width, landmarks[17].y * height),
                                 (landmarks[20].x * width, landmarks[20].y * height),
                                 (landmarks[9].x * width, landmarks[9].y * height)], dtype="double")
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
            winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
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

def calibration_wizard(frame, phase, instruction, timer=None, total_time=None, click_count=None, min_clicks=None):
    overlay = frame.copy()
    alpha = 0.6
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (50, 50, 50), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, f"{phase}: {instruction}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if "Point Anywhere" in phase and click_count is not None and min_clicks is not None:
        cv2.putText(frame, f"Clicks: {click_count}/{min_clicks} (Click anywhere while gazing at cursor)",
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    elif timer is not None and total_time is not None:
        countdown_text = f"Hold steady for: {timer:.1f}s"
        cv2.putText(frame, countdown_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        progress = int((total_time - timer) / total_time * frame.shape[1])
        cv2.rectangle(frame, (0, 75), (progress, 80), (0, 255, 0), -1)
        if "Head" in phase:
            target = (frame.shape[1] // 2, frame.shape[0] // 2)
            cv2.circle(frame, target, 8, (0, 0, 255), -1)
    return frame

def apply_anime_filter(frame, landmarks, face_landmarks, hand_landmarks_list, frame_count, prev_frame_gray, curr_frame_gray, face_history, hand_histories, show_lasers=True, show_video=True, show_overlay=False, distance_cm=None, pitch=None, yaw=None, roll=None, rmat=None, gaze_point=None, eye_yaw=None, eye_pitch=None, display_w=1920, display_h=1080):
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
            dot_x = int((gaze_point[0] / display_w) * width)
            dot_y = int((gaze_point[1] / display_h) * height)
            cv2.circle(frame, (dot_x, dot_y), 10, (0, 0, 255), 2)
            cv2.putText(frame, f"Gaze", (dot_x + 15, dot_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if show_overlay and landmarks is not None:
            nose_tip = (int(landmarks[1].x * width), int(landmarks[1].y * height))
            focal_length = width
            camera_matrix = np.array([[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]], dtype="double")
            axis_length = 100
            axis_points = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(-1, 3)
            axis_points_2d, _ = cv2.projectPoints(axis_points, cv2.Rodrigues(rmat)[0], np.zeros((3, 1)), camera_matrix, np.zeros((4, 1)))
            axis_points_2d = axis_points_2d.reshape(-1, 2).astype(int)
            cv2.line(frame, nose_tip, tuple(axis_points_2d[0]), (0, 0, 255), 2)
            cv2.line(frame, nose_tip, tuple(axis_points_2d[1]), (0, 255, 0), 2)
            cv2.line(frame, nose_tip, tuple(axis_points_2d[2]), (255, 0, 0), 2)
            left_eye_center = (int(np.mean([landmarks[i].x * width for i in [33, 133]])),
                               int(np.mean([landmarks[i].y * height for i in [33, 133]])))
            right_eye_center = (int(np.mean([landmarks[i].x * width for i in [263, 362]])),
                                int(np.mean([landmarks[i].y * height for i in [263, 362]])))
            eye_length = 150
            left_eye_end = (
                int(left_eye_center[0] + eye_length * math.sin(math.radians(eye_yaw))),
                int(left_eye_center[1] - eye_length * math.sin(math.radians(eye_pitch)))
            )
            right_eye_end = (
                int(right_eye_center[0] + eye_length * math.sin(math.radians(eye_yaw))),
                int(right_eye_center[1] - eye_length * math.sin(math.radians(eye_pitch)))
            )
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
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
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

def run_gaze_tracking():
    global running, click_count, calibration_mode, calibration_step_index, calibration_values
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
            print("Failed to open camera after retries. Trying index 0...")
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("Failed to open any camera. Exiting...")
                return

    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    focal_length = width
    display_width, display_height = get_display_size()
    print(f"Display size: {display_width}x{display_height}")

    mirror_mode = True
    show_lasers = True
    show_video = True
    show_overlay = False
    print("Video is mirrored. Press 'm' (mirror), 'l' (lasers), 'v' (video), 'a' (overlay), 'c' (calibration), 'q' (quit).")

    frame_count = 0
    prev_frame_gray = None
    face_history = deque()
    hand_histories = []
    angle_history = deque(maxlen=10)
    quat_history = deque(maxlen=10)
    gaze_history = deque(maxlen=5)
    camera_matrix = np.array([[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    initial_offset = None
    pitch_offset = None
    gaze_offset = None
    calibration_mode = False
    calibration_step_index = 0
    calibration_timer_start = 0
    calibration_values = {"initial_offset": None, "gaze_offset": None, "pitch_offset": None, "gaze_offset_points": []}

    calibration_steps = [
        {
            "phase": "Head Calibration",
            "instruction": "Center your FACE and look straight into the camera.",
            "duration": 3.0,
            "action": lambda p, y, r, gp, ep: calibration_values.update({"initial_offset": [p, y, r]})
        },
        {
            "phase": "Point Anywhere Calibration",
            "instruction": "Click anywhere on the screen while gazing at the cursor.",
            "duration": None,
            "action": None
        },
        {
            "phase": "Eye Pitch Calibration",
            "instruction": "Keep your eyes LEVEL and look straight ahead.",
            "duration": 3.0,
            "action": lambda p, y, r, gp, ep: calibration_values.update({"pitch_offset": ep})
        }
    ]

    window_name = "Bitcoin Laser Eyes with Gaze Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    cv2.setMouseCallback(window_name, mouse_callback,
                        {'gaze_point': None, 'display_size': (display_width, display_height),
                         'frame_width': width, 'frame_height': height})

    vis = GazeVisualization()
    vis_thread = threading.Thread(target=vis.start, daemon=True)
    vis_thread.start()
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    last_click_time = 0
    click_indicator_pos = None

    while running:
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
        face_landmarks = None
        if face_results.multi_face_landmarks and len(face_results.multi_face_landmarks) > 0:
            face_landmarks = face_results.multi_face_landmarks[0]
        hand_results = hands.process(frame_rgb)
        hand_landmarks_list = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else []

        distance_cm = None
        pitch, yaw, roll, rmat, tvec = None, None, None, None, None
        screen_gaze_point = None
        eye_yaw, eye_pitch = None, None
        landmarks = None

        if face_landmarks:
            landmarks = face_landmarks.landmark
            distance_cm = calculate_distance(landmarks, width, height, focal_length)
            pitch, yaw, roll, rmat, tvec = get_head_orientation(
                landmarks, width, height, camera_matrix, dist_coeffs,
                angle_history, quat_history, initial_offset
            )
            screen_gaze_point, eye_yaw, eye_pitch = calculate_gaze_projection(
                landmarks, width, height, distance_cm, pitch, yaw, rmat, tvec,
                gaze_history, mirror_mode, pitch_offset, gaze_offset,
                display_w=display_width, display_h=display_height
            )
            head_position = [0, 0, distance_cm * 10]
            eye_yaw_rad = math.radians(eye_yaw)
            eye_pitch_rad = math.radians(eye_pitch)
            gaze_vector = [
                math.sin(eye_yaw_rad),
                math.sin(eye_pitch_rad),
                -math.cos(eye_yaw_rad) * math.cos(eye_pitch_rad)
            ]
            screen_info = {
                'width': SCREEN_WIDTH * 10,
                'height': SCREEN_HEIGHT * 10,
                'distance': distance_cm * 10
            }
            data_str = f"{head_position[0]},{head_position[1]},{head_position[2]}," \
                       f"{gaze_vector[0]},{gaze_vector[1]},{gaze_vector[2]}," \
                       f"{screen_info['width']},{screen_info['height']},{screen_info['distance']}"
            try:
                udp_socket.sendto(data_str.encode(), ('127.0.0.1', 5555))
                verification_states["udp_transmission"] = {"valid": True, "msg": "✓ UDP sent"}
            except Exception as e:
                verification_states["udp_transmission"] = {"valid": False, "msg": f"✗ UDP failed: {str(e)}"}

        cv2.setMouseCallback(window_name, mouse_callback,
                           {'gaze_point': screen_gaze_point,
                            'display_size': (display_width, display_height),
                            'frame_width': width, 'frame_height': height})

        if calibration_mode:
            current_step = calibration_steps[calibration_step_index]
            if calibration_step_index == 1:  # Point Anywhere Calibration
                MIN_CLICKS = 3
                frame = calibration_wizard(
                    frame, current_step["phase"], current_step["instruction"],
                    click_count=click_count, min_clicks=MIN_CLICKS
                )
            else:  # Timed calibration steps
                current_time = time.time()
                time_elapsed = current_time - calibration_timer_start
                time_left = current_step["duration"] - time_elapsed
                frame = calibration_wizard(
                    frame, current_step["phase"], current_step["instruction"],
                    time_left, current_step["duration"]
                )
                if time_left <= 0:
                    if face_landmarks and pitch is not None and yaw is not None and roll is not None and screen_gaze_point is not None and eye_pitch is not None:
                        current_step["action"](pitch, yaw, roll, screen_gaze_point, eye_pitch)
                        print(f"{current_step['phase']} completed successfully.")
                    else:
                        print(f"No valid face data for {current_step['phase']}.")
                    calibration_step_index += 1
                    if calibration_step_index >= len(calibration_steps):
                        initial_offset = calibration_values["initial_offset"]
                        gaze_offset = calibration_values["gaze_offset"]
                        pitch_offset = calibration_values["pitch_offset"]
                        calibration_mode = False
                        print("All calibration steps completed!")
                    else:
                        calibration_timer_start = time.time()

        frame = apply_anime_filter(
            frame, landmarks, face_landmarks, hand_landmarks_list, frame_count,
            prev_frame_gray, curr_frame_gray, face_history, hand_histories,
            show_lasers, show_video, show_overlay, distance_cm, pitch, yaw, roll, rmat,
            gaze_point=screen_gaze_point, eye_yaw=eye_yaw, eye_pitch=eye_pitch,
            display_w=display_width, display_h=display_height
        )

        if click_indicator_pos and (time.time() - last_click_time) < 0.5:
            cv2.circle(frame, click_indicator_pos, 10, (0, 255, 0), 2)

        # Display verification states
        y_pos = 450
        for key, state in verification_states.items():
            color = (0, 255, 0) if state["valid"] else (0, 0, 255)
            cv2.putText(frame, state["msg"], (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_pos += 20

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

        instructions = "Press 'm' (mirror) | 'l' (lasers) | 'v' (video) | 'a' (overlay) | 'c' (calibration) | 'q' (quit)"
        cv2.putText(frame, instructions, (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(window_name, frame)

        def update_click_indicator(x, y):
            nonlocal last_click_time, click_indicator_pos
            last_click_time = time.time()
            click_indicator_pos = (x, y)
            print(f"Click indicator updated at ({x}, {y})")

        def mouse_callback_with_indicator(event, x, y, flags, param):
            global click_count, calibration_values, calibration_mode, calibration_step_index
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"Click detected at ({x}, {y})")
                update_click_indicator(x, y)
                print(f"Calibration mode: {calibration_mode}, Step: {calibration_step_index}")
                if calibration_mode and calibration_step_index == 1:
                    print("Processing click in Point Anywhere Calibration")
                    gaze_point = param.get('gaze_point')
                    if gaze_point is not None:
                        gaze_x, gaze_y = gaze_point
                        display_width, display_height = param['display_size']
                        frame_width, frame_height = param['frame_width'], param['frame_height']
                        click_x = (x / frame_width) * display_width
                        click_y = (y / frame_height) * display_height
                        offset_x = gaze_x - click_x
                        offset_y = gaze_y - click_y
                        calibration_values['gaze_offset_points'].append([offset_x, offset_y])
                        click_count += 1
                        print(f"Click {click_count} recorded at ({x}, {y}) with offset ({offset_x:.1f}, {offset_y:.1f})")
                        MIN_CLICKS = 3
                        if click_count >= MIN_CLICKS:
                            offsets = np.mean(calibration_values['gaze_offset_points'], axis=0)
                            calibration_values['gaze_offset'] = offsets.tolist()
                            print(f"Point Anywhere Calibration completed with average offset: {offsets}")
                            calibration_step_index += 1
                    else:
                        print("No gaze point available - click recorded but not processed")
                        click_count += 1
                else:
                    print("Click outside calibration mode or wrong step")

        cv2.setMouseCallback(window_name, mouse_callback_with_indicator,
                           {'gaze_point': screen_gaze_point,
                            'display_size': (display_width, display_height),
                            'frame_width': width, 'frame_height': height})

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
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
            if not calibration_mode:
                calibration_mode = True
                calibration_step_index = 0
                calibration_timer_start = time.time()
                click_count = 0
                calibration_values = {
                    "initial_offset": None,
                    "gaze_offset": None,
                    "pitch_offset": None,
                    "gaze_offset_points": []
                }
                print("Starting calibration wizard. Follow the on-screen instructions.")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    hands.close()
    udp_socket.close()

if __name__ == "__main__":
    try:
        run_gaze_tracking()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
