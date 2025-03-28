import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import threading
import time
import queue
import socket
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_HEAD_DISTANCE = 60  # cm
DEFAULT_SCREEN_SIZE = (34, 19)  # cm (width, height)
DEFAULT_HEAD_POSITION = np.array([0, 0, DEFAULT_HEAD_DISTANCE])
DEFAULT_GAZE_VECTOR = np.array([0, 0, -1])  # Unit vector forward
UDP_PORT = 5555
UDP_HOST = '127.0.0.1'
UPDATE_INTERVAL = 50  # ms (20 FPS)
QUEUE_SIZE = 10
HEAD_RADIUS = 10  # cm
EYE_OFFSET = 3  # cm
GAZE_LINE_LENGTH = 100  # cm
SCREEN_TILT_ANGLE = 75  # degrees
MAX_DISTANCE_CM = 200  # Maximum allowable distance

# Data queue for thread-safe communication
data_queue = queue.Queue(maxsize=QUEUE_SIZE)

# Verification states dictionary
verification_states = {
    "head_position": {"valid": False, "msg": "✗ Head position not set"},
    "gaze_vector": {"valid": False, "msg": "✗ Gaze vector not set"},
    "screen_model": {"valid": False, "msg": "✗ Screen model not valid"},
    "gaze_intersection": {"valid": False, "msg": "✗ Intersection not computed"},
    "udp_reception": {"valid": False, "msg": "✗ UDP data not received"}
}

class GazeVisualization:
    def __init__(self, screen_size=DEFAULT_SCREEN_SIZE, head_distance=DEFAULT_HEAD_DISTANCE):
        """
        Initialize the 3D gaze tracking visualization with verification.

        Args:
            screen_size (tuple): Screen width and height in cm.
            head_distance (float): Initial head distance from screen in cm.
        """
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.head_position = np.array([0, 0, head_distance], dtype=float)
        self.gaze_vector = np.array([0, 0, -1], dtype=float)
        self.screen_width, self.screen_height = screen_size
        self.screen_distance = head_distance
        self.laptop_body = None
        self.laptop_screen = None
        self.head = None
        self.left_eye = None
        self.right_eye = None
        self.gaze_line = None
        self.gaze_point = None
        self.distance_text = None
        self.gaze_text = None
        self.verification_text = None
        self._setup_visualization()
        self.running = True
        self.server_thread = threading.Thread(target=self._udp_server, daemon=True)

    def _setup_visualization(self):
        """Initialize plot elements with verification display."""
        self.ax.set_xlabel('X (cm)', fontsize=12, labelpad=10)
        self.ax.set_ylabel('Z (cm)', fontsize=12, labelpad=10)
        self.ax.set_zlabel('Y (cm)', fontsize=12, labelpad=10)
        self.ax.set_title('3D Gaze Tracking Visualization', fontsize=14, pad=20)
        self.ax.set_xlim([-40, 40])
        self.ax.set_ylim([0, 100])
        self.ax.set_zlim([-30, 30])
        self.laptop_body = self._create_laptop_body()
        self.laptop_screen = self._create_laptop_screen()
        self.head = self._create_head()
        gaze_xs = [self.head_position[0], self.head_position[0] + self.gaze_vector[0] * GAZE_LINE_LENGTH]
        gaze_ys = [self.head_position[2], self.head_position[2] + self.gaze_vector[2] * GAZE_LINE_LENGTH]
        gaze_zs = [self.head_position[1], self.head_position[1] + self.gaze_vector[1] * GAZE_LINE_LENGTH]
        self.gaze_line, = self.ax.plot(gaze_xs, gaze_ys, gaze_zs, 'r-', linewidth=2, label='Gaze Direction')
        self.gaze_point, = self.ax.plot([], [], [], 'ro', markersize=8, label='Gaze Point')
        self.distance_text = self.ax.text2D(0.02, 0.95, f"Distance: {self.screen_distance:.1f} cm",
                                            transform=self.ax.transAxes, fontsize=10, color='black')
        self.gaze_text = self.ax.text2D(0.02, 0.90, "Gaze: Not on screen",
                                        transform=self.ax.transAxes, fontsize=10, color='black')
        self.verification_text = self.ax.text2D(0.70, 0.95, "Verification:\n" + "\n".join(
            [f"{k}: {v['msg']}" for k, v in verification_states.items()]),
                                        transform=self.ax.transAxes, fontsize=8, color='black', verticalalignment='top')
        self.ax.view_init(elev=20, azim=-60)
        self.ax.legend(loc='upper right')

    def _create_laptop_body(self):
        """Create laptop body with verification."""
        width, height = 30, 20  # cm
        depth = 1.5  # cm
        x, y, z = 0, 0, 0
        v = np.array([
            [x-width/2, y, z], [x+width/2, y, z],
            [x+width/2, y, z+depth], [x-width/2, y, z+depth]
        ])
        surface = self.ax.plot_surface(
            np.array([[v[0,0], v[1,0]], [v[3,0], v[2,0]]]),
            np.array([[v[0,1], v[1,1]], [v[3,1], v[2,1]]]),
            np.array([[v[0,2], v[1,2]], [v[3,2], v[2,2]]]),
            color='silver', alpha=0.8, edgecolor='k'
        )
        verification_states["screen_model"] = {"valid": True, "msg": "✓ Screen model valid"}
        return surface

    def _create_laptop_screen(self):
        """Create laptop screen with verification."""
        width, height = self.screen_width, self.screen_height
        x, y_base, z = 0, 1, 0
        y = y_base + height/2
        rad_angle = math.radians(SCREEN_TILT_ANGLE)
        z_offset = height/2 * math.cos(rad_angle)
        y_offset = height/2 * math.sin(rad_angle)
        corners = np.array([
            [x-width/2, y-y_offset, z-z_offset],
            [x+width/2, y-y_offset, z-z_offset],
            [x+width/2, y+y_offset, z+z_offset],
            [x-width/2, y+y_offset, z+z_offset],
            [x-width/2, y-y_offset, z-z_offset]
        ])
        self.ax.plot(corners[:,0], corners[:,2], corners[:,1], 'b-', linewidth=3)
        x_grid = np.array([[corners[0,0], corners[1,0]], [corners[3,0], corners[2,0]]])
        y_grid = np.array([[corners[0,1], corners[1,1]], [corners[3,1], corners[2,1]]])
        z_grid = np.array([[corners[0,2], corners[1,2]], [corners[3,2], corners[2,2]]])
        screen_surface = self.ax.plot_surface(x_grid, z_grid, y_grid, color='lightblue', alpha=0.5, edgecolor='k')
        self.screen_corners = corners
        self.screen_normal = np.array([0, math.sin(rad_angle), math.cos(rad_angle)])
        self.screen_center = np.array([x, y, z])
        # Verify screen normal is unit length
        norm = np.linalg.norm(self.screen_normal)
        if 0.99 <= norm <= 1.01:
            verification_states["screen_model"] = {"valid": True, "msg": "✓ Screen model valid"}
        else:
            verification_states["screen_model"] = {"valid": False, "msg": "✗ Screen normal invalid"}
        return screen_surface

    def _create_head(self):
        """Create head visualization."""
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = self.head_position[0] + HEAD_RADIUS * np.outer(np.cos(u), np.sin(v))
        y = self.head_position[1] + HEAD_RADIUS * np.outer(np.sin(u), np.sin(v))
        z = self.head_position[2] + HEAD_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))
        head = self.ax.plot_surface(x, z, y, color='tan', alpha=0.7, edgecolor='k')
        left_eye = self.head_position + np.array([-EYE_OFFSET, 0, 0])
        right_eye = self.head_position + np.array([EYE_OFFSET, 0, 0])
        self.left_eye, = self.ax.plot([left_eye[0]], [left_eye[2]], [left_eye[1]], 'ko', markersize=5, label='Eyes')
        self.right_eye, = self.ax.plot([right_eye[0]], [right_eye[2]], [right_eye[1]], 'ko', markersize=5)
        return head

    def _verify_head_position(self, head_position):
        """Verify head position constraints."""
        distance = np.linalg.norm(head_position)
        if np.all(np.isfinite(head_position)) and 0 < distance <= MAX_DISTANCE_CM:
            return {"valid": True, "msg": "✓ Head position valid"}
        return {"valid": False, "msg": "✗ Head position invalid or out of range"}

    def _verify_gaze_vector(self, gaze_vector):
        """Verify gaze vector is unit length and finite."""
        norm = np.linalg.norm(gaze_vector)
        if np.all(np.isfinite(gaze_vector)) and 0.99 <= norm <= 1.01:
            return {"valid": True, "msg": "✓ Gaze vector valid"}
        return {"valid": False, "msg": "✗ Gaze vector non-unit or non-finite"}

    def _calculate_gaze_screen_intersection(self):
        """Calculate gaze-screen intersection with verification."""
        corners = self.screen_corners[:4]
        v1 = corners[1] - corners[0]
        v2 = corners[3] - corners[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            verification_states["gaze_intersection"] = {"valid": False, "msg": "✗ Invalid screen normal"}
            return None
        normal = normal / norm
        ray_origin = self.head_position
        ray_direction = self.gaze_vector
        ndotu = np.dot(normal, ray_direction)
        if abs(ndotu) < 1e-6:
            verification_states["gaze_intersection"] = {"valid": False, "msg": "✗ Gaze parallel to screen"}
            return None
        w = ray_origin - corners[0]
        t = -np.dot(normal, w) / ndotu
        if t < 0:
            verification_states["gaze_intersection"] = {"valid": False, "msg": "✗ Intersection behind head"}
            return None
        intersection = ray_origin + t * ray_direction
        origin = corners[0]
        basis1 = v1 / np.linalg.norm(v1)
        basis2 = v2 / np.linalg.norm(v2)
        screen_vector = intersection - origin
        u = np.dot(screen_vector, basis1)
        v = np.dot(screen_vector, basis2)
        if (0 <= u <= np.linalg.norm(v1)) and (0 <= v <= np.linalg.norm(v2)):
            verification_states["gaze_intersection"] = {"valid": True, "msg": "✓ Intersection valid"}
            return intersection
        verification_states["gaze_intersection"] = {"valid": False, "msg": "✗ Intersection out of bounds"}
        return None

    def update(self, head_position=None, gaze_vector=None, screen_info=None):
        """Update visualization with verification."""
        try:
            if head_position is not None:
                self.head_position = np.array(head_position, dtype=float) / 10.0  # mm to cm
                verification_states["head_position"] = self._verify_head_position(self.head_position)
            if gaze_vector is not None:
                gaze_vector = np.array(gaze_vector, dtype=float)
                norm = np.linalg.norm(gaze_vector)
                self.gaze_vector = gaze_vector / norm if norm > 0 else DEFAULT_GAZE_VECTOR
                verification_states["gaze_vector"] = self._verify_gaze_vector(self.gaze_vector)
            if screen_info is not None:
                self.screen_width = screen_info.get('width', self.screen_width * 10) / 10.0
                self.screen_height = screen_info.get('height', self.screen_height * 10) / 10.0
                self.screen_distance = screen_info.get('distance', self.screen_distance * 10) / 10.0
            self._update_head()
            self._update_gaze()
            self._update_screen()
            self._update_text()
        except Exception as e:
            logger.error(f"Error updating visualization: {e}")

    def _update_head(self):
        """Update head position."""
        eye_offset = EYE_OFFSET
        left_eye_pos = self.head_position + np.array([-eye_offset, 0, 0])
        right_eye_pos = self.head_position + np.array([eye_offset, 0, 0])
        self.left_eye.set_data_3d([left_eye_pos[0]], [left_eye_pos[2]], [left_eye_pos[1]])
        self.right_eye.set_data_3d([right_eye_pos[0]], [right_eye_pos[2]], [right_eye_pos[1]])

    def _update_gaze(self):
        """Update gaze line and intersection."""
        gaze_end = self.head_position + self.gaze_vector * GAZE_LINE_LENGTH
        self.gaze_line.set_data_3d(
            [self.head_position[0], gaze_end[0]],
            [self.head_position[2], gaze_end[2]],
            [self.head_position[1], gaze_end[1]]
        )
        intersection = self._calculate_gaze_screen_intersection()
        if intersection is not None:
            self.gaze_point.set_data_3d([intersection[0]], [intersection[2]], [intersection[1]])
            corners = self.screen_corners[:4]
            origin = corners[0]
            width_vec = corners[1] - corners[0]
            height_vec = corners[3] - corners[0]
            screen_vector = intersection - origin
            u = np.dot(screen_vector, width_vec) / np.dot(width_vec, width_vec)
            v = np.dot(screen_vector, height_vec) / np.dot(height_vec, height_vec)
            self.gaze_text.set_text(f"Gaze: ({u*100:.1f}%, {v*100:.1f}%)")
        else:
            self.gaze_point.set_data_3d([], [], [])
            self.gaze_text.set_text("Gaze: Not on screen")

    def _update_screen(self):
        """Update screen model."""
        if hasattr(self, 'laptop_screen') and self.laptop_screen is not None:
            try:
                self.laptop_screen.remove()
            except:
                pass
        self.laptop_screen = self._create_laptop_screen()

    def _update_text(self):
        """Update text displays including verification states."""
        self.distance_text.set_text(f"Distance: {np.linalg.norm(self.head_position):.1f} cm")
        self.verification_text.set_text("Verification:\n" + "\n".join(
            [f"{k}: {v['msg']}" for k, v in verification_states.items()]))

    def _udp_server(self):
        """UDP server with verification."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            server_socket.bind((UDP_HOST, UDP_PORT))
            server_socket.settimeout(0.1)
            logger.info(f"UDP server listening on {UDP_HOST}:{UDP_PORT}")
            while self.running:
                try:
                    data, addr = server_socket.recvfrom(1024)
                    values = [float(x) for x in data.decode().split(',')]
                    head_position = values[0:3]
                    gaze_vector = values[3:6]
                    screen_info = {'width': values[6], 'height': values[7], 'distance': values[8]}
                    try:
                        data_queue.put_nowait({
                            'head_position': head_position,
                            'gaze_vector': gaze_vector,
                            'screen_width': screen_info['width'],
                            'screen_height': screen_info['height'],
                            'screen_distance': screen_info['distance']
                        })
                        verification_states["udp_reception"] = {"valid": True, "msg": "✓ UDP data received"}
                    except queue.Full:
                        logger.warning("Data queue full, dropping packet.")
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"UDP server error: {e}")
                    verification_states["udp_reception"] = {"valid": False, "msg": "✗ UDP reception failed"}
                    time.sleep(1)
        except Exception as e:
            logger.error(f"Failed to start UDP server: {e}")
        finally:
            server_socket.close()

    def animate(self, i):
        """Animation function with verification updates."""
        try:
            if not data_queue.empty():
                data = data_queue.get_nowait()
                self.update(
                    head_position=data.get('head_position'),
                    gaze_vector=data.get('gaze_vector'),
                    screen_info={
                        'width': data.get('screen_width', DEFAULT_SCREEN_SIZE[0] * 10),
                        'height': data.get('screen_height', DEFAULT_SCREEN_SIZE[1] * 10),
                        'distance': data.get('screen_distance', DEFAULT_HEAD_DISTANCE * 10)
                    }
                )
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Animation error: {e}")
        return (self.gaze_line, self.gaze_point, self.left_eye, self.right_eye,
                self.distance_text, self.gaze_text, self.verification_text)

    def start(self):
        """Start the visualization."""
        self.server_thread.start()
        anim = FuncAnimation(self.fig, self.animate, interval=UPDATE_INTERVAL, blit=True)
        plt.show()

    def stop(self):
        """Stop the visualization."""
        self.running = False
        self.server_thread.join()
        plt.close(self.fig)

def run_visualization():
    """Run the visualization with error handling."""
    vis = None
    try:
        vis = GazeVisualization()
        vis.start()
    except KeyboardInterrupt:
        logger.info("Visualization stopped by user.")
        if vis: vis.stop()
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        if vis: vis.stop()

if __name__ == "__main__":
    run_visualization()
