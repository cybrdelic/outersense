import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import math
import threading
import time
import queue
import socket

# Data queue for communication with main tracking thread
data_queue = queue.Queue(maxsize=10)

# Default values
DEFAULT_HEAD_DISTANCE = 60  # cm
DEFAULT_SCREEN_SIZE = (34, 19)  # cm (typical 15.6" laptop)
DEFAULT_HEAD_POSITION = np.array([0, 0, DEFAULT_HEAD_DISTANCE])
DEFAULT_GAZE_VECTOR = np.array([0, 0, -1])  # Looking straight ahead

class GazeVisualization:
    def __init__(self):
        # Setup figure and 3D axis
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Initialize data
        self.head_position = DEFAULT_HEAD_POSITION
        self.gaze_vector = DEFAULT_GAZE_VECTOR
        self.screen_width, self.screen_height = DEFAULT_SCREEN_SIZE
        self.screen_distance = DEFAULT_HEAD_DISTANCE

        # Initialize visualization elements
        self._setup_visualization()

        # Setup UDP server for receiving data
        self.server_thread = threading.Thread(target=self._udp_server)
        self.server_thread.daemon = True

    def _setup_visualization(self):
        """Initialize all plot elements"""
        # Set labels and title
        self.ax.set_xlabel('X (cm)')
        self.ax.set_ylabel('Z (cm)')
        self.ax.set_zlabel('Y (cm)')
        self.ax.set_title('3D Gaze Tracking Visualization')

        # Set initial view limits
        self.ax.set_xlim([-40, 40])
        self.ax.set_ylim([0, 100])
        self.ax.set_zlim([-30, 30])

        # Create laptop model
        self.laptop_body = self._create_laptop_body()
        self.laptop_screen = self._create_laptop_screen()

        # Create head model
        self.head = self._create_head()

        # Create gaze line
        gaze_xs = [self.head_position[0], self.head_position[0] + self.gaze_vector[0] * 100]
        gaze_ys = [self.head_position[2], self.head_position[2] + self.gaze_vector[2] * 100]
        gaze_zs = [self.head_position[1], self.head_position[1] + self.gaze_vector[1] * 100]
        self.gaze_line, = self.ax.plot(gaze_xs, gaze_ys, gaze_zs, 'r-', linewidth=2, label='Gaze')

        # Create gaze intersection point
        self.gaze_point, = self.ax.plot([], [], [], 'ro', markersize=8)

        # Add text displays
        self.distance_text = self.ax.text2D(0.02, 0.95, f"Distance: {self.screen_distance:.1f} cm",
                                          transform=self.ax.transAxes)
        self.gaze_text = self.ax.text2D(0.02, 0.90, "Gaze: ", transform=self.ax.transAxes)

        # Set an optimal viewing angle
        self.ax.view_init(elev=20, azim=-60)

        # Add legend
        self.ax.legend()

    def _create_laptop_body(self):
        """Create the laptop base visualization"""
        width, height = 30, 20  # cm
        depth = 1.5  # cm

        # Position the laptop on a virtual desk
        x = 0  # Centered at origin
        y = 0  # At eye level
        z = 20  # Distance from origin

        # Define the vertices of the laptop base
        v = np.array([
            [x-width/2, y, z],          # Front left
            [x+width/2, y, z],          # Front right
            [x+width/2, y, z+depth],    # Back right
            [x-width/2, y, z+depth],    # Back left
            [x-width/2, y+height, z],   # Front left top (for reference, not visible)
        ])

        # Draw the laptop base as a rectangle
        laptop_body = self.ax.plot_surface(
            np.array([[v[0,0], v[1,0]], [v[3,0], v[2,0]]]),
            np.array([[v[0,1], v[1,1]], [v[3,1], v[2,1]]]),
            np.array([[v[0,2], v[1,2]], [v[3,2], v[2,2]]]),
            color='gray', alpha=0.7
        )

        return laptop_body

    def _create_laptop_screen(self):
        """Create the laptop screen visualization"""
        width, height = self.screen_width, self.screen_height  # cm

        # Screen position (slightly above laptop base)
        x = 0  # Centered
        base_height = 1  # cm
        y = base_height + height/2  # Centered vertically
        z = 20  # Same as laptop base

        # Screen tilt angle (degrees)
        tilt_angle = 75  # Typical laptop screen angle

        # Calculate screen corners with tilt
        rad_angle = math.radians(tilt_angle)
        z_offset = height/2 * math.cos(rad_angle)
        y_offset = height/2 * math.sin(rad_angle)

        # Define screen vertices
        corners = np.array([
            [x-width/2, y-y_offset, z-z_offset],  # Bottom left
            [x+width/2, y-y_offset, z-z_offset],  # Bottom right
            [x+width/2, y+y_offset, z+z_offset],  # Top right
            [x-width/2, y+y_offset, z+z_offset],  # Top left
            [x-width/2, y-y_offset, z-z_offset]   # Back to bottom left to close the shape
        ])

        # Draw screen outline
        screen = self.ax.plot(corners[:,0], corners[:,2], corners[:,1], 'b-', linewidth=2)

        # Draw screen surface
        x_grid = np.array([[corners[0,0], corners[1,0]], [corners[3,0], corners[2,0]]])
        y_grid = np.array([[corners[0,1], corners[1,1]], [corners[3,1], corners[2,1]]])
        z_grid = np.array([[corners[0,2], corners[1,2]], [corners[3,2], corners[2,2]]])

        screen_surface = self.ax.plot_surface(x_grid, z_grid, y_grid, color='blue', alpha=0.3)

        # Store screen parameters for intersection calculations
        self.screen_corners = corners
        self.screen_normal = np.array([0, math.sin(rad_angle), math.cos(rad_angle)])
        self.screen_center = np.array([x, y, z])

        return screen_surface

    def _create_head(self):
        """Create a simple head visualization"""
        # Head dimensions
        head_radius = 10  # cm

        # Create a sphere for the head
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = self.head_position[0] + head_radius * np.outer(np.cos(u), np.sin(v))
        y = self.head_position[1] + head_radius * np.outer(np.sin(u), np.sin(v))
        z = self.head_position[2] + head_radius * np.outer(np.ones(np.size(u)), np.cos(v))

        head = self.ax.plot_surface(x, z, y, color='tan', alpha=0.7)

        # Add eyes (simplified as points)
        eye_offset = 3  # cm
        left_eye = self.head_position + np.array([-eye_offset, 0, 0])
        right_eye = self.head_position + np.array([eye_offset, 0, 0])

        self.left_eye, = self.ax.plot([left_eye[0]], [left_eye[2]], [left_eye[1]], 'ko', markersize=5)
        self.right_eye, = self.ax.plot([right_eye[0]], [right_eye[2]], [right_eye[1]], 'ko', markersize=5)

        return head

    def _calculate_gaze_screen_intersection(self):
        """Calculate where the gaze ray intersects with the laptop screen"""
        # Extract the first 4 corners (the actual screen quad)
        corners = self.screen_corners[:4]

        # Get two vectors in the screen plane
        v1 = corners[1] - corners[0]  # Bottom edge
        v2 = corners[3] - corners[0]  # Left edge

        # Calculate screen normal
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)

        # Equation of the plane: normal · (point - corners[0]) = 0
        # Ray equation: head_position + t * gaze_vector
        # Solving for t: normal · (head_position + t * gaze_vector - corners[0]) = 0

        ray_origin = self.head_position
        ray_direction = self.gaze_vector

        ndotu = np.dot(normal, ray_direction)
        if abs(ndotu) < 1e-6:
            # Ray is parallel to the screen
            return None

        w = ray_origin - corners[0]
        t = -np.dot(normal, w) / ndotu

        if t < 0:
            # Intersection is behind the head
            return None

        # Calculate intersection point
        intersection = ray_origin + t * ray_direction

        # Check if the intersection is within the screen bounds
        # Project onto the screen plane
        origin = corners[0]
        basis1 = v1 / np.linalg.norm(v1)
        basis2 = v2 / np.linalg.norm(v2)

        # Get coordinates in the screen plane
        screen_vector = intersection - origin
        u = np.dot(screen_vector, basis1)
        v = np.dot(screen_vector, basis2)

        # Check bounds
        if (0 <= u <= np.linalg.norm(v1)) and (0 <= v <= np.linalg.norm(v2)):
            # Intersection is within screen bounds
            return intersection

        return None

    def update(self, head_position=None, gaze_vector=None, screen_info=None):
        """Update the visualization with new data"""
        if head_position is not None:
            # Convert from mm to cm
            self.head_position = np.array(head_position) / 10.0

        if gaze_vector is not None:
            self.gaze_vector = np.array(gaze_vector)
            # Normalize gaze vector
            if np.linalg.norm(self.gaze_vector) > 0:
                self.gaze_vector = self.gaze_vector / np.linalg.norm(self.gaze_vector)

        if screen_info is not None:
            if 'width' in screen_info and 'height' in screen_info:
                self.screen_width = screen_info['width'] / 10.0  # mm to cm
                self.screen_height = screen_info['height'] / 10.0  # mm to cm

            if 'distance' in screen_info:
                self.screen_distance = screen_info['distance'] / 10.0  # mm to cm

        # Update the plot elements
        self._update_head()
        self._update_gaze()
        self._update_screen()
        self._update_text()

    def _update_head(self):
        """Update head position and orientation"""
        # For a more comprehensive implementation, this would update a 3D head model
        # Here we just update the eye positions
        eye_offset = 3  # cm
        left_eye_pos = self.head_position + np.array([-eye_offset, 0, 0])
        right_eye_pos = self.head_position + np.array([eye_offset, 0, 0])

        self.left_eye.set_data([left_eye_pos[0]], [left_eye_pos[2]])
        self.left_eye.set_3d_properties([left_eye_pos[1]])
        self.right_eye.set_data([right_eye_pos[0]], [right_eye_pos[2]])
        self.right_eye.set_3d_properties([right_eye_pos[1]])

    def _update_gaze(self):
        """Update gaze line and intersection point"""
        # Calculate gaze line
        gaze_length = 100  # cm (long enough to ensure intersection)
        gaze_end = self.head_position + self.gaze_vector * gaze_length

        self.gaze_line.set_data([self.head_position[0], gaze_end[0]],
                               [self.head_position[2], gaze_end[2]])
        self.gaze_line.set_3d_properties([self.head_position[1], gaze_end[1]])

        # Calculate and update intersection point
        intersection = self._calculate_gaze_screen_intersection()
        if intersection is not None:
            self.gaze_point.set_data([intersection[0]], [intersection[2]])
            self.gaze_point.set_3d_properties([intersection[1]])

            # Calculate normalized screen coordinates (0-1)
            corners = self.screen_corners[:4]
            origin = corners[0]
            width_vec = corners[1] - corners[0]
            height_vec = corners[3] - corners[0]

            screen_vector = intersection - origin
            u = np.dot(screen_vector, width_vec) / np.dot(width_vec, width_vec)
            v = np.dot(screen_vector, height_vec) / np.dot(height_vec, height_vec)

            # Update text
            self.gaze_text.set_text(f"Gaze: ({u*100:.1f}%, {v*100:.1f}%)")
        else:
            # Hide point if no intersection
            self.gaze_point.set_data([], [])
            self.gaze_point.set_3d_properties([])
            self.gaze_text.set_text("Gaze: Not on screen")

    def _update_screen(self):
        """Update screen model if dimensions change"""
        # Remove old screen
        if hasattr(self, 'laptop_screen') and self.laptop_screen is not None:
            try:
                self.laptop_screen.remove()
            except:
                pass

        # Create new screen with updated dimensions
        self.laptop_screen = self._create_laptop_screen()

    def _update_text(self):
        """Update text displays"""
        self.distance_text.set_text(f"Distance: {np.linalg.norm(self.head_position):.1f} cm")

    def _udp_server(self):
        """UDP server to receive data from the gaze tracking process"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_socket.bind(('127.0.0.1', 5555))
        server_socket.settimeout(0.1)

        print("UDP server listening on 127.0.0.1:5555")

        while True:
            try:
                data, addr = server_socket.recvfrom(1024)
                try:
                    # Parse received data
                    # Format: "head_x,head_y,head_z,gaze_x,gaze_y,gaze_z,screen_width,screen_height,screen_distance"
                    values = [float(x) for x in data.decode().split(',')]

                    head_position = values[0:3]
                    gaze_vector = values[3:6]
                    screen_info = {
                        'width': values[6],
                        'height': values[7],
                        'distance': values[8]
                    }

                    # Update visualization
                    self.update(head_position, gaze_vector, screen_info)
                except Exception as e:
                    print(f"Error processing data: {e}")
            except socket.timeout:
                continue
            except Exception as e:
                print(f"UDP server error: {e}")
                time.sleep(1)

    def animate(self, i):
        """Animation function for matplotlib"""
        try:
            if not data_queue.empty():
                data = data_queue.get(block=False)
                head_pos = data.get('head_position')
                gaze_vec = data.get('gaze_vector')
                screen_info = {
                    'width': data.get('screen_width', DEFAULT_SCREEN_SIZE[0] * 10),  # cm to mm
                    'height': data.get('screen_height', DEFAULT_SCREEN_SIZE[1] * 10),
                    'distance': data.get('screen_distance', DEFAULT_HEAD_DISTANCE * 10)
                }

                self.update(head_pos, gaze_vec, screen_info)
        except queue.Empty:
            pass

        # Return all artists that need to be redrawn
        return (self.gaze_line, self.gaze_point, self.left_eye, self.right_eye,
                self.distance_text, self.gaze_text)

    def start(self):
        """Start the visualization"""
        # Start UDP server thread
        self.server_thread.start()

        # Start animation
        anim = FuncAnimation(self.fig, self.animate, interval=100, blit=True)
        plt.show()

# Main function
def run_visualization():
    """Run the 3D visualization sandbox"""
    # Create and start visualization
    vis = GazeVisualization()
    vis.start()

if __name__ == "__main__":
    run_visualization()
