# Outersense – Autonomous Gaze Tracking & 3D Visualization System

Outersense is an advanced system that captures and processes user attention with a high degree of mathematical precision. It fuses computer vision, real-time calibration routines, and sophisticated 3D geometric computations to accurately track head pose, eye orientation, and gaze direction. The system projects this data into an interactive 3D spatial model—laying the groundwork for future human interaction layers that monitor both digital and physical environments.

> **Note:** Outersense is a work-in-progress and is designed with future integration in mind (e.g., LiDAR or wearable eye trackers) to evolve into a comprehensive human interaction interface.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
  - [Gaze Tracking Module](#gaze-tracking-module)
  - [3D Visualization Module](#3d-visualization-module)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Calibration Process](#calibration-process)
- [Future Roadmap](#future-roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features

- **Advanced Iris & Eye Orientation Tracking:**
  Uses detailed iris filling techniques to calculate an accurate center from multiple fine-grained landmarks, which enhances the determination of eye orientation.

- **Gaze Ray Computation:**
  Integrates head pose, neck alignment, and refined eye data to compute a precise 3D gaze vector. Sophisticated ray calculations ensure an accurate projection of where the user is looking.

- **Distance & Orientation Estimation:**
  Utilizes the interocular distance as a metric to compute the distance from the screen, and leverages quaternion smoothing (via SLERP) for robust head, neck, and eye orientation estimation.

- **Real-Time Mathematical Projections:**
  Solves complex ray-plane intersections to project the computed gaze ray onto a 3D-modeled screen with preset tilt and dimensions.

- **Custom Calibration & Continuous Verification:**
  Multiple calibration phases (including head alignment, “point anywhere” calibration, and eye pitch adjustments) provide continuous validation, ensuring ongoing accuracy.

- **Interactive 3D Visualization:**
  Streams tracking data via UDP to a Matplotlib-based visualization module that renders a dynamic 3D view of head position, gaze direction, and intersection points on the virtual screen.

- **Future Integration:**
  Designed with scalability in mind for future integration with LiDAR sensors and wearable eye-tracking devices (e.g., a smart monocle or glasses) to ultimately build a complete human interaction layer.

---

## Architecture

Outersense is divided into two primary modules:

### Gaze Tracking Module

- **Video Capture & Preprocessing:**
  Utilizes OpenCV to capture frames from a webcam. Frames are optionally mirrored and preprocessed for analysis.

- **Landmark Detection:**
  Employs MediaPipe’s Face Mesh and Hands solutions to detect facial and hand landmarks in real time.

- **Distance Estimation:**
  Calculates the distance from the camera to the user using the measured interocular distance and a known average eye distance constant.

- **Head & Eye Orientation:**
  Implements custom routines (with support from quaternion-based SLERP) to estimate head pose, neck alignment, and precise eye orientations based on gathered landmarks.

- **Gaze Ray & Projection Computation:**
  Computes a 3D gaze vector by integrating head, neck, and eye data, then projects this vector onto a modeled screen by solving ray-plane intersection equations.

- **Data Transmission:**
  Packages computed values (head position, gaze vector, screen dimensions, etc.) into a comma-separated string and streams them via UDP to the visualization module.

### 3D Visualization Module

- **UDP Server:**
  Listens on a designated port (e.g., 5555 on `127.0.0.1`) to receive real-time tracking data.

- **Matplotlib 3D Rendering:**
  Uses Matplotlib’s `FuncAnimation` for rendering a dynamic 3D scene. The scene includes:
  - A 3D head model with accurately positioned eye markers.
  - A red gaze ray, dynamically computed from tracking data.
  - A virtual screen with a preset tilt, complete with computed intersection points where the gaze meets the screen.
  - On-screen text overlays displaying distance, calibration status, and verification messages.

- **Interactive Feedback:**
  Continuously updates the visualization based on new UDP data, ensuring that the 3D model reflects the current state of the user’s head and gaze positions.

---

## Installation & Requirements

**Requirements:**

- Python 3.8 or higher
- OpenCV
- MediaPipe
- NumPy
- SciPy
- Matplotlib
- screeninfo
- Other standard libraries (threading, socket, logging, etc.)

**Installation:**

1. Clone the repository:

   ```bash
   git clone https://github.com/cybrdelic/outsersense.git
   cd outsersense
   ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Gaze Tracking Module:**
   Run the gaze tracking script (e.g., `python run_gaze_tracking.py`). This starts the webcam capture, processes frames, and transmits tracking data over UDP.

2. **3D Visualization Module:**
   Run the visualization script (e.g., `python run_visualization.py`). A Matplotlib window will open displaying the 3D visualization and updating in real-time based on the tracking data received.

3. **Calibration & Interaction:**
   Follow on-screen calibration instructions and interact with the system using provided keyboard and mouse controls. See the documentation for detailed control mappings.

---

## Calibration Process

Outersense includes a multi-stage calibration wizard to ensure high tracking accuracy:

- **Head Calibration:**
  Align your head with the center of the camera to establish a baseline head pose.

- **Point Anywhere Calibration:**
  Click on the screen while gazing at a cursor to capture offsets, which fine-tune the gaze projection.

- **Eye Pitch Calibration:**
  Maintain eye level and a steady gaze to adjust for natural eye pitch variations.

The calibration process uses continuous visual feedback and verification states to dynamically refine tracking performance.

---

## Future Roadmap

- **Integration with LiDAR:**
  Improve spatial understanding by combining LiDAR data with gaze tracking.

- **Wearable Devices:**
  Explore integration with advanced eye-tracker hardware (monocles or glasses) to create a comprehensive interaction layer.

- **Enhanced UX Layer:**
  Develop a complete digital twin for human awareness that integrates both digital and physical interaction data—eventually powering a “Jarvis”-like assistant.

- **Expanded Calibration Features:**
  Further enhance the calibration process using additional sensor data and machine learning to adapt in real time.

---

## Contributing

Contributions are welcome! If you have ideas or improvements, please submit a pull request or open an issue. For major changes, please discuss your ideas in an issue first.

1. Fork the repository.
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

- **Email:** [cybrvybe@gmail.com](mailto:cybrvybe@gmail.com)
- **GitHub:** [cybrdelic](https://github.com/cybrdelic)

---

Outersense represents a step toward bridging digital and physical interactions using advanced computer vision and real-time data processing. Whether you’re exploring the future of autonomous systems, developing innovative UI solutions, or seeking a robust foundation for a digital twin of human awareness, Outersense offers a rich starting point. Thanks for checking it out, and let’s build something impactful together!

---
