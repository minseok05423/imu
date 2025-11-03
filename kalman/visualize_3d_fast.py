"""
Real-time 3D visualization of IMU orientation using Complementary Filter.
FAST response with no lag - perfect for visualization!
"""

import serial
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R


# ============================================================================
# CALIBRATION PARAMETERS
# ============================================================================

GYRO_BIAS = np.array([-0.093434, 0.038554, -0.004698])
ACCEL_SCALE = 1.0
MAG_OFFSET = np.array([0.0857, -0.7558, -0.6930])
MAG_A = np.array([
    [0.995363, 0.000293, 0.026286],
    [0.000293, 0.950979, -0.001707],
    [0.026286, -0.001707, 1.057143]
])

# ============================================================================


class ComplementaryFilter:
    """
    Fast complementary filter for IMU orientation.
    Much faster response than UKF, perfect for visualization.
    """

    def __init__(self, dt=0.01, alpha=0.98):
        """
        Initialize complementary filter.

        Args:
            dt: Time step (seconds)
            alpha: Filter weight (0.95-0.99). Higher = trust gyro more (faster response)
        """
        self.dt = dt
        self.alpha = alpha

        # Current orientation as quaternion [w, x, y, z]
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, accel, gyro, mag):
        """
        Update orientation with new sensor data.

        Args:
            accel: Accelerometer [ax, ay, az] in m/s²
            gyro: Gyroscope [wx, wy, wz] in rad/s
            mag: Magnetometer [mx, my, mz] (calibrated)
        """
        # ========================================
        # STEP 1: Integrate gyroscope (high-frequency, fast response)
        # ========================================

        # Convert angular velocity to quaternion derivative
        w_norm = np.linalg.norm(gyro)
        if w_norm > 0:
            angle = w_norm * self.dt
            axis = gyro / w_norm
            dq = np.array([
                np.cos(angle/2),
                axis[0] * np.sin(angle/2),
                axis[1] * np.sin(angle/2),
                axis[2] * np.sin(angle/2)
            ])
            q_gyro = self.quaternion_multiply(self.q, dq)
        else:
            q_gyro = self.q.copy()

        # Normalize
        q_gyro = q_gyro / np.linalg.norm(q_gyro)

        # ========================================
        # STEP 2: Calculate orientation from accel + mag (low-frequency, drift-free)
        # ========================================

        # Normalize accelerometer
        accel_norm = accel / np.linalg.norm(accel)

        # Normalize magnetometer
        mag_norm = mag / np.linalg.norm(mag)

        # Remove component of mag in direction of gravity
        mag_horizontal = mag_norm - np.dot(mag_norm, accel_norm) * accel_norm
        mag_horizontal = mag_horizontal / np.linalg.norm(mag_horizontal)

        # Build rotation matrix from accel and mag
        # accel points down (Z axis in body frame)
        # mag points north (X axis in horizontal plane)

        # East = North × Down
        east = np.cross(mag_horizontal, accel_norm)
        east = east / np.linalg.norm(east)

        # North = Down × East (reorthogonalize)
        north = np.cross(accel_norm, east)

        # Build rotation matrix [North, East, Down]
        R_am = np.column_stack([north, east, accel_norm])

        # Convert to quaternion
        q_accel_mag = self.rotation_matrix_to_quaternion(R_am)

        # ========================================
        # STEP 3: Complementary filter (blend gyro and accel/mag)
        # ========================================

        # Use SLERP (Spherical Linear Interpolation) for smooth blending
        q_fused = self.slerp(q_accel_mag, q_gyro, self.alpha)

        # Update state
        self.q = q_fused / np.linalg.norm(q_fused)

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion."""
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([w, x, y, z])

    def slerp(self, q1, q2, t):
        """Spherical linear interpolation between quaternions."""
        dot = np.dot(q1, q2)

        # If negative dot, negate one quaternion to take shorter path
        if dot < 0.0:
            q2 = -q2
            dot = -dot

        # Clamp dot product
        dot = np.clip(dot, -1.0, 1.0)

        theta = np.arccos(dot)

        if abs(theta) < 0.001:
            # Quaternions very close, use linear interpolation
            return q1 * (1 - t) + q2 * t

        return (q1 * np.sin((1-t) * theta) + q2 * np.sin(t * theta)) / np.sin(theta)

    def get_quaternion(self):
        """Get current orientation as quaternion [w, x, y, z]."""
        return self.q

    def get_euler(self):
        """Get current orientation as Euler angles (roll, pitch, yaw) in degrees."""
        # Convert to scipy format [x, y, z, w]
        q_scipy = [self.q[1], self.q[2], self.q[3], self.q[0]]
        rot = R.from_quat(q_scipy)
        return rot.as_euler('xyz', degrees=True)


def find_serial_ports():
    """Find available serial ports."""
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    available = []

    print("Available serial ports:")
    for port in ports:
        print(f"  {port.device} - {port.description}")
        available.append(port.device)

    return available


def parse_imu_data(line):
    """Parse IMU data from serial line."""
    try:
        line = line.strip()

        if ',' in line and '{' not in line:
            values = [float(x) for x in line.split(',')]
            if len(values) >= 9:
                return np.array(values[0:3]), np.array(values[3:6]), np.array(values[6:9])

        elif line.startswith('{'):
            data = json.loads(line)
            return (np.array([data['ax'], data['ay'], data['az']]),
                   np.array([data['gx'], data['gy'], data['gz']]),
                   np.array([data['mx'], data['my'], data['mz']]))

    except:
        return None

    return None


def get_cube_vertices():
    """Get vertices of a cube."""
    vertices = np.array([
        [-0.5, -0.3, -0.2], [ 0.5, -0.3, -0.2],
        [ 0.5,  0.3, -0.2], [-0.5,  0.3, -0.2],
        [-0.5, -0.3,  0.2], [ 0.5, -0.3,  0.2],
        [ 0.5,  0.3,  0.2], [-0.5,  0.3,  0.2],
    ])
    return vertices


def get_cube_faces():
    """Get faces of the cube."""
    return [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [0,3,7,4], [1,2,6,5]]


def rotate_vertices(vertices, quaternion):
    """Rotate vertices using quaternion."""
    q_scipy = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    rot = R.from_quat(q_scipy)
    return rot.apply(vertices)


class IMU_Visualizer:
    def __init__(self, port='COM3', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.filter = ComplementaryFilter(dt=0.01, alpha=0.98)  # Fast response!

        self.quaternion = np.array([1., 0., 0., 0.])
        self.euler = np.array([0., 0., 0.])

        self.base_vertices = get_cube_vertices()
        self.faces = get_cube_faces()

        self.setup_plot()

    def setup_plot(self):
        self.fig = plt.figure(figsize=(14, 8))

        self.ax = self.fig.add_subplot(121, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('IMU Orientation - FAST MODE', fontsize=14, fontweight='bold')

        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])

        axis_length = 0.8
        self.ax.quiver(0,0,0, axis_length,0,0, color='r', arrow_length_ratio=0.1, linewidth=2)
        self.ax.quiver(0,0,0, 0,axis_length,0, color='g', arrow_length_ratio=0.1, linewidth=2)
        self.ax.quiver(0,0,0, 0,0,axis_length, color='b', arrow_length_ratio=0.1, linewidth=2)

        self.cube = None

        self.ax_text = self.fig.add_subplot(122)
        self.ax_text.axis('off')
        self.text_display = self.ax_text.text(0.1, 0.5, '', fontsize=16, family='monospace',
                                              verticalalignment='center')

        plt.tight_layout()

    def connect(self):
        try:
            self.serial = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=0.01)
            time.sleep(2)
            self.serial.reset_input_buffer()
            print(f"Connected to {self.port}")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            return False

    def read_imu(self):
        if self.serial and self.serial.in_waiting > 0:
            try:
                line = self.serial.readline().decode('utf-8', errors='ignore')
                parsed = parse_imu_data(line)

                if parsed is not None:
                    accel_raw, gyro_raw, mag_raw = parsed

                    # Apply calibrations
                    accel = accel_raw * ACCEL_SCALE
                    gyro = gyro_raw - GYRO_BIAS
                    mag = (mag_raw - MAG_OFFSET) @ MAG_A.T

                    # Update filter (FAST!)
                    self.filter.update(accel, gyro, mag)

                    self.quaternion = self.filter.get_quaternion()
                    self.euler = self.filter.get_euler()

                    return True
            except:
                pass
        return False

    def update_plot(self, frame):
        # Read ALL available samples for instant response
        for _ in range(20):  # Process up to 20 samples per frame
            self.read_imu()

        rotated_vertices = rotate_vertices(self.base_vertices, self.quaternion)
        cube_faces = [[rotated_vertices[j] for j in face] for face in self.faces]

        if self.cube:
            self.cube.remove()

        self.cube = Poly3DCollection(cube_faces, alpha=0.7, facecolor='cyan',
                                     edgecolor='black', linewidth=2)
        self.ax.add_collection3d(self.cube)

        # Draw body axes
        axis_len = 0.6
        x_axis = rotate_vertices(np.array([[0,0,0], [axis_len,0,0]]), self.quaternion)
        self.ax.plot(x_axis[:,0], x_axis[:,1], x_axis[:,2], 'r-', linewidth=3)

        y_axis = rotate_vertices(np.array([[0,0,0], [0,axis_len,0]]), self.quaternion)
        self.ax.plot(y_axis[:,0], y_axis[:,1], y_axis[:,2], 'g-', linewidth=3)

        z_axis = rotate_vertices(np.array([[0,0,0], [0,0,axis_len]]), self.quaternion)
        self.ax.plot(z_axis[:,0], z_axis[:,1], z_axis[:,2], 'b-', linewidth=3)

        text = f"""
╔════════════════════════════════════╗
║   FAST IMU VISUALIZATION MODE      ║
╚════════════════════════════════════╝

  EULER ANGLES (degrees)
  ────────────────────────────
  Roll  (X): {self.euler[0]:8.2f}°
  Pitch (Y): {self.euler[1]:8.2f}°
  Yaw   (Z): {self.euler[2]:8.2f}°

  QUATERNION
  ────────────────────────────
  w: {self.quaternion[0]:7.4f}
  x: {self.quaternion[1]:7.4f}
  y: {self.quaternion[2]:7.4f}
  z: {self.quaternion[3]:7.4f}

  FILTER: Complementary (98% gyro)
  ────────────────────────────
  ✓ INSTANT response
  ✓ No lag
  ✓ Drift-free (mag + accel correction)

╔════════════════════════════════════╗
║  Move the IMU - see instant update!║
╚════════════════════════════════════╝
"""
        self.text_display.set_text(text)

    def run(self):
        if not self.connect():
            print("Failed to connect!")
            return

        print("\n" + "="*60)
        print("FAST 3D IMU VISUALIZATION")
        print("="*60)
        print("\nUsing Complementary Filter (98% gyro trust)")
        print("  - INSTANT response to movement")
        print("  - No lag or delay")
        print("  - Accel + Mag correct for drift")
        print("\nClose window to exit.")
        print("="*60 + "\n")

        anim = FuncAnimation(self.fig, self.update_plot, interval=20,  # 50Hz update rate
                           blit=False, cache_frame_data=False)
        plt.show()

        if self.serial:
            self.serial.close()


if __name__ == "__main__":
    ports = find_serial_ports()
    if not ports:
        print("No serial ports found!")
        exit(1)

    viz = IMU_Visualizer(port=ports[0], baudrate=115200)
    viz.run()
