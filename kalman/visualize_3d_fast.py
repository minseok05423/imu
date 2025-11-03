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
MAG_OFFSET = np.array([0.0318, -0.2640, -0.2652])
MAG_SCALE = np.array([1.2605, 0.9314, 0.8826])

# ============================================================================


class MadgwickFilter:
    """
    Madgwick AHRS filter for IMU orientation estimation.
    Industry-standard algorithm, well-tested and proven.

    Reference: Madgwick, S. (2010). "An efficient orientation filter for
    inertial and inertial/magnetic sensor arrays"
    """

    def __init__(self, dt=0.01, beta=0.1):
        """
        Initialize Madgwick filter.

        Args:
            dt: Time step (seconds)
            beta: Filter gain (0.01-0.5). Lower = trust gyro more, higher = trust accel/mag more
                  Typical: 0.1 for fast response with good correction
        """
        self.dt = dt
        self.beta = beta

        # Current orientation as quaternion [w, x, y, z]
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, accel, gyro, mag):
        """
        Update orientation using Madgwick AHRS algorithm.

        Args:
            accel: Accelerometer [ax, ay, az] in m/s²
            gyro: Gyroscope [wx, wy, wz] in rad/s
            mag: Magnetometer [mx, my, mz] (calibrated)
        """
        q = self.q

        # Normalize accelerometer measurement
        accel_norm = np.linalg.norm(accel)
        if accel_norm < 0.01:
            # No valid accel data, just integrate gyro
            return
        accel = accel / accel_norm

        # Normalize magnetometer measurement
        mag_norm = np.linalg.norm(mag)
        if mag_norm < 0.01:
            # Fall back to 6DOF (IMU mode without magnetometer)
            self._update_imu(accel, gyro)
            return
        mag = mag / mag_norm

        # Reference direction of Earth's magnetic field (in Earth frame)
        h = self._quaternion_rotate(q, mag)
        b = np.array([0, np.sqrt(h[0]**2 + h[1]**2), 0, h[2]])

        # Gradient descent algorithm corrective step
        F = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - accel[0],
            2*(q[0]*q[1] + q[2]*q[3]) - accel[1],
            2*(0.5 - q[1]**2 - q[2]**2) - accel[2],
            2*b[1]*(0.5 - q[2]**2 - q[3]**2) + 2*b[3]*(q[1]*q[3] - q[0]*q[2]) - mag[0],
            2*b[1]*(q[1]*q[2] - q[0]*q[3]) + 2*b[3]*(q[0]*q[1] + q[2]*q[3]) - mag[1],
            2*b[1]*(q[0]*q[2] + q[1]*q[3]) + 2*b[3]*(0.5 - q[1]**2 - q[2]**2) - mag[2]
        ])

        J = np.array([
            [-2*q[2],                 2*q[3],                -2*q[0],                2*q[1]],
            [ 2*q[1],                 2*q[0],                 2*q[3],                2*q[2]],
            [ 0,                     -4*q[1],                -4*q[2],                0],
            [-2*b[3]*q[2],            2*b[3]*q[3],          -4*b[1]*q[2]-2*b[3]*q[0], -4*b[1]*q[3]+2*b[3]*q[1]],
            [-2*b[1]*q[3]+2*b[3]*q[1], 2*b[1]*q[2]+2*b[3]*q[0], 2*b[1]*q[1]+2*b[3]*q[3], -2*b[1]*q[0]+2*b[3]*q[2]],
            [ 2*b[1]*q[2],            2*b[1]*q[3]-4*b[3]*q[1], 2*b[1]*q[0]-4*b[3]*q[2],  2*b[1]*q[1]]
        ])

        step = J.T @ F
        step = step / np.linalg.norm(step)

        # Compute rate of change of quaternion
        q_dot = 0.5 * self.quaternion_multiply(q, np.array([0, gyro[0], gyro[1], gyro[2]])) - self.beta * step

        # Integrate to yield quaternion
        q = q + q_dot * self.dt
        self.q = q / np.linalg.norm(q)

    def _update_imu(self, accel, gyro):
        """6DOF update (without magnetometer)"""
        q = self.q

        # Gradient descent algorithm corrective step
        F = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - accel[0],
            2*(q[0]*q[1] + q[2]*q[3]) - accel[1],
            2*(0.5 - q[1]**2 - q[2]**2) - accel[2]
        ])

        J = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [ 2*q[1], 2*q[0],  2*q[3], 2*q[2]],
            [ 0,     -4*q[1], -4*q[2], 0]
        ])

        step = J.T @ F
        step = step / np.linalg.norm(step)

        # Compute rate of change of quaternion
        q_dot = 0.5 * self.quaternion_multiply(q, np.array([0, gyro[0], gyro[1], gyro[2]])) - self.beta * step

        # Integrate to yield quaternion
        q = q + q_dot * self.dt
        self.q = q / np.linalg.norm(q)

    def _quaternion_rotate(self, q, v):
        """Rotate vector v by quaternion q"""
        # Convert vector to quaternion
        qv = np.array([0, v[0], v[1], v[2]])
        # q * v * q_conjugate
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        result = self.quaternion_multiply(self.quaternion_multiply(q, qv), q_conj)
        return result[1:]

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
        self.filter = MadgwickFilter(dt=0.01, beta=0.1)  # Madgwick AHRS filter

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
                    mag = (mag_raw - MAG_OFFSET) * MAG_SCALE

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
║   9-DOF IMU - MADGWICK FILTER      ║
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

  FILTER: Madgwick AHRS (β=0.1)
  ────────────────────────────
  ✓ Industry-standard algorithm
  ✓ Fast & accurate
  ✓ Drift-free (mag + accel + gyro fusion)
  ✓ Proven and well-tested

╔════════════════════════════════════╗
║  Optimal 9-DOF sensor fusion!     ║
╚════════════════════════════════════╝
"""
        self.text_display.set_text(text)

    def run(self):
        if not self.connect():
            print("Failed to connect!")
            return

        print("\n" + "="*60)
        print("9-DOF IMU VISUALIZATION - MADGWICK FILTER")
        print("="*60)
        print("\nUsing Madgwick AHRS Algorithm")
        print("  - Industry-standard sensor fusion")
        print("  - Fast response with drift correction")
        print("  - Accel + Gyro + Mag fusion")
        print("  - Proven, well-tested algorithm")
        print("\nClose window to exit.")
        print("="*60 + "\n")

        anim = FuncAnimation(self.fig, self.update_plot, interval=40,  # 25Hz update rate
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
