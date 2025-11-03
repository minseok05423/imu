"""
Real-time 3D visualization using 6-DOF only (accel + gyro).
NO MAGNETOMETER - perfect for indoor use!

Roll and Pitch: Accurate and stable
Yaw: Will drift slowly (no magnetic reference) but responsive to rotation
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

# ============================================================================


class ComplementaryFilter6DOF:
    """
    6-DOF Complementary Filter (Accel + Gyro ONLY).
    No magnetometer interference!
    """

    def __init__(self, dt=0.01, alpha=0.98):
        """
        Initialize 6-DOF filter.

        Args:
            dt: Time step (seconds)
            alpha: Filter weight (higher = trust gyro more)
        """
        self.dt = dt
        self.alpha = alpha

        # Current orientation as quaternion [w, x, y, z]
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, accel, gyro):
        """
        Update orientation with accel and gyro only.

        Args:
            accel: Accelerometer [ax, ay, az] in m/s²
            gyro: Gyroscope [wx, wy, wz] in rad/s
        """
        # ========================================
        # STEP 1: Integrate gyroscope
        # ========================================

        w_norm = np.linalg.norm(gyro)
        if w_norm > 0.001:
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

        q_gyro = q_gyro / np.linalg.norm(q_gyro)

        # ========================================
        # STEP 2: Calculate roll/pitch from accelerometer
        # ========================================

        # Normalize accelerometer
        accel_norm = np.linalg.norm(accel)
        if accel_norm < 0.1:
            # Accelerometer reading too small, just use gyro
            self.q = q_gyro
            return

        accel = accel / accel_norm

        # Calculate roll and pitch from accelerometer
        # Gravity vector in world frame: [0, 0, 1] (pointing down)
        # We want to find rotation that maps [0, 0, 1] to accel reading

        # Simple method: calculate rotation to align Z axis with gravity
        roll_accel = np.arctan2(accel[1], accel[2])
        pitch_accel = np.arctan2(-accel[0], np.sqrt(accel[1]**2 + accel[2]**2))

        # Get current yaw from gyro integration (preserve it!)
        euler_gyro = self.quaternion_to_euler(q_gyro)
        yaw_gyro = euler_gyro[2]

        # Build quaternion from roll, pitch (from accel), yaw (from gyro)
        q_accel = self.euler_to_quaternion(roll_accel, pitch_accel, yaw_gyro)

        # ========================================
        # STEP 3: Complementary filter
        # ========================================

        # Blend: mostly gyro (fast) + a bit of accel (drift correction for roll/pitch)
        q_fused = self.slerp(q_accel, q_gyro, self.alpha)

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

    def quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw) in radians."""
        w, x, y, z = q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion."""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z])

    def slerp(self, q1, q2, t):
        """Spherical linear interpolation."""
        dot = np.dot(q1, q2)

        if dot < 0.0:
            q2 = -q2
            dot = -dot

        dot = np.clip(dot, -1.0, 1.0)
        theta = np.arccos(dot)

        if abs(theta) < 0.001:
            return q1 * (1 - t) + q2 * t

        return (q1 * np.sin((1-t) * theta) + q2 * np.sin(t * theta)) / np.sin(theta)

    def get_quaternion(self):
        """Get current quaternion."""
        return self.q

    def get_euler(self):
        """Get Euler angles in degrees."""
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
    """Parse IMU data."""
    try:
        line = line.strip()

        if ',' in line and '{' not in line:
            values = [float(x) for x in line.split(',')]
            if len(values) >= 6:
                return np.array(values[0:3]), np.array(values[3:6])

        elif line.startswith('{'):
            data = json.loads(line)
            return (np.array([data['ax'], data['ay'], data['az']]),
                   np.array([data['gx'], data['gy'], data['gz']]))

    except:
        return None

    return None


def get_cube_vertices():
    """Get cube vertices."""
    return np.array([
        [-0.5, -0.3, -0.2], [ 0.5, -0.3, -0.2],
        [ 0.5,  0.3, -0.2], [-0.5,  0.3, -0.2],
        [-0.5, -0.3,  0.2], [ 0.5, -0.3,  0.2],
        [ 0.5,  0.3,  0.2], [-0.5,  0.3,  0.2],
    ])


def get_cube_faces():
    """Get cube faces."""
    return [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [0,3,7,4], [1,2,6,5]]


def rotate_vertices(vertices, quaternion):
    """Rotate vertices."""
    q_scipy = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    rot = R.from_quat(q_scipy)
    return rot.apply(vertices)


class IMU_Visualizer:
    def __init__(self, port='COM3', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.filter = ComplementaryFilter6DOF(dt=0.01, alpha=0.98)

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
        self.ax.set_title('6-DOF IMU (Accel + Gyro Only)', fontsize=14, fontweight='bold')

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
                    accel_raw, gyro_raw = parsed

                    # Apply calibrations
                    accel = accel_raw * ACCEL_SCALE
                    gyro = gyro_raw - GYRO_BIAS

                    # Update filter (NO MAGNETOMETER!)
                    self.filter.update(accel, gyro)

                    self.quaternion = self.filter.get_quaternion()
                    self.euler = self.filter.get_euler()

                    return True
            except:
                pass
        return False

    def update_plot(self, frame):
        # Process all available data
        for _ in range(20):
            self.read_imu()

        rotated_vertices = rotate_vertices(self.base_vertices, self.quaternion)
        cube_faces = [[rotated_vertices[j] for j in face] for face in self.faces]

        if self.cube:
            self.cube.remove()

        self.cube = Poly3DCollection(cube_faces, alpha=0.7, facecolor='lime',
                                     edgecolor='black', linewidth=2)
        self.ax.add_collection3d(self.cube)

        # Body axes
        axis_len = 0.6
        x_axis = rotate_vertices(np.array([[0,0,0], [axis_len,0,0]]), self.quaternion)
        self.ax.plot(x_axis[:,0], x_axis[:,1], x_axis[:,2], 'r-', linewidth=3)

        y_axis = rotate_vertices(np.array([[0,0,0], [0,axis_len,0]]), self.quaternion)
        self.ax.plot(y_axis[:,0], y_axis[:,1], y_axis[:,2], 'g-', linewidth=3)

        z_axis = rotate_vertices(np.array([[0,0,0], [0,0,axis_len]]), self.quaternion)
        self.ax.plot(z_axis[:,0], z_axis[:,1], z_axis[:,2], 'b-', linewidth=3)

        text = f"""
╔════════════════════════════════════╗
║   6-DOF MODE (NO MAGNETOMETER)     ║
╚════════════════════════════════════╝

  EULER ANGLES (degrees)
  ────────────────────────────
  Roll  (X): {self.euler[0]:8.2f}°  ✓ Stable
  Pitch (Y): {self.euler[1]:8.2f}°  ✓ Stable
  Yaw   (Z): {self.euler[2]:8.2f}°  ⚠ Drifts slowly

  QUATERNION
  ────────────────────────────
  w: {self.quaternion[0]:7.4f}
  x: {self.quaternion[1]:7.4f}
  y: {self.quaternion[2]:7.4f}
  z: {self.quaternion[3]:7.4f}

  STATUS
  ────────────────────────────
  ✓ NO magnetic interference
  ✓ INSTANT response
  ✓ Roll/Pitch accurate
  ⚠ Yaw drifts (no mag reference)

╔════════════════════════════════════╗
║ Perfect for indoor use! No mag    ║
║ interference from electronics!     ║
╚════════════════════════════════════╝
"""
        self.text_display.set_text(text)

    def run(self):
        if not self.connect():
            print("Failed to connect!")
            return

        print("\n" + "="*60)
        print("6-DOF IMU VISUALIZATION (Accel + Gyro Only)")
        print("="*60)
        print("\nNO Magnetometer = NO magnetic interference!")
        print("  ✓ Roll and Pitch: Stable and accurate")
        print("  ⚠ Yaw: Will drift slowly (no compass)")
        print("\nPerfect for indoor use with electronics nearby!")
        print("\nClose window to exit.")
        print("="*60 + "\n")

        anim = FuncAnimation(self.fig, self.update_plot, interval=20,
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
