"""
Real-time 3D visualization of recorded IMU data with timestamps.
Uses Madgwick filter for orientation estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
import sys


# Calibration values (from previous calibrations)
GYRO_BIAS = np.array([0.0152, -0.0037, -0.0074])
ACCEL_SCALE = np.array([1.0, 1.0, 1.0])
MAG_OFFSET = np.array([0.0318, -0.2640, -0.2652])
MAG_SCALE = np.array([1.2605, 0.9314, 0.8826])


class MadgwickAHRS:
    """Madgwick AHRS filter for orientation estimation."""

    def __init__(self, sample_rate=100.0, beta=0.1):
        self.sample_rate = sample_rate
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]

    def update(self, gyro, accel, mag, dt=None):
        """Update filter with gyroscope, accelerometer, and magnetometer data."""
        if dt is None:
            dt = 1.0 / self.sample_rate

        q = self.q
        gx, gy, gz = gyro
        ax, ay, az = accel
        mx, my, mz = mag

        # Normalize accelerometer
        accel_norm = np.sqrt(ax**2 + ay**2 + az**2)
        if accel_norm > 0:
            ax /= accel_norm
            ay /= accel_norm
            az /= accel_norm
        else:
            return

        # Normalize magnetometer
        mag_norm = np.sqrt(mx**2 + my**2 + mz**2)
        if mag_norm > 0:
            mx /= mag_norm
            my /= mag_norm
            mz /= mag_norm
        else:
            return

        # Extract quaternion components
        q0, q1, q2, q3 = q

        # Reference direction of Earth's magnetic field
        h0 = 2.0 * (mx * (0.5 - q2**2 - q3**2) + my * (q1*q2 - q0*q3) + mz * (q1*q3 + q0*q2))
        h1 = 2.0 * (mx * (q1*q2 + q0*q3) + my * (0.5 - q1**2 - q3**2) + mz * (q2*q3 - q0*q1))
        bx = np.sqrt(h0**2 + h1**2)
        bz = 2.0 * (mx * (q1*q3 - q0*q2) + my * (q2*q3 + q0*q1) + mz * (0.5 - q1**2 - q2**2))

        # Gradient descent algorithm corrective step
        F = np.array([
            2.0*(q1*q3 - q0*q2) - ax,
            2.0*(q0*q1 + q2*q3) - ay,
            2.0*(0.5 - q1**2 - q2**2) - az,
            2.0*bx*(0.5 - q2**2 - q3**2) + 2.0*bz*(q1*q3 - q0*q2) - mx,
            2.0*bx*(q1*q2 - q0*q3) + 2.0*bz*(q0*q1 + q2*q3) - my,
            2.0*bx*(q0*q2 + q1*q3) + 2.0*bz*(0.5 - q1**2 - q2**2) - mz
        ])

        J = np.array([
            [-2.0*q2, 2.0*q3, -2.0*q0, 2.0*q1],
            [2.0*q1, 2.0*q0, 2.0*q3, 2.0*q2],
            [0, -4.0*q1, -4.0*q2, 0],
            [-2.0*bz*q2, 2.0*bz*q3, -4.0*bx*q2 - 2.0*bz*q0, -4.0*bx*q3 + 2.0*bz*q1],
            [-2.0*bx*q3 + 2.0*bz*q1, 2.0*bx*q2 + 2.0*bz*q0, 2.0*bx*q1 + 2.0*bz*q3, -2.0*bx*q0 + 2.0*bz*q2],
            [2.0*bx*q2, 2.0*bx*q3 - 4.0*bz*q1, 2.0*bx*q0 - 4.0*bz*q2, 2.0*bx*q1]
        ])

        step = J.T @ F
        step_norm = np.linalg.norm(step)
        if step_norm > 0:
            step /= step_norm

        # Compute rate of change of quaternion
        qDot = 0.5 * self.quaternion_multiply(q, np.array([0, gx, gy, gz])) - self.beta * step

        # Integrate to yield quaternion
        q = q + qDot * dt
        self.q = q / np.linalg.norm(q)

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


def get_cube_vertices():
    """Get vertices of a cube (IMU body)."""
    return np.array([
        [-0.5, -0.3, -0.2], [ 0.5, -0.3, -0.2],
        [ 0.5,  0.3, -0.2], [-0.5,  0.3, -0.2],
        [-0.5, -0.3,  0.2], [ 0.5, -0.3,  0.2],
        [ 0.5,  0.3,  0.2], [-0.5,  0.3,  0.2],
    ])


def get_cube_faces():
    """Get faces of the cube."""
    return [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [0,3,7,4], [1,2,6,5]]


def rotate_vertices(vertices, quaternion):
    """Rotate vertices using quaternion [w, x, y, z]."""
    q_scipy = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    rot = R.from_quat(q_scipy)
    return rot.apply(vertices)


def quaternion_to_euler(quaternion):
    """Convert quaternion to Euler angles in degrees."""
    q_scipy = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    rot = R.from_quat(q_scipy)
    return rot.as_euler('xyz', degrees=True)


def load_imu_data(filename):
    """Load timestamped IMU data from CSV file."""
    data = []
    timestamps = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                values = [float(x) for x in line.split(',')]
                if len(values) >= 10:
                    timestamps.append(values[0])
                    data.append(values[1:10])  # ax,ay,az,gx,gy,gz,mx,my,mz
            except ValueError:
                continue

    return np.array(timestamps), np.array(data)


class RealtimeIMUVisualizer:
    """Visualize IMU orientation in real-time."""

    def __init__(self, timestamps, imu_data):
        self.timestamps = timestamps
        self.imu_data = imu_data
        self.N = len(timestamps)

        # Convert timestamps to seconds
        duration = timestamps[-1] - timestamps[0]
        if duration > 1000000:  # Microseconds
            self.timestamps = timestamps / 1000000.0
        elif duration > 1000:  # Milliseconds
            self.timestamps = timestamps / 1000.0

        self.duration = self.timestamps[-1] - self.timestamps[0]
        self.sample_rate = (self.N - 1) / self.duration

        # Apply calibrations
        self.accel = imu_data[:, 0:3] * ACCEL_SCALE
        self.gyro = imu_data[:, 3:6] - GYRO_BIAS
        self.mag = (imu_data[:, 6:9] - MAG_OFFSET) * MAG_SCALE

        # Initialize filter
        self.madgwick = MadgwickAHRS(sample_rate=self.sample_rate, beta=0.1)

        # Pre-compute all orientations
        print(f"Pre-computing orientations for {self.N} samples...")
        self.quaternions = np.zeros((self.N, 4))
        self.euler_angles = np.zeros((self.N, 3))

        for i in range(self.N):
            if i > 0:
                dt = self.timestamps[i] - self.timestamps[i-1]
            else:
                dt = 1.0 / self.sample_rate

            self.madgwick.update(self.gyro[i], self.accel[i], self.mag[i], dt)
            self.quaternions[i] = self.madgwick.q.copy()
            self.euler_angles[i] = quaternion_to_euler(self.madgwick.q)

            if (i+1) % 500 == 0:
                print(f"  Processed {i+1}/{self.N} samples ({100*(i+1)/self.N:.1f}%)")

        print("Pre-computation complete!\n")

        self.current_idx = 0
        self.base_vertices = get_cube_vertices()
        self.faces = get_cube_faces()

        self.setup_plot()

    def setup_plot(self):
        """Setup matplotlib figure."""
        self.fig = plt.figure(figsize=(16, 8))

        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_text = self.fig.add_subplot(122)

        # Setup 3D axis
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_title('Real-time IMU Orientation', fontsize=14, fontweight='bold')

        self.ax_3d.set_xlim([-1, 1])
        self.ax_3d.set_ylim([-1, 1])
        self.ax_3d.set_zlim([-1, 1])

        # World frame axes
        axis_length = 0.8
        self.ax_3d.quiver(0,0,0, axis_length,0,0, color='red', arrow_length_ratio=0.1, linewidth=2, alpha=0.3)
        self.ax_3d.quiver(0,0,0, 0,axis_length,0, color='green', arrow_length_ratio=0.1, linewidth=2, alpha=0.3)
        self.ax_3d.quiver(0,0,0, 0,0,axis_length, color='blue', arrow_length_ratio=0.1, linewidth=2, alpha=0.3)

        # Text display
        self.ax_text.axis('off')
        self.text_display = self.ax_text.text(0.1, 0.5, '', fontsize=12, family='monospace',
                                              verticalalignment='center')

        self.cube = None

    def update_plot(self, frame):
        """Update animation frame."""
        # Calculate which sample index to show based on elapsed time
        # This makes it play in real-time (1 second = 1 second)
        elapsed_time = frame / 30.0  # 30 FPS

        # Find the sample closest to this time
        idx = np.searchsorted(self.timestamps - self.timestamps[0], elapsed_time)
        if idx >= self.N:
            idx = self.N - 1

        self.current_idx = idx

        t = self.timestamps[self.current_idx] - self.timestamps[0]  # Relative time
        q = self.quaternions[self.current_idx]
        euler = self.euler_angles[self.current_idx]

        # Rotate cube
        rotated_vertices = rotate_vertices(self.base_vertices, q)
        cube_faces = [[rotated_vertices[j] for j in face] for face in self.faces]

        # Remove old cube
        if self.cube:
            self.cube.remove()

        # Draw new cube
        self.cube = Poly3DCollection(cube_faces, alpha=0.7, facecolor='cyan',
                                    edgecolor='black', linewidth=2)
        self.ax_3d.add_collection3d(self.cube)

        # Draw body axes
        axis_len = 0.6
        x_axis = rotate_vertices(np.array([[0,0,0], [axis_len,0,0]]), q)
        y_axis = rotate_vertices(np.array([[0,0,0], [0,axis_len,0]]), q)
        z_axis = rotate_vertices(np.array([[0,0,0], [0,0,axis_len]]), q)

        self.ax_3d.plot(x_axis[:,0], x_axis[:,1], x_axis[:,2], 'r-', linewidth=3)
        self.ax_3d.plot(y_axis[:,0], y_axis[:,1], y_axis[:,2], 'g-', linewidth=3)
        self.ax_3d.plot(z_axis[:,0], z_axis[:,1], z_axis[:,2], 'b-', linewidth=3)

        # Text display
        accel = self.accel[self.current_idx]
        gyro = self.gyro[self.current_idx]
        mag = self.mag[self.current_idx]

        text = f"""
╔═══════════════════════════════════════╗
║   REAL-TIME IMU PLAYBACK              ║
╚═══════════════════════════════════════╝

Time: {t:.2f} / {self.duration:.2f} seconds
Progress: {100*self.current_idx/self.N:.1f}%

ORIENTATION:
─────────────────────────────────────
  Roll  (X): {euler[0]:8.2f}°
  Pitch (Y): {euler[1]:8.2f}°
  Yaw   (Z): {euler[2]:8.2f}°

QUATERNION:
─────────────────────────────────────
  w: {q[0]:7.4f}
  x: {q[1]:7.4f}
  y: {q[2]:7.4f}
  z: {q[3]:7.4f}

SENSOR DATA (calibrated):
─────────────────────────────────────
Accelerometer (m/s²):
  X: {accel[0]:7.3f}  Y: {accel[1]:7.3f}  Z: {accel[2]:7.3f}

Gyroscope (rad/s):
  X: {gyro[0]:7.3f}  Y: {gyro[1]:7.3f}  Z: {gyro[2]:7.3f}

Magnetometer (normalized):
  X: {mag[0]:7.3f}  Y: {mag[1]:7.3f}  Z: {mag[2]:7.3f}

Sample rate: {self.sample_rate:.1f} Hz
Total samples: {self.N}
"""

        self.text_display.set_text(text)

    def run(self):
        """Run the visualization."""
        print("="*60)
        print("REAL-TIME IMU PLAYBACK")
        print("="*60)
        print(f"Duration: {self.duration:.2f} seconds")
        print(f"Samples: {self.N}")
        print(f"Sample rate: {self.sample_rate:.1f} Hz")
        print("\nClose window to exit.")
        print("="*60 + "\n")

        # Animation at 30 FPS, playing in real-time (1 second = 1 second)
        interval = 1000 / 30  # 30 FPS
        total_frames = int(self.duration * 30)  # Total frames for the animation

        anim = FuncAnimation(self.fig, self.update_plot, frames=total_frames,
                           interval=interval, blit=False, cache_frame_data=False, repeat=True)
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_real_imu_3d.py <imu_data.csv>")
        print("\nExample:")
        print("  python visualize_real_imu_3d.py real_imu_data.csv")
        sys.exit(1)

    data_file = sys.argv[1]

    print(f"Loading IMU data from {data_file}...\n")
    timestamps, imu_data = load_imu_data(data_file)

    if len(imu_data) == 0:
        print("ERROR: No data loaded!")
        sys.exit(1)

    print(f"Loaded {len(imu_data)} samples\n")

    viz = RealtimeIMUVisualizer(timestamps, imu_data)
    viz.run()
