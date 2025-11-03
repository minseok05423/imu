"""
3D visualization of recorded/estimated orientations.
Can compare with ground truth if available.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
import sys


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


class OrientationVisualizer:
    """Visualize orientation data in 3D."""

    def __init__(self, timestamps, quaternions, ground_truth_quaternions=None):
        """
        Initialize visualizer.

        Args:
            timestamps: Time array [N]
            quaternions: Estimated quaternions [N, 4] in [w,x,y,z] format
            ground_truth_quaternions: Optional ground truth [N, 4]
        """
        self.timestamps = timestamps
        self.quaternions = quaternions
        self.ground_truth = ground_truth_quaternions
        self.has_ground_truth = ground_truth_quaternions is not None

        self.N = len(timestamps)
        self.current_idx = 0

        self.base_vertices = get_cube_vertices()
        self.faces = get_cube_faces()

        # Calculate playback speed
        self.duration = timestamps[-1] - timestamps[0]
        self.sample_rate = self.N / self.duration

        # Convert to Euler angles
        self.euler = self.quaternions_to_euler(quaternions)
        if self.has_ground_truth:
            self.euler_gt = self.quaternions_to_euler(ground_truth_quaternions)

        self.setup_plot()

    def quaternions_to_euler(self, quaternions):
        """Convert quaternions to Euler angles."""
        N = len(quaternions)
        euler = np.zeros((N, 3))
        for i in range(N):
            q = quaternions[i]
            q_scipy = [q[1], q[2], q[3], q[0]]
            rot = R.from_quat(q_scipy)
            euler[i] = rot.as_euler('xyz', degrees=True)
        return euler

    def setup_plot(self):
        """Setup matplotlib figure."""
        if self.has_ground_truth:
            self.fig = plt.figure(figsize=(18, 8))
            gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

            # 3D views
            self.ax_est = self.fig.add_subplot(gs[:, 0], projection='3d')
            self.ax_gt = self.fig.add_subplot(gs[:, 1], projection='3d')

            # Error plot
            self.ax_error = self.fig.add_subplot(gs[0, 2])
            self.ax_text = self.fig.add_subplot(gs[1, 2])

            self.setup_3d_axis(self.ax_est, 'Estimated Orientation')
            self.setup_3d_axis(self.ax_gt, 'Ground Truth')
        else:
            self.fig = plt.figure(figsize=(14, 8))

            self.ax_est = self.fig.add_subplot(121, projection='3d')
            self.ax_text = self.fig.add_subplot(122)

            self.setup_3d_axis(self.ax_est, 'Estimated Orientation')

        # Text display
        self.ax_text.axis('off')
        self.text_display = self.ax_text.text(0.1, 0.5, '', fontsize=12, family='monospace',
                                              verticalalignment='center')

        self.cube_est = None
        self.cube_gt = None

    def setup_3d_axis(self, ax, title):
        """Setup a 3D axis."""
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title, fontsize=12, fontweight='bold')

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        # World frame axes
        axis_length = 0.8
        ax.quiver(0,0,0, axis_length,0,0, color='red', arrow_length_ratio=0.1, linewidth=2, alpha=0.3)
        ax.quiver(0,0,0, 0,axis_length,0, color='green', arrow_length_ratio=0.1, linewidth=2, alpha=0.3)
        ax.quiver(0,0,0, 0,0,axis_length, color='blue', arrow_length_ratio=0.1, linewidth=2, alpha=0.3)

    def update_plot(self, frame):
        """Update animation frame."""
        # Advance to next sample
        self.current_idx = frame % self.N

        t = self.timestamps[self.current_idx]
        q_est = self.quaternions[self.current_idx]
        euler_est = self.euler[self.current_idx]

        # Update estimated orientation
        self.draw_orientation(self.ax_est, q_est, 'cyan')

        if self.has_ground_truth:
            q_gt = self.ground_truth[self.current_idx]
            euler_gt = self.euler_gt[self.current_idx]

            # Update ground truth
            self.draw_orientation(self.ax_gt, q_gt, 'lime')

            # Calculate error
            error = euler_est - euler_gt

            # Update error plot
            self.ax_error.clear()
            self.ax_error.set_title('Orientation Error', fontsize=11, fontweight='bold')

            history_samples = min(500, self.current_idx)
            start_idx = max(0, self.current_idx - history_samples)
            time_window = self.timestamps[start_idx:self.current_idx+1]
            error_window = self.euler[start_idx:self.current_idx+1] - self.euler_gt[start_idx:self.current_idx+1]

            self.ax_error.plot(time_window, error_window[:, 0], 'r-', label='Roll error', linewidth=1)
            self.ax_error.plot(time_window, error_window[:, 1], 'g-', label='Pitch error', linewidth=1)
            self.ax_error.plot(time_window, error_window[:, 2], 'b-', label='Yaw error', linewidth=1)
            self.ax_error.axhline(0, color='black', linestyle='--', alpha=0.3)
            self.ax_error.set_xlabel('Time (s)', fontsize=9)
            self.ax_error.set_ylabel('Error (degrees)', fontsize=9)
            self.ax_error.legend(loc='upper right', fontsize=8)
            self.ax_error.grid(True, alpha=0.3)

            # Text display with ground truth
            text = f"""
╔═══════════════════════════════════════╗
║   ORIENTATION PLAYBACK WITH GT        ║
╚═══════════════════════════════════════╝

Time: {t:.2f} / {self.duration:.2f} seconds
Progress: {100*self.current_idx/self.N:.1f}%

ESTIMATED ORIENTATION:
─────────────────────────────────────
  Roll  (X): {euler_est[0]:8.2f}°
  Pitch (Y): {euler_est[1]:8.2f}°
  Yaw   (Z): {euler_est[2]:8.2f}°

GROUND TRUTH:
─────────────────────────────────────
  Roll  (X): {euler_gt[0]:8.2f}°
  Pitch (Y): {euler_gt[1]:8.2f}°
  Yaw   (Z): {euler_gt[2]:8.2f}°

ERROR:
─────────────────────────────────────
  Roll:  {error[0]:7.2f}°
  Pitch: {error[1]:7.2f}°
  Yaw:   {error[2]:7.2f}°

RMS Error: {np.sqrt(np.mean(error**2)):.2f}°
"""
        else:
            # Text display without ground truth
            text = f"""
╔═══════════════════════════════════════╗
║   ORIENTATION PLAYBACK                ║
╚═══════════════════════════════════════╝

Time: {t:.2f} / {self.duration:.2f} seconds
Progress: {100*self.current_idx/self.N:.1f}%

ORIENTATION:
─────────────────────────────────────
  Roll  (X): {euler_est[0]:8.2f}°
  Pitch (Y): {euler_est[1]:8.2f}°
  Yaw   (Z): {euler_est[2]:8.2f}°

QUATERNION:
─────────────────────────────────────
  w: {q_est[0]:7.4f}
  x: {q_est[1]:7.4f}
  y: {q_est[2]:7.4f}
  z: {q_est[3]:7.4f}

Sample rate: {self.sample_rate:.1f} Hz
Total samples: {self.N}
"""

        self.text_display.set_text(text)

    def draw_orientation(self, ax, quaternion, color):
        """Draw cube and axes for given orientation."""
        # Rotate cube
        rotated_vertices = rotate_vertices(self.base_vertices, quaternion)
        cube_faces = [[rotated_vertices[j] for j in face] for face in self.faces]

        # Remove old cube if exists
        if ax == self.ax_est and self.cube_est:
            self.cube_est.remove()
        elif ax == self.ax_gt and self.cube_gt:
            self.cube_gt.remove()

        # Draw new cube
        cube = Poly3DCollection(cube_faces, alpha=0.7, facecolor=color,
                               edgecolor='black', linewidth=2)
        ax.add_collection3d(cube)

        if ax == self.ax_est:
            self.cube_est = cube
        elif ax == self.ax_gt:
            self.cube_gt = cube

        # Draw body axes
        axis_len = 0.6
        x_axis = rotate_vertices(np.array([[0,0,0], [axis_len,0,0]]), quaternion)
        y_axis = rotate_vertices(np.array([[0,0,0], [0,axis_len,0]]), quaternion)
        z_axis = rotate_vertices(np.array([[0,0,0], [0,0,axis_len]]), quaternion)

        ax.plot(x_axis[:,0], x_axis[:,1], x_axis[:,2], 'r-', linewidth=3)
        ax.plot(y_axis[:,0], y_axis[:,1], y_axis[:,2], 'g-', linewidth=3)
        ax.plot(z_axis[:,0], z_axis[:,1], z_axis[:,2], 'b-', linewidth=3)

    def run(self):
        """Run the visualization."""
        print("\n" + "="*60)
        print("ORIENTATION PLAYBACK")
        print("="*60)
        if self.has_ground_truth:
            print("Comparing estimated vs ground truth orientations")
        else:
            print("Visualizing estimated orientations")
        print(f"Duration: {self.duration:.2f} seconds")
        print(f"Samples: {self.N}")
        print(f"Sample rate: {self.sample_rate:.1f} Hz")
        print("\nClose window to exit.")
        print("="*60 + "\n")

        # Animation at 30 FPS, but skip frames to play at ~2x speed
        interval = 1000 / 30  # 30 FPS
        frame_skip = max(1, int(self.sample_rate / 60))  # Play at ~2x real speed

        anim = FuncAnimation(self.fig, self.update_plot, frames=range(0, self.N, frame_skip),
                           interval=interval, blit=False, cache_frame_data=False, repeat=True)
        plt.show()


def load_orientation_data(filename):
    """Load orientation data from NPZ file."""
    data = np.load(filename)
    timestamps = data['timestamps']
    quaternions = data['quaternions']
    return timestamps, quaternions


def load_ground_truth(filename):
    """Load ground truth from NPZ file."""
    data = np.load(filename)
    timestamps = data['timestamps']
    quaternions = data['quaternions']
    return timestamps, quaternions


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_orientations.py <orientations.npz> [ground_truth.npz]")
        print("\nExample:")
        print("  python visualize_orientations.py imu_data_orientations.npz")
        print("  python visualize_orientations.py synthetic_imu_data_orientations.npz synthetic_ground_truth.npz")
        sys.exit(1)

    orientation_file = sys.argv[1]

    print(f"Loading orientations from {orientation_file}...")
    timestamps, quaternions = load_orientation_data(orientation_file)
    print(f"  Loaded {len(quaternions)} orientations\n")

    ground_truth_quaternions = None
    if len(sys.argv) > 2:
        ground_truth_file = sys.argv[2]
        print(f"Loading ground truth from {ground_truth_file}...")
        gt_timestamps, ground_truth_quaternions = load_ground_truth(ground_truth_file)

        # Verify timestamps match
        if len(gt_timestamps) != len(timestamps):
            print("WARNING: Ground truth and estimation have different lengths!")
            print(f"  Estimation: {len(timestamps)} samples")
            print(f"  Ground truth: {len(gt_timestamps)} samples")
            # Truncate to shorter length
            min_len = min(len(timestamps), len(gt_timestamps))
            timestamps = timestamps[:min_len]
            quaternions = quaternions[:min_len]
            ground_truth_quaternions = ground_truth_quaternions[:min_len]
        print(f"  Loaded {len(ground_truth_quaternions)} ground truth orientations\n")

    viz = OrientationVisualizer(timestamps, quaternions, ground_truth_quaternions)
    viz.run()
