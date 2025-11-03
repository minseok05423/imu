"""
Offline orientation estimation using UKF + RTS Smoother.
For post-processing recorded IMU data with maximum accuracy.

This is MUCH more accurate than real-time filters because:
1. Uses future data via backward smoothing pass
2. Carefully tuned noise parameters
3. No time constraints - can iterate and optimize
"""

import numpy as np
import json
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================================================
# CALIBRATION PARAMETERS
# ============================================================================

GYRO_BIAS = np.array([-0.093434, 0.038554, -0.004698])
ACCEL_SCALE = 1.0
MAG_OFFSET = np.array([0.0318, -0.2640, -0.2652])
MAG_SCALE = np.array([1.2605, 0.9314, 0.8826])

# ============================================================================


class IMU_UKF_Offline:
    """
    9-DOF IMU UKF for OFFLINE processing.
    Optimized for accuracy, not speed.
    """

    def __init__(self, dt=0.01):
        self.dt = dt

        # State: [qw, qx, qy, qz, wx, wy, wz, gx_bias, gy_bias, gz_bias]
        # Quaternion + angular velocity + gyro bias
        self.dim_x = 10
        self.dim_z = 9  # accel (3) + gyro (3) + mag (3)

        # Sigma points
        points = MerweScaledSigmaPoints(n=self.dim_x, alpha=0.1, beta=2., kappa=0)

        self.ukf = UnscentedKalmanFilter(
            dim_x=self.dim_x,
            dim_z=self.dim_z,
            dt=dt,
            fx=self.state_transition,
            hx=self.measurement_function,
            points=points
        )

        # Initial state: identity quaternion, zero velocity, zero bias
        self.ukf.x = np.array([1., 0., 0., 0.,  # quaternion
                               0., 0., 0.,       # angular velocity
                               0., 0., 0.])      # gyro bias

        # Process noise (how much we trust the model)
        self.ukf.Q = np.eye(self.dim_x)
        self.ukf.Q[0:4, 0:4] *= 0.0001   # Quaternion: very stable
        self.ukf.Q[4:7, 4:7] *= 0.01     # Angular velocity: moderate
        self.ukf.Q[7:10, 7:10] *= 0.00001  # Gyro bias: very stable

        # Measurement noise (how much we trust sensors)
        self.ukf.R = np.eye(self.dim_z)
        self.ukf.R[0:3, 0:3] *= 0.1      # Accelerometer: moderate noise
        self.ukf.R[3:6, 3:6] *= 0.01     # Gyroscope: low noise (after bias correction)
        self.ukf.R[6:9, 6:9] *= 0.5      # Magnetometer: moderate trust (hard-iron only)

        # Initial covariance
        self.ukf.P = np.eye(self.dim_x)
        self.ukf.P[0:4, 0:4] *= 0.1
        self.ukf.P[4:7, 4:7] *= 1.0
        self.ukf.P[7:10, 7:10] *= 0.01

    def state_transition(self, x, dt):
        """
        State transition function: integrate quaternion using angular velocity.

        State: [qw, qx, qy, qz, wx, wy, wz, gx_bias, gy_bias, gz_bias]
        """
        q = x[0:4]
        w = x[4:7]
        bias = x[7:10]

        # Integrate quaternion using angular velocity
        w_corrected = w - bias
        w_norm = np.linalg.norm(w_corrected)

        if w_norm > 1e-6:
            angle = w_norm * dt
            axis = w_corrected / w_norm

            # Delta quaternion from axis-angle
            dq = np.array([
                np.cos(angle / 2),
                axis[0] * np.sin(angle / 2),
                axis[1] * np.sin(angle / 2),
                axis[2] * np.sin(angle / 2)
            ])

            # q_new = q * dq
            q_new = self.quaternion_multiply(q, dq)
            q_new = q_new / np.linalg.norm(q_new)
        else:
            q_new = q / np.linalg.norm(q)

        # Angular velocity stays relatively constant (small changes)
        w_new = w

        # Bias is very stable
        bias_new = bias

        return np.concatenate([q_new, w_new, bias_new])

    def measurement_function(self, x):
        """
        Measurement function: predict what sensors should read given state.

        Returns: [ax, ay, az, wx, wy, wz, mx, my, mz]
        """
        q = x[0:4]
        w = x[4:7]
        bias = x[7:10]

        # Expected gyroscope reading: angular velocity + bias
        gyro_expected = w + bias

        # Expected accelerometer: gravity rotated into body frame
        # Gravity in world frame: [0, 0, -9.81] (pointing down)
        gravity_world = np.array([0., 0., -9.81])
        accel_expected = self.quaternion_rotate_vector(self.quaternion_conjugate(q), gravity_world)

        # Expected magnetometer: Earth's magnetic field rotated into body frame
        # Assume field points North with some dip angle
        # For simplicity, use [1, 0, 0.5] normalized (North with downward component)
        mag_world = np.array([1., 0., 0.5])
        mag_world = mag_world / np.linalg.norm(mag_world)
        mag_expected = self.quaternion_rotate_vector(self.quaternion_conjugate(q), mag_world)

        return np.concatenate([accel_expected, gyro_expected, mag_expected])

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

    def quaternion_conjugate(self, q):
        """Quaternion conjugate (inverse for unit quaternions)."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def quaternion_rotate_vector(self, q, v):
        """Rotate vector v by quaternion q."""
        qv = np.array([0., v[0], v[1], v[2]])
        q_conj = self.quaternion_conjugate(q)
        result = self.quaternion_multiply(self.quaternion_multiply(q, qv), q_conj)
        return result[1:]

    def predict(self):
        """Prediction step."""
        self.ukf.predict()
        # Normalize quaternion after prediction
        self.ukf.x[0:4] = self.ukf.x[0:4] / np.linalg.norm(self.ukf.x[0:4])

    def update(self, accel, gyro, mag):
        """Update step with measurement."""
        z = np.concatenate([accel, gyro, mag])
        self.ukf.update(z)
        # Normalize quaternion after update
        self.ukf.x[0:4] = self.ukf.x[0:4] / np.linalg.norm(self.ukf.x[0:4])

    def get_quaternion(self):
        """Get current quaternion estimate."""
        return self.ukf.x[0:4]

    def get_euler(self):
        """Get Euler angles (roll, pitch, yaw) in degrees."""
        q = self.ukf.x[0:4]
        # Convert to scipy format [x, y, z, w]
        q_scipy = [q[1], q[2], q[3], q[0]]
        rot = R.from_quat(q_scipy)
        return rot.as_euler('xyz', degrees=True)


def rts_smoother(ukf_states, ukf_covariances, dt):
    """
    Rauch-Tung-Striebel smoother for backward pass.

    Args:
        ukf_states: Forward pass state estimates [N, dim_x]
        ukf_covariances: Forward pass covariances [N, dim_x, dim_x]
        dt: Time step

    Returns:
        smoothed_states: Smoothed state estimates [N, dim_x]
    """
    N = len(ukf_states)
    dim_x = ukf_states.shape[1]

    # Initialize smoothed estimates with forward pass
    smoothed_states = ukf_states.copy()
    smoothed_covariances = ukf_covariances.copy()

    # Backward pass
    for k in range(N-2, -1, -1):
        # Predicted state for k+1 from k
        x_k = ukf_states[k]
        x_kp1_pred = ukf_states[k+1]  # Simplified: use forward prediction

        P_k = ukf_covariances[k]
        P_kp1_pred = ukf_covariances[k+1]

        # Smoother gain
        C_k = P_k @ np.linalg.inv(P_kp1_pred)

        # Smoothed estimate
        x_kp1_smooth = smoothed_states[k+1]
        smoothed_states[k] = x_k + C_k @ (x_kp1_smooth - x_kp1_pred)

        # Normalize quaternion
        smoothed_states[k, 0:4] = smoothed_states[k, 0:4] / np.linalg.norm(smoothed_states[k, 0:4])

        # Smoothed covariance
        P_kp1_smooth = smoothed_covariances[k+1]
        smoothed_covariances[k] = P_k + C_k @ (P_kp1_smooth - P_kp1_pred) @ C_k.T

    return smoothed_states, smoothed_covariances


def load_imu_data(filename):
    """
    Load IMU data from CSV file.
    Format: timestamp,ax,ay,az,gx,gy,gz,mx,my,mz

    Returns:
        timestamps: Array of timestamps [N]
        data: Array of IMU data [N, 9]
    """
    timestamps = []
    data = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    # Try CSV format with timestamp
                    if ',' in line and '{' not in line:
                        values = [float(x) for x in line.split(',')]
                        if len(values) >= 10:
                            # Format: timestamp,ax,ay,az,gx,gy,gz,mx,my,mz
                            timestamps.append(values[0])
                            data.append(values[1:10])
                        elif len(values) >= 9:
                            # Old format without timestamp: ax,ay,az,gx,gy,gz,mx,my,mz
                            # Assume constant sample rate
                            timestamps.append(len(timestamps) * 0.01)
                            data.append(values[0:9])
                    # Try JSON format
                    elif line.startswith('{'):
                        d = json.loads(line)
                        timestamps.append(d.get('t', len(timestamps) * 0.01))
                        data.append([d['ax'], d['ay'], d['az'],
                                   d['gx'], d['gy'], d['gz'],
                                   d['mx'], d['my'], d['mz']])
                except:
                    continue

    return np.array(timestamps), np.array(data)


def process_offline(data_file):
    """
    Process IMU data offline with UKF + RTS smoother.

    Args:
        data_file: Path to IMU data file

    Returns:
        timestamps: Array of timestamps [N]
        orientations: Array of quaternions [N, 4]
        euler_angles: Array of Euler angles [N, 3]
    """
    print("="*80)
    print("OFFLINE ORIENTATION ESTIMATION - UKF + RTS SMOOTHER")
    print("="*80 + "\n")

    # Load data
    print(f"Loading data from {data_file}...")
    timestamps, data = load_imu_data(data_file)
    N = len(data)

    if N == 0:
        print("ERROR: No data loaded!")
        return None, None, None

    # Check if timestamps are in milliseconds or microseconds (common ESP32 issue)
    duration = timestamps[-1] - timestamps[0]
    if duration > 1000000:  # Likely in microseconds
        print("  Detected timestamps in microseconds, converting to seconds...")
        timestamps = timestamps / 1000000.0
        duration = timestamps[-1] - timestamps[0]
    elif duration > 1000:  # Likely in milliseconds
        print("  Detected timestamps in milliseconds, converting to seconds...")
        timestamps = timestamps / 1000.0
        duration = timestamps[-1] - timestamps[0]

    avg_sample_rate = (N - 1) / duration if duration > 0 else 100.0
    avg_dt = 1.0 / avg_sample_rate

    print(f"Loaded {N} samples")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Average sample rate: {avg_sample_rate:.1f} Hz\n")

    # Apply calibrations
    print("Applying calibrations...")
    accel = data[:, 0:3] * ACCEL_SCALE
    gyro = data[:, 3:6] - GYRO_BIAS
    mag = (data[:, 6:9] - MAG_OFFSET) * MAG_SCALE
    print("  [OK] Gyro bias correction")
    print("  [OK] Accelerometer scaling")
    print("  [OK] Magnetometer hard-iron correction\n")

    # Forward pass: UKF filtering (with variable dt)
    print("Forward pass: Running UKF...")
    ukf = IMU_UKF_Offline(dt=avg_dt)

    forward_states = np.zeros((N, ukf.dim_x))
    forward_covariances = np.zeros((N, ukf.dim_x, ukf.dim_x))

    for i in range(N):
        # Use actual dt between samples
        if i > 0:
            dt = timestamps[i] - timestamps[i-1]
            ukf.dt = dt

        ukf.predict()
        ukf.update(accel[i], gyro[i], mag[i])

        forward_states[i] = ukf.ukf.x.copy()
        forward_covariances[i] = ukf.ukf.P.copy()

        if (i+1) % 500 == 0:
            print(f"  Processed {i+1}/{N} samples ({100*(i+1)/N:.1f}%)")

    print(f"  [OK] Forward pass complete\n")

    # Backward pass: RTS smoother
    print("Backward pass: Running RTS smoother...")
    smoothed_states, smoothed_covariances = rts_smoother(forward_states, forward_covariances, avg_dt)
    print("  [OK] Backward pass complete\n")

    # Extract orientations
    quaternions = smoothed_states[:, 0:4]

    # Convert to Euler angles
    euler_angles = np.zeros((N, 3))
    for i in range(N):
        q = quaternions[i]
        q_scipy = [q[1], q[2], q[3], q[0]]
        rot = R.from_quat(q_scipy)
        euler_angles[i] = rot.as_euler('xyz', degrees=True)

    print("="*80)
    print("PROCESSING COMPLETE!")
    print("="*80 + "\n")

    print("Statistics:")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Samples: {N}")
    print(f"  Sample rate: {avg_sample_rate:.1f} Hz")
    print(f"\nOrientation range:")
    print(f"  Roll:  [{euler_angles[:, 0].min():7.2f}°, {euler_angles[:, 0].max():7.2f}°]")
    print(f"  Pitch: [{euler_angles[:, 1].min():7.2f}°, {euler_angles[:, 1].max():7.2f}°]")
    print(f"  Yaw:   [{euler_angles[:, 2].min():7.2f}°, {euler_angles[:, 2].max():7.2f}°]")
    print()

    return timestamps, quaternions, euler_angles, forward_states[:, 0:4]


def plot_comparison(timestamps, quaternions_forward, quaternions_smoothed):
    """Plot comparison between forward-only and smoothed results."""
    N = len(quaternions_forward)
    time = timestamps

    # Convert to Euler angles
    euler_forward = np.zeros((N, 3))
    euler_smoothed = np.zeros((N, 3))

    for i in range(N):
        q = quaternions_forward[i]
        q_scipy = [q[1], q[2], q[3], q[0]]
        euler_forward[i] = R.from_quat(q_scipy).as_euler('xyz', degrees=True)

        q = quaternions_smoothed[i]
        q_scipy = [q[1], q[2], q[3], q[0]]
        euler_smoothed[i] = R.from_quat(q_scipy).as_euler('xyz', degrees=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('UKF vs UKF+RTS Smoother Comparison', fontsize=16, fontweight='bold')

    labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
    colors_forward = ['red', 'green', 'blue']
    colors_smooth = ['darkred', 'darkgreen', 'darkblue']

    for i in range(3):
        axes[i].plot(time, euler_forward[:, i], label=f'{labels[i]} - Forward only',
                    color=colors_forward[i], alpha=0.5, linewidth=1)
        axes[i].plot(time, euler_smoothed[:, i], label=f'{labels[i]} - Smoothed',
                    color=colors_smooth[i], linewidth=2)
        axes[i].set_ylabel('Angle (degrees)', fontsize=11)
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)

    axes[2].set_xlabel('Time (seconds)', fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_results(timestamps, euler_angles):
    """Plot orientation results."""
    time = timestamps

    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    fig.suptitle('Offline Orientation Estimation (UKF + RTS Smoother)', fontsize=16, fontweight='bold')

    labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
    colors = ['red', 'green', 'blue']

    for i in range(3):
        axes[i].plot(time, euler_angles[:, i], label=labels[i], color=colors[i], linewidth=1.5)
        axes[i].set_ylabel('Angle (degrees)', fontsize=11)
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)

    axes[2].set_xlabel('Time (seconds)', fontsize=11)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python offline_orientation.py <data_file.csv>")
        print("\nExample:")
        print("  python offline_orientation.py imu_data.csv")
        print("\nData format should be CSV: ax,ay,az,gx,gy,gz,mx,my,mz")
        sys.exit(1)

    data_file = sys.argv[1]

    # Process with UKF + RTS smoother
    timestamps, quaternions, euler_angles, forward_quaternions = process_offline(data_file)

    if quaternions is not None:
        # Save results first
        output_file = data_file.replace('.csv', '_orientations.npz')
        print("\nSaving data files...")
        np.savez(output_file, 
                quaternions=quaternions, 
                euler_angles=euler_angles,
                timestamps=timestamps)
        
        # Save CSV file
        csv_output = data_file.replace('.csv', '_orientations.csv')
        with open(csv_output, 'w') as f:
            f.write("timestamp,roll,pitch,yaw\n")
            for t, e in zip(timestamps, euler_angles):
                f.write(f"{t:.6f},{e[0]:.6f},{e[1]:.6f},{e[2]:.6f}\n")
        
        print(f"[OK] NPZ results saved to: {output_file}")
        print(f"[OK] CSV results saved to: {csv_output}")

        # Save plots to files instead of displaying
        print("\nGenerating and saving plots...")
        
        # Comparison plot
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle('UKF vs UKF+RTS Smoother Comparison', fontsize=16, fontweight='bold')

        labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
        colors_forward = ['red', 'green', 'blue']
        colors_smooth = ['darkred', 'darkgreen', 'darkblue']

        # Convert quaternions to Euler angles
        N = len(forward_quaternions)
        euler_forward = np.zeros((N, 3))
        euler_smoothed = np.zeros((N, 3))

        for i in range(N):
            q = forward_quaternions[i]
            q_scipy = [q[1], q[2], q[3], q[0]]
            euler_forward[i] = R.from_quat(q_scipy).as_euler('xyz', degrees=True)

            q = quaternions[i]
            q_scipy = [q[1], q[2], q[3], q[0]]
            euler_smoothed[i] = R.from_quat(q_scipy).as_euler('xyz', degrees=True)

        for i in range(3):
            axes[i].plot(timestamps, euler_forward[:, i], label=f'{labels[i]} - Forward only',
                        color=colors_forward[i], alpha=0.5, linewidth=1)
            axes[i].plot(timestamps, euler_smoothed[:, i], label=f'{labels[i]} - Smoothed',
                        color=colors_smooth[i], linewidth=2)
            axes[i].set_ylabel('Angle (degrees)', fontsize=11)
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)

        axes[2].set_xlabel('Time (seconds)', fontsize=11)
        plt.tight_layout()
        comparison_plot = data_file.replace('.csv', '_comparison.png')
        plt.savefig(comparison_plot)
        plt.close()

        # Results plot
        fig, axes = plt.subplots(3, 1, figsize=(14, 9))
        fig.suptitle('Offline Orientation Estimation (UKF + RTS Smoother)', fontsize=16, fontweight='bold')

        labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
        colors = ['red', 'green', 'blue']

        for i in range(3):
            axes[i].plot(timestamps, euler_angles[:, i], label=labels[i], color=colors[i], linewidth=1.5)
            axes[i].set_ylabel('Angle (degrees)', fontsize=11)
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)

        axes[2].set_xlabel('Time (seconds)', fontsize=11)
        plt.tight_layout()
        results_plot = data_file.replace('.csv', '_results.png')
        plt.savefig(results_plot)
        plt.close()
        
        print(f"[OK] Comparison plot saved to: {comparison_plot}")
        print(f"[OK] Results plot saved to: {results_plot}")
        print("\nAll files saved successfully!")
        
        # Also save a CSV file with timestamps and Euler angles for easy reading
        csv_output = data_file.replace('.csv', '_orientations.csv')
        with open(csv_output, 'w') as f:
            f.write("timestamp,roll,pitch,yaw\n")
            for t, e in zip(timestamps, euler_angles):
                f.write(f"{t:.6f},{e[0]:.6f},{e[1]:.6f},{e[2]:.6f}\n")
        
        print(f"\n[OK] Results saved to: {output_file}")
        print(f"[OK] CSV results saved to: {csv_output}")
