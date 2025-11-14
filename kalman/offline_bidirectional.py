"""
Bi-directional UKF processing for maximum accuracy.
Runs UKF forward and backward, then averages results to eliminate initialization artifacts.
"""

import numpy as np
from offline_orientation import (
    IMU_UKF_Offline, load_imu_data, compute_initial_orientation,
    GYRO_BIAS, ACCEL_SCALE, MAG_OFFSET, MAG_SCALE
)
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import sys


def quaternion_slerp(q1, q2, t=0.5):
    """Spherical linear interpolation between two quaternions."""
    # Ensure shortest path
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot

    # Clamp dot product
    dot = np.clip(dot, -1, 1)

    theta = np.arccos(dot)

    if abs(theta) < 1e-6:
        return q1  # Quaternions are nearly identical

    sin_theta = np.sin(theta)
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta

    return w1 * q1 + w2 * q2


def process_bidirectional(data_file):
    """
    Process IMU data with bi-directional UKF for maximum accuracy.

    Runs UKF forward and backward, then averages results.
    """
    print("="*80)
    print("BI-DIRECTIONAL UKF PROCESSING")
    print("Maximum accuracy post-processing")
    print("="*80 + "\n")

    # Load data
    print(f"Loading data from {data_file}...")
    timestamps, data = load_imu_data(data_file)
    N = len(data)

    if N == 0:
        print("ERROR: No data loaded!")
        return None, None, None

    # Convert timestamps
    duration = timestamps[-1] - timestamps[0]
    if duration > 1000000:
        timestamps = timestamps / 1000000.0
        duration = timestamps[-1] - timestamps[0]
    elif duration > 1000:
        timestamps = timestamps / 1000.0
        duration = timestamps[-1] - timestamps[0]

    avg_sample_rate = (N - 1) / duration if duration > 0 else 100.0
    avg_dt = 1.0 / avg_sample_rate

    print(f"Loaded {N} samples")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Average sample rate: {avg_sample_rate:.1f} Hz\n")

    # Apply calibrations
    accel = data[:, 0:3] * ACCEL_SCALE
    gyro = data[:, 3:6] - GYRO_BIAS
    mag = (data[:, 6:9] - MAG_OFFSET) * MAG_SCALE

    # ========================================================================
    # FORWARD PASS
    # ========================================================================
    print("="*80)
    print("FORWARD PASS (Start → End)")
    print("="*80 + "\n")

    # Compute initial orientation from beginning
    skip_samples = int(0.5 * avg_sample_rate)
    init_samples = min(50, N - skip_samples)

    if skip_samples + init_samples < N:
        accel_init = accel[skip_samples:skip_samples+init_samples]
        mag_init = mag[skip_samples:skip_samples+init_samples]
        initial_q_fwd, mag_ref_fwd = compute_initial_orientation(accel_init, mag_init)
        q_scipy = [initial_q_fwd[1], initial_q_fwd[2], initial_q_fwd[3], initial_q_fwd[0]]
        init_euler = R.from_quat(q_scipy).as_euler('xyz', degrees=True)
        print(f"Initial orientation: Roll={init_euler[0]:.1f}°, Pitch={init_euler[1]:.1f}°, Yaw={init_euler[2]:.1f}°\n")
    else:
        initial_q_fwd = None
        mag_ref_fwd = None

    ukf_fwd = IMU_UKF_Offline(dt=avg_dt, initial_orientation=initial_q_fwd)
    if mag_ref_fwd is not None:
        ukf_fwd.mag_reference = mag_ref_fwd

    quaternions_fwd = np.zeros((N, 4))

    for i in range(N):
        if i > 0:
            dt = timestamps[i] - timestamps[i-1]
            ukf_fwd.dt = dt

        ukf_fwd.predict()
        ukf_fwd.update(accel[i], gyro[i], mag[i])
        quaternions_fwd[i] = ukf_fwd.get_quaternion()

        if (i+1) % 500 == 0:
            print(f"  Processed {i+1}/{N} samples ({100*(i+1)/N:.1f}%)")

    print("  [OK] Forward pass complete\n")

    # ========================================================================
    # BACKWARD PASS
    # ========================================================================
    print("="*80)
    print("BACKWARD PASS (End → Start)")
    print("="*80 + "\n")

    # Compute initial orientation from end
    if skip_samples + init_samples < N:
        accel_init = accel[-(skip_samples+init_samples):-skip_samples]
        mag_init = mag[-(skip_samples+init_samples):-skip_samples]
        initial_q_bwd, mag_ref_bwd = compute_initial_orientation(accel_init, mag_init)
        q_scipy = [initial_q_bwd[1], initial_q_bwd[2], initial_q_bwd[3], initial_q_bwd[0]]
        init_euler = R.from_quat(q_scipy).as_euler('xyz', degrees=True)
        print(f"Initial orientation: Roll={init_euler[0]:.1f}°, Pitch={init_euler[1]:.1f}°, Yaw={init_euler[2]:.1f}°\n")
    else:
        initial_q_bwd = None
        mag_ref_bwd = None

    ukf_bwd = IMU_UKF_Offline(dt=avg_dt, initial_orientation=initial_q_bwd)
    if mag_ref_bwd is not None:
        ukf_bwd.mag_reference = mag_ref_bwd

    quaternions_bwd = np.zeros((N, 4))

    for i in range(N-1, -1, -1):
        if i < N-1:
            dt = timestamps[i+1] - timestamps[i]
            ukf_bwd.dt = dt

        # Negate gyro for backward integration
        gyro_neg = -gyro[i]

        ukf_bwd.predict()
        ukf_bwd.update(accel[i], gyro_neg, mag[i])
        quaternions_bwd[i] = ukf_bwd.get_quaternion()

        if (N-i) % 500 == 0:
            print(f"  Processed {N-i}/{N} samples ({100*(N-i)/N:.1f}%)")

    print("  [OK] Backward pass complete\n")

    # ========================================================================
    # AVERAGE RESULTS
    # ========================================================================
    print("="*80)
    print("AVERAGING RESULTS")
    print("="*80 + "\n")

    quaternions_avg = np.zeros((N, 4))
    for i in range(N):
        # Use SLERP to average quaternions
        quaternions_avg[i] = quaternion_slerp(quaternions_fwd[i], quaternions_bwd[i], t=0.5)
        # Normalize
        quaternions_avg[i] = quaternions_avg[i] / np.linalg.norm(quaternions_avg[i])

    print("  [OK] Averaging complete\n")

    # Convert to Euler angles
    euler_angles = np.zeros((N, 3))
    for i in range(N):
        q = quaternions_avg[i]
        q_scipy = [q[1], q[2], q[3], q[0]]
        euler_angles[i] = R.from_quat(q_scipy).as_euler('xyz', degrees=True)

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

    return timestamps, quaternions_avg, euler_angles, quaternions_fwd, quaternions_bwd


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python offline_bidirectional.py <data_file.csv>")
        sys.exit(1)

    data_file = sys.argv[1]
    result = process_bidirectional(data_file)

    if result[0] is not None:
        timestamps, quaternions_avg, euler_angles, quaternions_fwd, quaternions_bwd = result

        # Save results
        output_file = data_file.replace('.csv', '_orientations_bidirectional.npz')
        np.savez(output_file,
                quaternions=quaternions_avg,
                euler_angles=euler_angles,
                timestamps=timestamps,
                quaternions_forward=quaternions_fwd,
                quaternions_backward=quaternions_bwd)

        csv_output = data_file.replace('.csv', '_orientations_bidirectional.csv')
        with open(csv_output, 'w') as f:
            f.write("timestamp,roll,pitch,yaw\n")
            for t, e in zip(timestamps, euler_angles):
                f.write(f"{t:.6f},{e[0]:.6f},{e[1]:.6f},{e[2]:.6f}\n")

        print(f"\n[OK] Results saved to: {output_file}")
        print(f"[OK] CSV results saved to: {csv_output}")

        # Plot comparison
        euler_fwd = np.zeros((len(timestamps), 3))
        euler_bwd = np.zeros((len(timestamps), 3))
        for i in range(len(timestamps)):
            q = quaternions_fwd[i]
            q_scipy = [q[1], q[2], q[3], q[0]]
            euler_fwd[i] = R.from_quat(q_scipy).as_euler('xyz', degrees=True)

            q = quaternions_bwd[i]
            q_scipy = [q[1], q[2], q[3], q[0]]
            euler_bwd[i] = R.from_quat(q_scipy).as_euler('xyz', degrees=True)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle('Bi-Directional UKF Comparison', fontsize=16, fontweight='bold')

        labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
        colors = ['red', 'green', 'blue']

        for i in range(3):
            axes[i].plot(timestamps, euler_fwd[:, i], label=f'{labels[i]} - Forward',
                        color=colors[i], alpha=0.4, linewidth=1)
            axes[i].plot(timestamps, euler_bwd[:, i], label=f'{labels[i]} - Backward',
                        color=colors[i], alpha=0.4, linewidth=1, linestyle='--')
            axes[i].plot(timestamps, euler_angles[:, i], label=f'{labels[i]} - Average',
                        color='black', linewidth=2)
            axes[i].set_ylabel('Angle (degrees)', fontsize=11)
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)

        axes[2].set_xlabel('Time (seconds)', fontsize=11)
        plt.tight_layout()

        plot_file = data_file.replace('.csv', '_bidirectional_comparison.png')
        plt.savefig(plot_file)
        print(f"[OK] Comparison plot saved to: {plot_file}")
        plt.close()
