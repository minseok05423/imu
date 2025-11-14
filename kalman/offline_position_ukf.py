"""
Bi-directional UKF for GPS + IMU + Barometer sensor fusion.
Estimates high-accuracy position by fusing:
- IMU: acceleration + gyroscope (high rate, relative)
- GPS: position (low rate, absolute)
- Barometer: altitude (medium rate, absolute)
- Magnetometer: heading (absolute)

State vector: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, bax, bay, baz, bgx, bgy, bgz]
- Position (3): x, y, z in local ENU frame
- Velocity (3): vx, vy, vz
- Orientation (4): quaternion
- Accel bias (3): accelerometer bias
- Gyro bias (3): gyroscope bias
Total: 16 states
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


# ============================================================================
# CALIBRATION PARAMETERS (update these with your calibration)
# ============================================================================
GYRO_BIAS = np.array([-0.093434, 0.038554, -0.004698])
ACCEL_SCALE = 1.0
MAG_OFFSET = np.array([0.0318, -0.2640, -0.2652])
MAG_SCALE = np.array([1.2605, 0.9314, 0.8826])

# Reference point for local coordinates (will be set from first GPS fix)
LAT_REF = None
LON_REF = None
ALT_REF = None

# Earth radius for lat/lon to meters conversion
EARTH_RADIUS = 6371000.0  # meters


def latlon_to_meters(lat, lon, lat_ref, lon_ref):
    """Convert lat/lon to local ENU coordinates in meters."""
    dlat = lat - lat_ref
    dlon = lon - lon_ref

    x = dlon * EARTH_RADIUS * np.cos(np.radians(lat_ref)) * np.pi / 180.0
    y = dlat * EARTH_RADIUS * np.pi / 180.0

    return x, y


class PositionUKF:
    """
    UKF for position estimation with GPS + IMU + Barometer fusion.

    State: [px, py, pz, vx, vy, vz, qw, qx, qy, qz, bax, bay, baz, bgx, bgy, bgz]
    """

    def __init__(self, dt=0.01, initial_state=None):
        self.dt = dt
        self.dim_x = 16  # State dimension

        # Sigma points
        points = MerweScaledSigmaPoints(n=self.dim_x, alpha=0.1, beta=2., kappa=0)

        self.ukf = UnscentedKalmanFilter(
            dim_x=self.dim_x,
            dim_z=10,  # Will vary based on available measurements
            dt=dt,
            fx=self.state_transition,
            hx=self.measurement_function_gps_baro,
            points=points
        )

        # Initial state
        if initial_state is not None:
            self.ukf.x = initial_state.copy()
        else:
            # Default: zero position/velocity, identity quaternion, zero biases
            self.ukf.x = np.zeros(16)
            self.ukf.x[6] = 1.0  # qw = 1 (identity quaternion)

        # Process noise Q (how much we trust the model)
        self.ukf.Q = np.eye(self.dim_x)
        self.ukf.Q[0:3, 0:3] *= 0.1      # Position: moderate (will be corrected by GPS)
        self.ukf.Q[3:6, 3:6] *= 0.5      # Velocity: moderate
        self.ukf.Q[6:10, 6:10] *= 0.0001 # Quaternion: very stable
        self.ukf.Q[10:13, 10:13] *= 0.001  # Accel bias: slow drift
        self.ukf.Q[13:16, 13:16] *= 0.00001  # Gyro bias: very slow drift

        # Initial covariance
        self.ukf.P = np.eye(self.dim_x)
        self.ukf.P[0:3, 0:3] *= 10.0    # Position: uncertain
        self.ukf.P[3:6, 3:6] *= 5.0     # Velocity: uncertain
        self.ukf.P[6:10, 6:10] *= 0.1   # Quaternion: somewhat certain
        self.ukf.P[10:13, 10:13] *= 0.1 # Accel bias
        self.ukf.P[13:16, 13:16] *= 0.01 # Gyro bias

        # Current measurements (for measurement function)
        self.current_accel = np.zeros(3)
        self.current_gyro = np.zeros(3)

    def state_transition(self, x, dt):
        """
        State transition function.

        Integrates:
        - Position using velocity
        - Velocity using acceleration (rotated to world frame)
        - Orientation using gyroscope
        - Biases remain constant (random walk)
        """
        # Extract state
        pos = x[0:3]
        vel = x[3:6]
        q = x[6:10]
        accel_bias = x[10:13]
        gyro_bias = x[13:16]

        # Normalize quaternion
        q = q / np.linalg.norm(q)

        # Corrected IMU measurements
        accel_corrected = self.current_accel - accel_bias
        gyro_corrected = self.current_gyro - gyro_bias

        # Rotate acceleration to world frame and subtract gravity
        accel_world = self.quaternion_rotate(q, accel_corrected)
        accel_world[2] -= 9.81  # Remove gravity (NED convention: gravity is +z)

        # Update position and velocity
        pos_new = pos + vel * dt + 0.5 * accel_world * dt**2
        vel_new = vel + accel_world * dt

        # Update orientation (integrate gyroscope)
        w_norm = np.linalg.norm(gyro_corrected)
        if w_norm > 1e-8:
            angle = w_norm * dt
            axis = gyro_corrected / w_norm
            dq = np.array([
                np.cos(angle / 2),
                axis[0] * np.sin(angle / 2),
                axis[1] * np.sin(angle / 2),
                axis[2] * np.sin(angle / 2)
            ])
            q_new = self.quaternion_multiply(q, dq)
            q_new = q_new / np.linalg.norm(q_new)
        else:
            q_new = q

        # Biases remain constant (random walk handled by process noise)
        accel_bias_new = accel_bias
        gyro_bias_new = gyro_bias

        return np.concatenate([pos_new, vel_new, q_new, accel_bias_new, gyro_bias_new])

    def measurement_function_gps_baro(self, x):
        """
        Measurement function for GPS + Barometer.

        Returns: [px, py, pz, vx, vy, vz, heading, pressure_altitude, ax, ay]
        """
        pos = x[0:3]
        vel = x[3:6]
        q = x[6:10]

        # GPS position (x, y, z)
        gps_pos = pos

        # GPS velocity (vx, vy, vz)
        gps_vel = vel

        # Heading from quaternion
        rot = R.from_quat([q[1], q[2], q[3], q[0]])
        euler = rot.as_euler('xyz', degrees=False)
        heading = euler[2]  # yaw

        # Barometer altitude
        baro_alt = pos[2]

        # GPS horizontal speed (for validation)
        h_speed = np.sqrt(vel[0]**2 + vel[1]**2)

        return np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2],
                        heading, baro_alt, h_speed, 0])

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

    def quaternion_rotate(self, q, v):
        """Rotate vector v by quaternion q."""
        qv = np.array([0., v[0], v[1], v[2]])
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        result = self.quaternion_multiply(self.quaternion_multiply(q, qv), q_conj)
        return result[1:]

    def predict(self, accel, gyro):
        """Prediction step with IMU measurements."""
        self.current_accel = accel
        self.current_gyro = gyro
        self.ukf.predict()
        # Normalize quaternion
        self.ukf.x[6:10] = self.ukf.x[6:10] / np.linalg.norm(self.ukf.x[6:10])

    def update_gps(self, gps_x, gps_y, gps_alt, gps_vx, gps_vy):
        """Update with GPS measurement."""
        # Measurement: [x, y, z, vx, vy]
        z = np.array([gps_x, gps_y, gps_alt, gps_vx, gps_vy])

        # Measurement noise
        R_gps = np.diag([5.0, 5.0, 10.0, 2.0, 2.0])  # GPS position/velocity noise

        # Simplified measurement function for GPS only
        def h_gps(x):
            return np.array([x[0], x[1], x[2], x[3], x[4]])

        # Temporarily change measurement function
        original_hx = self.ukf.hx
        self.ukf.hx = h_gps
        self.ukf.R = R_gps

        self.ukf.update(z, hx=h_gps, R=R_gps)

        # Restore
        self.ukf.hx = original_hx

        # Normalize quaternion after update
        self.ukf.x[6:10] = self.ukf.x[6:10] / np.linalg.norm(self.ukf.x[6:10])

    def update_baro(self, altitude):
        """Update with barometer measurement."""
        z = np.array([altitude])
        R_baro = np.array([[2.0]])  # Barometer noise

        def h_baro(x):
            return np.array([x[2]])  # Just altitude

        self.ukf.update(z, hx=h_baro, R=R_baro)
        self.ukf.x[6:10] = self.ukf.x[6:10] / np.linalg.norm(self.ukf.x[6:10])

    def get_position(self):
        """Get current position estimate [x, y, z]."""
        return self.ukf.x[0:3]

    def get_velocity(self):
        """Get current velocity estimate [vx, vy, vz]."""
        return self.ukf.x[3:6]

    def get_orientation(self):
        """Get current orientation as quaternion [qw, qx, qy, qz]."""
        return self.ukf.x[6:10]


def load_sensor_data(filename):
    """
    Load sensor data from CSV.

    Format: timestamp_us,ax,ay,az,gx,gy,gz,mx,my,mz,pressure_pa,temperature_c,
            baro_altitude_m,lat,lon,speed_kmh,altitude_m,gps_time,satellites,hdop,heading_deg
    """
    timestamps = []
    imu_data = []
    baro_data = []
    gps_data = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line.startswith('timestamp') or not line:
                continue

            try:
                values = [float(x) if x else np.nan for x in line.split(',')]
                if len(values) >= 21:
                    timestamps.append(values[0])
                    imu_data.append(values[1:10])  # ax,ay,az,gx,gy,gz,mx,my,mz
                    baro_data.append([values[10], values[11], values[12]])  # pressure, temp, baro_alt
                    gps_data.append(values[13:21])  # lat,lon,speed,alt,time,sats,hdop,heading
            except:
                continue

    return (np.array(timestamps), np.array(imu_data),
            np.array(baro_data), np.array(gps_data))


def process_bidirectional_position(data_file):
    """
    Process sensor data with bi-directional UKF for position estimation.
    """
    print("="*80)
    print("BI-DIRECTIONAL POSITION UKF")
    print("GPS + IMU + Barometer Sensor Fusion")
    print("="*80 + "\n")

    # Load data
    print(f"Loading data from {data_file}...")
    timestamps, imu_data, baro_data, gps_data = load_sensor_data(data_file)
    N = len(timestamps)

    if N == 0:
        print("ERROR: No data loaded!")
        return None

    print(f"Loaded {N} samples\n")

    # Convert timestamps to seconds
    timestamps = timestamps / 1e6
    duration = timestamps[-1] - timestamps[0]
    avg_dt = np.mean(np.diff(timestamps))

    print(f"Duration: {duration:.2f} seconds")
    print(f"Average sample rate: {1/avg_dt:.1f} Hz\n")

    # Apply IMU calibrations
    accel = imu_data[:, 0:3] * ACCEL_SCALE
    gyro = (imu_data[:, 3:6] - GYRO_BIAS)
    mag = (imu_data[:, 6:9] - MAG_OFFSET) * MAG_SCALE

    # Set GPS reference from first valid fix
    global LAT_REF, LON_REF, ALT_REF
    for i in range(N):
        if not np.isnan(gps_data[i, 0]) and not np.isnan(gps_data[i, 1]):
            LAT_REF = gps_data[i, 0]
            LON_REF = gps_data[i, 1]
            ALT_REF = gps_data[i, 3]
            print(f"GPS reference set:")
            print(f"  Lat: {LAT_REF:.6f}°")
            print(f"  Lon: {LON_REF:.6f}°")
            print(f"  Alt: {ALT_REF:.2f} m\n")
            break

    if LAT_REF is None:
        print("ERROR: No valid GPS fixes found!")
        return None

    # Convert GPS to local coordinates
    gps_local = np.zeros((N, 3))
    gps_valid = np.zeros(N, dtype=bool)

    for i in range(N):
        if not np.isnan(gps_data[i, 0]) and not np.isnan(gps_data[i, 1]):
            x, y = latlon_to_meters(gps_data[i, 0], gps_data[i, 1], LAT_REF, LON_REF)
            z = gps_data[i, 3] - ALT_REF if not np.isnan(gps_data[i, 3]) else 0
            gps_local[i] = [x, y, z]
            gps_valid[i] = True

    print(f"Valid GPS fixes: {np.sum(gps_valid)}/{N} ({100*np.sum(gps_valid)/N:.1f}%)\n")

    # ========================================================================
    # FORWARD PASS
    # ========================================================================
    print("="*80)
    print("FORWARD PASS")
    print("="*80 + "\n")

    # Initialize with first GPS fix
    first_gps_idx = np.where(gps_valid)[0][0]
    initial_state = np.zeros(16)
    initial_state[0:3] = gps_local[first_gps_idx]  # Initial position from GPS
    initial_state[6] = 1.0  # Identity quaternion

    ukf_fwd = PositionUKF(dt=avg_dt, initial_state=initial_state)

    positions_fwd = np.zeros((N, 3))
    velocities_fwd = np.zeros((N, 3))

    for i in range(N):
        # Predict with IMU
        ukf_fwd.predict(accel[i], gyro[i])

        # Update with GPS if available
        if gps_valid[i]:
            # GPS velocity from speed and heading
            speed_ms = gps_data[i, 2] / 3.6  # km/h to m/s
            if not np.isnan(gps_data[i, 7]):  # heading available
                heading_rad = np.radians(gps_data[i, 7])
                vx = speed_ms * np.cos(heading_rad)
                vy = speed_ms * np.sin(heading_rad)
            else:
                vx = vy = 0

            ukf_fwd.update_gps(gps_local[i, 0], gps_local[i, 1], gps_local[i, 2], vx, vy)

        # Update with barometer
        if not np.isnan(baro_data[i, 2]):
            ukf_fwd.update_baro(baro_data[i, 2] - ALT_REF)

        positions_fwd[i] = ukf_fwd.get_position()
        velocities_fwd[i] = ukf_fwd.get_velocity()

        if (i+1) % 500 == 0:
            print(f"  Processed {i+1}/{N} samples ({100*(i+1)/N:.1f}%)")

    print("  [OK] Forward pass complete\n")

    # ========================================================================
    # BACKWARD PASS
    # ========================================================================
    print("="*80)
    print("BACKWARD PASS")
    print("="*80 + "\n")

    # Initialize with last GPS fix
    last_gps_idx = np.where(gps_valid)[0][-1]
    initial_state_bwd = np.zeros(16)
    initial_state_bwd[0:3] = gps_local[last_gps_idx]
    initial_state_bwd[6] = 1.0

    ukf_bwd = PositionUKF(dt=avg_dt, initial_state=initial_state_bwd)

    positions_bwd = np.zeros((N, 3))
    velocities_bwd = np.zeros((N, 3))

    for i in range(N-1, -1, -1):
        # Predict with IMU (negate for backward)
        ukf_bwd.predict(-accel[i], -gyro[i])

        # Update with GPS
        if gps_valid[i]:
            speed_ms = gps_data[i, 2] / 3.6
            if not np.isnan(gps_data[i, 7]):
                heading_rad = np.radians(gps_data[i, 7])
                vx = speed_ms * np.cos(heading_rad)
                vy = speed_ms * np.sin(heading_rad)
            else:
                vx = vy = 0

            ukf_bwd.update_gps(gps_local[i, 0], gps_local[i, 1], gps_local[i, 2], -vx, -vy)

        # Update with barometer
        if not np.isnan(baro_data[i, 2]):
            ukf_bwd.update_baro(baro_data[i, 2] - ALT_REF)

        positions_bwd[i] = ukf_bwd.get_position()
        velocities_bwd[i] = -ukf_bwd.get_velocity()  # Negate back

        if (N-i) % 500 == 0:
            print(f"  Processed {N-i}/{N} samples ({100*(N-i)/N:.1f}%)")

    print("  [OK] Backward pass complete\n")

    # ========================================================================
    # AVERAGE RESULTS
    # ========================================================================
    print("="*80)
    print("AVERAGING RESULTS")
    print("="*80 + "\n")

    positions_avg = (positions_fwd + positions_bwd) / 2.0
    velocities_avg = (velocities_fwd + velocities_bwd) / 2.0

    print("  [OK] Averaging complete\n")

    print("="*80)
    print("PROCESSING COMPLETE!")
    print("="*80 + "\n")

    print("Position statistics (meters):")
    print(f"  X range: [{positions_avg[:, 0].min():.2f}, {positions_avg[:, 0].max():.2f}]")
    print(f"  Y range: [{positions_avg[:, 1].min():.2f}, {positions_avg[:, 1].max():.2f}]")
    print(f"  Z range: [{positions_avg[:, 2].min():.2f}, {positions_avg[:, 2].max():.2f}]")

    total_distance = np.sum(np.linalg.norm(np.diff(positions_avg, axis=0), axis=1))
    print(f"\nTotal distance traveled: {total_distance:.2f} m")

    return timestamps, positions_avg, velocities_avg, positions_fwd, positions_bwd, gps_local, gps_valid


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python offline_position_ukf.py <data_file.csv>")
        sys.exit(1)

    data_file = sys.argv[1]
    result = process_bidirectional_position(data_file)

    if result is not None:
        timestamps, pos_avg, vel_avg, pos_fwd, pos_bwd, gps_local, gps_valid = result

        # Save results
        output_file = data_file.replace('.csv', '_position_ukf.npz')
        np.savez(output_file,
                timestamps=timestamps,
                positions=pos_avg,
                velocities=vel_avg,
                positions_forward=pos_fwd,
                positions_backward=pos_bwd,
                gps_positions=gps_local,
                gps_valid=gps_valid)

        csv_output = data_file.replace('.csv', '_position_ukf.csv')
        with open(csv_output, 'w') as f:
            f.write("timestamp,x,y,z,vx,vy,vz\n")
            for t, p, v in zip(timestamps, pos_avg, vel_avg):
                f.write(f"{t:.6f},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},{v[0]:.6f},{v[1]:.6f},{v[2]:.6f}\n")

        print(f"\n[OK] Results saved to: {output_file}")
        print(f"[OK] CSV results saved to: {csv_output}")

        # Plot results
        fig = plt.figure(figsize=(16, 10))

        # 2D trajectory
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(gps_local[gps_valid, 0], gps_local[gps_valid, 1], 'r.',
                alpha=0.3, markersize=3, label='GPS raw')
        ax1.plot(pos_fwd[:, 0], pos_fwd[:, 1], 'b-', alpha=0.3, linewidth=1, label='Forward')
        ax1.plot(pos_bwd[:, 0], pos_bwd[:, 1], 'g-', alpha=0.3, linewidth=1, label='Backward')
        ax1.plot(pos_avg[:, 0], pos_avg[:, 1], 'k-', linewidth=2, label='Average (final)')
        ax1.set_xlabel('X (meters)')
        ax1.set_ylabel('Y (meters)')
        ax1.set_title('2D Trajectory (Top View)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # Altitude
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(timestamps, gps_local[:, 2], 'r.', alpha=0.3, markersize=3, label='GPS altitude')
        ax2.plot(timestamps, pos_fwd[:, 2], 'b-', alpha=0.3, linewidth=1, label='Forward')
        ax2.plot(timestamps, pos_bwd[:, 2], 'g-', alpha=0.3, linewidth=1, label='Backward')
        ax2.plot(timestamps, pos_avg[:, 2], 'k-', linewidth=2, label='Average (final)')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Altitude (meters)')
        ax2.set_title('Altitude Profile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Velocity
        ax3 = fig.add_subplot(2, 2, 3)
        speed = np.linalg.norm(vel_avg, axis=1)
        ax3.plot(timestamps, speed, 'k-', linewidth=1.5)
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Speed (m/s)')
        ax3.set_title('Speed Profile')
        ax3.grid(True, alpha=0.3)

        # 3D trajectory
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.plot(pos_avg[:, 0], pos_avg[:, 1], pos_avg[:, 2], 'k-', linewidth=2, label='UKF trajectory')
        ax4.scatter(gps_local[gps_valid, 0], gps_local[gps_valid, 1], gps_local[gps_valid, 2],
                   c='r', marker='.', s=10, alpha=0.3, label='GPS points')
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_zlabel('Z (m)')
        ax4.set_title('3D Trajectory')
        ax4.legend()

        plt.tight_layout()
        plot_file = data_file.replace('.csv', '_position_trajectory.png')
        plt.savefig(plot_file, dpi=150)
        print(f"[OK] Trajectory plot saved to: {plot_file}")
        plt.close()
