import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from scipy.spatial.transform import Rotation as R


class IMU_UKF_6DOF:
    """
    Unscented Kalman Filter for 6-DOF IMU (MPU6050: accel + gyro only).
    NO MAGNETOMETER - more stable, no drift from bad mag data.

    State vector (10 elements):
    - Orientation quaternion (4): [qw, qx, qy, qz]
    - Angular velocity (3): [wx, wy, wz] in rad/s
    - Gyro bias (3): [bwx, bwy, bwz] in rad/s

    Measurements (6 elements):
    - Accelerometer (3): [ax, ay, az] in m/s²
    - Gyroscope (3): [wx, wy, wz] in rad/s
    """

    def __init__(self, dt=0.01):
        """
        Initialize the UKF for 6-DOF IMU (accel + gyro only).

        Args:
            dt: Time step between measurements (seconds)
        """
        self.dt = dt
        self.dim_x = 10  # State dimension
        self.dim_z = 6   # Measurement dimension (no magnetometer)

        # Create sigma points
        points = MerweScaledSigmaPoints(n=self.dim_x, alpha=0.1, beta=2., kappa=0)

        # Initialize UKF
        self.ukf = UKF(dim_x=self.dim_x, dim_z=self.dim_z, dt=dt,
                       fx=self.state_transition,
                       hx=self.measurement_function,
                       points=points)

        # Initial state: identity quaternion, zero velocity/bias
        self.ukf.x = np.array([1., 0., 0., 0.,  # quaternion [w, x, y, z]
                               0., 0., 0.,       # angular velocity
                               0., 0., 0.])      # gyro bias

        # Initial covariance
        self.ukf.P = np.eye(self.dim_x) * 0.1
        self.ukf.P[0:4, 0:4] *= 0.01  # Low uncertainty in initial orientation

        # Process noise - tuned for MPU6050
        self.ukf.Q = np.eye(self.dim_x) * 0.001
        self.ukf.Q[0:4, 0:4] *= 0.0001   # Quaternion (very stable)
        self.ukf.Q[4:7, 4:7] *= 0.01     # Angular velocity
        self.ukf.Q[7:10, 7:10] *= 0.00001  # Gyro bias (very slow change)

        # Measurement noise - tuned for MPU6050
        self.ukf.R = np.eye(self.dim_z)
        self.ukf.R[0:3, 0:3] *= 0.05     # Accelerometer noise
        self.ukf.R[3:6, 3:6] *= 0.01     # Gyroscope noise

    def normalize_quaternion(self, q):
        """Normalize a quaternion."""
        norm = np.linalg.norm(q)
        if norm > 0:
            return q / norm
        return q

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

    def state_transition(self, x, dt):
        """
        State transition function (process model).
        """
        # Extract state components
        q = x[0:4]         # quaternion
        w = x[4:7]         # angular velocity
        bias = x[7:10]     # gyro bias

        # Correct angular velocity with bias
        w_corrected = w - bias

        # Update quaternion using angular velocity
        w_norm = np.linalg.norm(w_corrected)
        if w_norm > 0:
            angle = w_norm * dt
            axis = w_corrected / w_norm
            dq = np.array([
                np.cos(angle/2),
                axis[0] * np.sin(angle/2),
                axis[1] * np.sin(angle/2),
                axis[2] * np.sin(angle/2)
            ])
            q_new = self.quaternion_multiply(q, dq)
        else:
            q_new = q

        # Normalize quaternion
        q_new = self.normalize_quaternion(q_new)

        # Angular velocity and bias remain similar
        new_state = np.hstack([q_new, w, bias])
        return new_state

    def measurement_function(self, x):
        """
        Measurement function for 6-DOF (accel + gyro only).
        """
        q = x[0:4]         # quaternion
        w = x[4:7]         # angular velocity

        # Gyroscope measurement = angular velocity
        gyro_meas = w.copy()

        # Accelerometer measures gravity in body frame
        gravity_world = np.array([0., 0., 9.81])

        # Convert quaternion to rotation matrix
        qw, qx, qy, qz = q
        rot_mat = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
        ])

        # Gravity in body frame (what accelerometer should read when stationary)
        accel_meas = rot_mat.T @ gravity_world

        # Combine measurements
        z = np.hstack([accel_meas, gyro_meas])
        return z

    def update(self, accel, gyro):
        """
        Update the filter with new sensor measurements.

        Args:
            accel: Accelerometer reading [ax, ay, az] in m/s²
            gyro: Gyroscope reading [wx, wy, wz] in rad/s
        """
        # Combine measurements (no magnetometer)
        z = np.hstack([accel, gyro])

        # Predict and update
        self.ukf.predict()
        self.ukf.update(z)

        # Normalize quaternion after update
        self.ukf.x[0:4] = self.normalize_quaternion(self.ukf.x[0:4])

    def get_orientation_quaternion(self):
        """Get current orientation as quaternion [w, x, y, z]."""
        return self.ukf.x[0:4]

    def get_orientation_euler(self):
        """Get current orientation as Euler angles (roll, pitch, yaw) in degrees."""
        q = self.ukf.x[0:4]
        # Convert to scipy Rotation format [x, y, z, w]
        r = R.from_quat([q[1], q[2], q[3], q[0]])
        euler = r.as_euler('xyz', degrees=True)
        # Note: Yaw will drift without magnetometer, but roll/pitch are stable
        return euler

    def get_angular_velocity(self):
        """Get current angular velocity [wx, wy, wz] in rad/s."""
        return self.ukf.x[4:7]

    def get_gyro_bias(self):
        """Get estimated gyroscope bias [bx, by, bz] in rad/s."""
        return self.ukf.x[7:10]


# Example usage
if __name__ == "__main__":
    # Create UKF with 10ms sampling rate
    imu_filter = IMU_UKF_6DOF(dt=0.01)

    # Simulate some IMU data
    for i in range(100):
        # Simulated sensor readings (stationary)
        accel = np.array([0.0, 0.0, 9.81]) + np.random.randn(3) * 0.1
        gyro = np.array([0.0, 0.0, 0.0]) + np.random.randn(3) * 0.01

        # Update filter
        imu_filter.update(accel, gyro)

        # Get results every 10 iterations
        if i % 10 == 0:
            euler = imu_filter.get_orientation_euler()
            print(f"Iteration {i}: Roll={euler[0]:.2f}°, Pitch={euler[1]:.2f}°, Yaw={euler[2]:.2f}°")

    print("\nFinal state:")
    print(f"Quaternion: {imu_filter.get_orientation_quaternion()}")
    print(f"Euler angles: {imu_filter.get_orientation_euler()}")
    print(f"Gyro bias: {imu_filter.get_gyro_bias()}")
    print("\nNote: Yaw will drift without magnetometer. Roll and pitch are stable.")
