import serial
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg


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

        # CSV format
        if ',' in line and '{' not in line:
            values = [float(x) for x in line.split(',')]
            if len(values) >= 9:
                mag = np.array(values[6:9])
                return mag

        # JSON format
        elif line.startswith('{'):
            import json
            data = json.loads(line)
            mag = np.array([data['mx'], data['my'], data['mz']])
            return mag

    except Exception as e:
        return None

    return None


def calibrate_magnetometer(mag_data):
    """
    Calibrate magnetometer using ellipsoid fitting.
    Finds both hard-iron offsets and soft-iron correction matrix.

    Args:
        mag_data: Nx3 array of magnetometer readings

    Returns:
        offset: Hard-iron offset (3,)
        A: Soft-iron correction matrix (3x3)
        field_strength: Expected magnetic field strength
    """
    # Center the data
    offset = mag_data.mean(axis=0)
    mag_centered = mag_data - offset

    # Fit ellipsoid using algebraic method
    # Form the design matrix D
    D = np.array([
        mag_centered[:, 0]**2,
        mag_centered[:, 1]**2,
        mag_centered[:, 2]**2,
        2 * mag_centered[:, 1] * mag_centered[:, 2],
        2 * mag_centered[:, 0] * mag_centered[:, 2],
        2 * mag_centered[:, 0] * mag_centered[:, 1],
        2 * mag_centered[:, 0],
        2 * mag_centered[:, 1],
        2 * mag_centered[:, 2],
        np.ones(len(mag_centered))
    ]).T

    # Solve least squares
    _, _, v = linalg.svd(D)
    a = v[-1, :]

    # Form the matrix
    A_matrix = np.array([
        [a[0], a[5], a[4]],
        [a[5], a[1], a[3]],
        [a[4], a[3], a[2]]
    ])

    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = linalg.eig(A_matrix)

    # Construct soft-iron correction matrix
    # The correction matrix is the inverse square root of the scaled matrix
    A = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

    # Make sure A is real
    A = np.real(A)

    # Normalize A so that det(A) = 1
    A = A / np.cbrt(linalg.det(A))

    # Calculate expected field strength
    mag_corrected = (mag_data - offset) @ A.T
    field_strength = np.linalg.norm(mag_corrected, axis=1).mean()

    return offset, A, field_strength


def collect_calibration_data(port='COM3', baudrate=115200, duration=15):
    """
    Collect magnetometer data for calibration.

    Args:
        port: Serial port
        baudrate: Baud rate
        duration: Collection time in seconds
    """
    try:
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=1.0)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"Connected to {port}")
    except serial.SerialException as e:
        print(f"Failed to connect: {e}")
        return None

    print("\n" + "="*80)
    print("MAGNETOMETER CALIBRATION TOOL")
    print("="*80)
    print(f"\nWill collect data for {duration} seconds.\n")
    print("GET READY to rotate the sensor!")
    print("\nWhen I say GO:")
    print("  - Pick up the sensor")
    print("  - Rotate it in ALL directions")
    print("  - Flip upside down")
    print("  - Rotate 360° horizontally")
    print("  - Tilt at all angles")
    print("  - Make figure-8 patterns")
    print("\n" + "="*80)

    input("\nPress ENTER when ready...")

    print("\n" + "="*80)
    print(f"GO! START ROTATING NOW! ({duration} seconds)")
    print("="*80 + "\n")

    mag_data = []
    start_time = time.time()
    last_print_time = start_time

    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        remaining = duration - elapsed

        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8', errors='ignore')
                mag = parse_imu_data(line)

                if mag is not None:
                    mag_data.append(mag)

                    # Progress every 2 seconds
                    if (time.time() - last_print_time) > 2:
                        print(f"[{elapsed:.1f}s / {duration}s] Collected {len(mag_data)} samples - Mag: [{mag[0]:.2f}, {mag[1]:.2f}, {mag[2]:.2f}]")
                        last_print_time = time.time()
                else:
                    # Debug: print lines that couldn't be parsed (occasionally)
                    if time.time() - last_print_time > 3:
                        print(f"[{elapsed:.1f}s] Waiting for valid data... Last line: {line[:50]}")
                        last_print_time = time.time()
            except Exception as e:
                if time.time() - last_print_time > 3:
                    print(f"[{elapsed:.1f}s] Error: {e}")
                    last_print_time = time.time()
        else:
            time.sleep(0.001)  # Small delay if no data

    elapsed = time.time() - start_time
    ser.close()

    print(f"\nCollection complete! Collected {len(mag_data)} samples in {elapsed:.1f} seconds")
    print("STOP rotating.\n")

    if len(mag_data) < 100:
        print("ERROR: Not enough samples collected! Need at least 100.")
        return None

    return np.array(mag_data)


def visualize_calibration(mag_raw, mag_cal, offset, field_strength):
    """
    Visualize calibration results.
    """
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Magnetometer Calibration Results', fontsize=16, fontweight='bold')

    # 3D scatter - before
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(mag_raw[:, 0], mag_raw[:, 1], mag_raw[:, 2], c='b', marker='o', s=1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Raw Data (Before Calibration)')
    ax1.set_box_aspect([1,1,1])

    # XY plane - before
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(mag_raw[:, 0], mag_raw[:, 1], c='b', s=1)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Plane - Before')
    ax2.axis('equal')
    ax2.grid(True)

    # XZ plane - before
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(mag_raw[:, 0], mag_raw[:, 2], c='b', s=1)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Plane - Before')
    ax3.axis('equal')
    ax3.grid(True)

    # 3D scatter - after
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.scatter(mag_cal[:, 0], mag_cal[:, 1], mag_cal[:, 2], c='g', marker='o', s=1)

    # Draw ideal sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = field_strength * np.outer(np.cos(u), np.sin(v))
    y = field_strength * np.outer(np.sin(u), np.sin(v))
    z = field_strength * np.outer(np.ones(np.size(u)), np.cos(v))
    ax4.plot_surface(x, y, z, alpha=0.1, color='r')

    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('Calibrated Data (Should be Spherical)')
    ax4.set_box_aspect([1,1,1])

    # XY plane - after
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(mag_cal[:, 0], mag_cal[:, 1], c='g', s=1)
    circle = plt.Circle((0, 0), field_strength, color='r', fill=False, linestyle='--', label='Ideal')
    ax5.add_patch(circle)
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_title('XY Plane - After (Should be Circular)')
    ax5.axis('equal')
    ax5.grid(True)
    ax5.legend()

    # XZ plane - after
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.scatter(mag_cal[:, 0], mag_cal[:, 2], c='g', s=1)
    circle = plt.Circle((0, 0), field_strength, color='r', fill=False, linestyle='--', label='Ideal')
    ax6.add_patch(circle)
    ax6.set_xlabel('X')
    ax6.set_ylabel('Z')
    ax6.set_title('XZ Plane - After')
    ax6.axis('equal')
    ax6.grid(True)
    ax6.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Find ports
    ports = find_serial_ports()

    if not ports:
        print("No serial ports found!")
        exit(1)

    PORT = ports[0]
    BAUDRATE = 115200
    DURATION = 15  # Collect for 15 seconds

    print(f"\nUsing port: {PORT}")
    print(f"Baud rate: {BAUDRATE}\n")

    # Collect data
    mag_raw = collect_calibration_data(PORT, BAUDRATE, DURATION)

    if mag_raw is None or len(mag_raw) < 100:
        print("Failed to collect enough data!")
        exit(1)

    # Calibrate
    print("="*80)
    print("Calculating calibration parameters...")
    print("="*80 + "\n")

    offset, A, field_strength = calibrate_magnetometer(mag_raw)

    # Apply calibration
    mag_cal = (mag_raw - offset) @ A.T

    # Calculate error
    magnitudes = np.linalg.norm(mag_cal, axis=1)
    residuals = magnitudes - field_strength
    rms_error = np.sqrt(np.mean(residuals**2))

    # Print results
    print("Results:\n")
    print("Hard-Iron Offsets:")
    print(f"  offset = [{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}]")
    print("\nSoft-Iron Correction Matrix:")
    print(f"  A = [[{A[0,0]:.6f}, {A[0,1]:.6f}, {A[0,2]:.6f}],")
    print(f"       [{A[1,0]:.6f}, {A[1,1]:.6f}, {A[1,2]:.6f}],")
    print(f"       [{A[2,0]:.6f}, {A[2,1]:.6f}, {A[2,2]:.6f}]]")
    print(f"\nExpected Magnetic Field Strength: {field_strength:.2f}")
    print(f"RMS Residual Error: {rms_error:.4f}\n")

    # Save to file
    np.savez('mag_calibration.npz', offset=offset, A=A, field_strength=field_strength)
    print("="*80)
    print("Calibration parameters saved to: mag_calibration.npz")
    print("="*80 + "\n")

    # Print Python code
    print("="*80)
    print("PASTE THIS INTO YOUR PYTHON CODE:")
    print("="*80 + "\n")
    print("# Magnetometer calibration")
    print(f"MAG_OFFSET = np.array([{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}])")
    print(f"MAG_A = np.array([")
    print(f"    [{A[0,0]:.6f}, {A[0,1]:.6f}, {A[0,2]:.6f}],")
    print(f"    [{A[1,0]:.6f}, {A[1,1]:.6f}, {A[1,2]:.6f}],")
    print(f"    [{A[2,0]:.6f}, {A[2,1]:.6f}, {A[2,2]:.6f}]")
    print(f"])")
    print("\n# Apply calibration:")
    print("mag_calibrated = (mag_raw - MAG_OFFSET) @ MAG_A.T")
    print("\n" + "="*80)

    # Visualize
    print("\nGenerating visualization...")
    visualize_calibration(mag_raw, mag_cal, offset, field_strength)

    print("\n" + "="*80)
    print("CALIBRATION COMPLETE!")
    print("="*80)
