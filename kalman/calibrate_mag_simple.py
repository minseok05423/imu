"""
Simple and robust magnetometer calibration.
Works even with normalized/small magnitude data.
"""

import serial
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
            if len(values) >= 9:
                return np.array(values[6:9])

        elif line.startswith('{'):
            data = json.loads(line)
            return np.array([data['mx'], data['my'], data['mz']])

    except:
        return None

    return None


def calibrate_magnetometer_simple(mag_data):
    """
    Simple calibration: Just offset and scaling.
    More robust than full ellipsoid fitting.

    Returns:
        offset: Hard-iron offset
        scale: Soft-iron scale factors (diagonal only)
    """
    # Hard-iron offset: center of the data
    offset = (mag_data.max(axis=0) + mag_data.min(axis=0)) / 2

    # Soft-iron: scale each axis to have same range
    centered = mag_data - offset
    ranges = mag_data.max(axis=0) - mag_data.min(axis=0)
    avg_range = ranges.mean()

    # Scale factors to normalize all axes
    scale = avg_range / ranges

    # Expected field strength after calibration
    mag_cal = (mag_data - offset) * scale
    field_strength = np.linalg.norm(mag_cal, axis=1).mean()

    return offset, scale, field_strength


def collect_calibration_data(port='COM3', baudrate=115200, duration=15):
    """Collect magnetometer data."""
    try:
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=1.0)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"Connected to {port}")
    except serial.SerialException as e:
        print(f"Failed to connect: {e}")
        return None

    print("\n" + "="*80)
    print("SIMPLE MAGNETOMETER CALIBRATION")
    print("="*80)
    print(f"\nCollecting data for {duration} seconds...")
    print("\nWhen I say GO, rotate the sensor in ALL directions!")
    print("  - Rotate 360° horizontally")
    print("  - Tilt forward/backward")
    print("  - Roll left/right")
    print("  - Flip upside down")
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

        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8', errors='ignore')
                mag = parse_imu_data(line)

                if mag is not None:
                    mag_data.append(mag)

                    if (time.time() - last_print_time) > 2:
                        print(f"[{elapsed:.1f}s] {len(mag_data)} samples - Mag: [{mag[0]:.2f}, {mag[1]:.2f}, {mag[2]:.2f}]")
                        last_print_time = time.time()
            except:
                pass
        else:
            time.sleep(0.001)

    elapsed = time.time() - start_time
    ser.close()

    print(f"\nCollection complete! {len(mag_data)} samples in {elapsed:.1f}s")
    print("STOP rotating.\n")

    if len(mag_data) < 100:
        print("ERROR: Not enough samples!")
        return None

    return np.array(mag_data)


def visualize_calibration(mag_raw, mag_cal, offset, scale, field_strength):
    """Visualize results."""
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Magnetometer Calibration Results', fontsize=16, fontweight='bold')

    # 3D before
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(mag_raw[:, 0], mag_raw[:, 1], mag_raw[:, 2], c='b', s=1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Raw Data (Before)')
    ax1.set_box_aspect([1,1,1])

    # XY before
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(mag_raw[:, 0], mag_raw[:, 1], c='b', s=1)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Plane - Before')
    ax2.axis('equal')
    ax2.grid(True)

    # XZ before
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(mag_raw[:, 0], mag_raw[:, 2], c='b', s=1)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Plane - Before')
    ax3.axis('equal')
    ax3.grid(True)

    # 3D after
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.scatter(mag_cal[:, 0], mag_cal[:, 1], mag_cal[:, 2], c='g', s=1)

    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    x = field_strength * np.outer(np.cos(u), np.sin(v))
    y = field_strength * np.outer(np.sin(u), np.sin(v))
    z = field_strength * np.outer(np.ones(np.size(u)), np.cos(v))
    ax4.plot_surface(x, y, z, alpha=0.1, color='r')

    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('Calibrated Data (Should be Spherical)')
    ax4.set_box_aspect([1,1,1])

    # XY after
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(mag_cal[:, 0], mag_cal[:, 1], c='g', s=1)
    circle = plt.Circle((0, 0), field_strength, color='r', fill=False, linestyle='--')
    ax5.add_patch(circle)
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_title('XY Plane - After')
    ax5.axis('equal')
    ax5.grid(True)

    # XZ after
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.scatter(mag_cal[:, 0], mag_cal[:, 2], c='g', s=1)
    circle = plt.Circle((0, 0), field_strength, color='r', fill=False, linestyle='--')
    ax6.add_patch(circle)
    ax6.set_xlabel('X')
    ax6.set_ylabel('Z')
    ax6.set_title('XZ Plane - After')
    ax6.axis('equal')
    ax6.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ports = find_serial_ports()

    if not ports:
        print("No serial ports found!")
        exit(1)

    PORT = ports[0]
    BAUDRATE = 115200
    DURATION = 15

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

    offset, scale, field_strength = calibrate_magnetometer_simple(mag_raw)

    # Apply calibration
    mag_cal = (mag_raw - offset) * scale

    # Calculate error
    magnitudes = np.linalg.norm(mag_cal, axis=1)
    residuals = magnitudes - field_strength
    rms_error = np.sqrt(np.mean(residuals**2))

    # Print results
    print("Results:\n")
    print("Hard-Iron Offset:")
    print(f"  MAG_OFFSET = np.array([{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}])")
    print("\nSoft-Iron Scale Factors:")
    print(f"  MAG_SCALE = np.array([{scale[0]:.4f}, {scale[1]:.4f}, {scale[2]:.4f}])")
    print(f"\nExpected Field Strength: {field_strength:.4f}")
    print(f"RMS Error: {rms_error:.4f}\n")

    # Save
    np.savez('mag_calibration_simple.npz', offset=offset, scale=scale, field_strength=field_strength)
    print("="*80)
    print("Calibration saved to: mag_calibration_simple.npz")
    print("="*80 + "\n")

    # Print code
    print("="*80)
    print("PASTE THIS INTO YOUR CODE:")
    print("="*80 + "\n")
    print("# Magnetometer calibration (simple method)")
    print(f"MAG_OFFSET = np.array([{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}])")
    print(f"MAG_SCALE = np.array([{scale[0]:.4f}, {scale[1]:.4f}, {scale[2]:.4f}])")
    print("\n# Apply calibration:")
    print("mag_calibrated = (mag_raw - MAG_OFFSET) * MAG_SCALE")
    print("\n" + "="*80)

    # Visualize
    print("\nGenerating visualization...")
    visualize_calibration(mag_raw, mag_cal, offset, scale, field_strength)

    print("\n" + "="*80)
    print("CALIBRATION COMPLETE!")
    print("="*80)
