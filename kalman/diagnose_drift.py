import serial
import numpy as np
import time
import json


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
            if len(values) == 9:
                accel = np.array(values[0:3])
                gyro = np.array(values[3:6])
                mag = np.array(values[6:9])
                return accel, gyro, mag

        # JSON format
        elif line.startswith('{'):
            data = json.loads(line)
            accel = np.array([data['ax'], data['ay'], data['az']])
            gyro = np.array([data['gx'], data['gy'], data['gz']])
            mag = np.array([data['mx'], data['my'], data['mz']])
            return accel, gyro, mag

    except Exception as e:
        return None

    return None


def diagnose_drift(port='COM3', baudrate=115200, duration=30):
    """
    Monitor sensor data and computed orientation to diagnose drift.
    """
    try:
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=1.0)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"Connected to {port}")
    except serial.SerialException as e:
        print(f"Failed to connect: {e}")
        return

    # Gyro bias
    gyro_bias = np.array([-0.093434, 0.038554, -0.004698])

    print("\n" + "="*80)
    print("DRIFT DIAGNOSIS - Keep IMU stationary!")
    print("="*80)
    print("\nWatching sensor values in real-time...")
    print("Looking for:")
    print("  1. Gyro values near zero (after bias correction)")
    print("  2. Stable accelerometer readings")
    print("  3. Stable magnetometer readings")
    print("\n" + "="*80 + "\n")

    start_time = time.time()
    samples = []

    # Simple orientation tracking (just integrate gyro to see drift)
    orientation = np.zeros(3)  # [roll, pitch, yaw] in degrees
    dt = 0.01  # 100Hz

    try:
        while (time.time() - start_time) < duration:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore')
                except:
                    continue

                parsed = parse_imu_data(line)
                if parsed is None:
                    continue

                accel, gyro_raw, mag = parsed

                # Apply gyro bias correction
                gyro = gyro_raw - gyro_bias

                # Integrate gyro (very simple - just to show drift)
                orientation += np.degrees(gyro * dt)

                samples.append({
                    'time': time.time() - start_time,
                    'gyro': gyro,
                    'gyro_raw': gyro_raw,
                    'accel': accel,
                    'mag': mag,
                    'orientation': orientation.copy()
                })

                # Print every 100 samples (1 second)
                if len(samples) % 100 == 0:
                    t = samples[-1]['time']
                    print(f"[{t:5.1f}s]")
                    print(f"  Gyro (raw):       [{gyro_raw[0]:7.4f}, {gyro_raw[1]:7.4f}, {gyro_raw[2]:7.4f}] rad/s")
                    print(f"  Gyro (corrected): [{gyro[0]:7.4f}, {gyro[1]:7.4f}, {gyro[2]:7.4f}] rad/s")
                    print(f"  Accel:            [{accel[0]:7.3f}, {accel[1]:7.3f}, {accel[2]:7.3f}] m/s²")
                    print(f"  Mag:              [{mag[0]:7.2f}, {mag[1]:7.2f}, {mag[2]:7.2f}]")
                    print(f"  Integrated angle: Roll={orientation[0]:7.2f}° Pitch={orientation[1]:7.2f}° Yaw={orientation[2]:7.2f}°")
                    print()

    except KeyboardInterrupt:
        print("\nStopped by user")

    ser.close()

    if len(samples) < 100:
        print("Not enough samples!")
        return

    # Analysis
    print("\n" + "="*80)
    print("DRIFT ANALYSIS")
    print("="*80)

    gyros = np.array([s['gyro'] for s in samples])
    accels = np.array([s['accel'] for s in samples])
    mags = np.array([s['mag'] for s in samples])

    print("\n1. GYROSCOPE (after bias correction - should be ~0)")
    gyro_mean = gyros.mean(axis=0)
    print(f"   Mean:   [{gyro_mean[0]:7.4f}, {gyro_mean[1]:7.4f}, {gyro_mean[2]:7.4f}] rad/s")
    print(f"   StdDev: [{gyros.std(axis=0)[0]:7.4f}, {gyros.std(axis=0)[1]:7.4f}, {gyros.std(axis=0)[2]:7.4f}] rad/s")

    # Calculate drift rate
    drift_rate_deg_per_sec = np.degrees(gyro_mean)
    print(f"\n   Expected drift rate: [{drift_rate_deg_per_sec[0]:.3f}, {drift_rate_deg_per_sec[1]:.3f}, {drift_rate_deg_per_sec[2]:.3f}] °/s")

    if np.any(np.abs(gyro_mean) > 0.02):
        print("   ⚠️  WARNING: Gyro still has residual bias after correction!")
        print(f"   💡 Update GYRO_BIAS by adding: [{gyro_mean[0]:.6f}, {gyro_mean[1]:.6f}, {gyro_mean[2]:.6f}]")
        new_bias = gyro_bias + gyro_mean
        print(f"   New GYRO_BIAS = [{new_bias[0]:.6f}, {new_bias[1]:.6f}, {new_bias[2]:.6f}]")

    print("\n2. ACCELEROMETER (should be stable)")
    accel_mean = accels.mean(axis=0)
    accel_std = accels.std(axis=0)
    print(f"   Mean:   [{accel_mean[0]:7.3f}, {accel_mean[1]:7.3f}, {accel_mean[2]:7.3f}] m/s²")
    print(f"   StdDev: [{accel_std[0]:7.3f}, {accel_std[1]:7.3f}, {accel_std[2]:7.3f}] m/s²")

    if np.any(accel_std > 0.5):
        print("   ⚠️  WARNING: High accelerometer noise or IMU is moving!")

    print("\n3. MAGNETOMETER (should be stable)")
    mag_mean = mags.mean(axis=0)
    mag_std = mags.std(axis=0)
    print(f"   Mean:   [{mag_mean[0]:7.2f}, {mag_mean[1]:7.2f}, {mag_mean[2]:7.2f}]")
    print(f"   StdDev: [{mag_std[0]:7.2f}, {mag_std[1]:7.2f}, {mag_std[2]:7.2f}]")
    print(f"   Magnitude: {np.linalg.norm(mag_mean):.2f}")

    if np.linalg.norm(mag_mean) < 10:
        print("   ⚠️  WARNING: Magnetometer readings very low!")
        print("   💡 This might be in arbitrary units. UKF may struggle with this.")

    if np.any(mag_std > 5):
        print("   ⚠️  WARNING: Magnetometer unstable - possible interference!")

    # Compute simple gyro-only integrated orientation
    final_orientation = samples[-1]['orientation']
    print(f"\n4. INTEGRATED ORIENTATION (gyro-only, no filtering)")
    print(f"   Final: Roll={final_orientation[0]:.2f}° Pitch={final_orientation[1]:.2f}° Yaw={final_orientation[2]:.2f}°")
    print(f"   Drift in {duration}s: {np.linalg.norm(final_orientation):.2f}° total")
    print(f"   Drift rate: {np.linalg.norm(final_orientation)/duration:.3f} °/s")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)

    if np.linalg.norm(final_orientation) < 5:
        print("✓ Gyro drift is minimal (<5° total). UKF issue might be magnetometer!")
        print("  Try disabling magnetometer or increasing its noise in UKF.")
    elif np.any(np.abs(gyro_mean) > 0.01):
        print("✗ Gyro bias correction is incomplete. Update GYRO_BIAS as shown above.")
    else:
        print("? Gyro looks OK but still drifting. Possible causes:")
        print("  - Temperature-dependent bias")
        print("  - Magnetometer interference causing UKF to drift")
        print("  - IMU actually moving slightly")

    print("="*80)


if __name__ == "__main__":
    ports = find_serial_ports()

    if not ports:
        print("No serial ports found!")
        exit(1)

    PORT = ports[0]
    print(f"\nUsing port: {PORT}")
    print("Keep IMU COMPLETELY STILL for 30 seconds...\n")

    diagnose_drift(port=PORT, baudrate=115200, duration=30)
