import serial
import numpy as np
import time


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
            import json
            data = json.loads(line)
            accel = np.array([data['ax'], data['ay'], data['az']])
            gyro = np.array([data['gx'], data['gy'], data['gz']])
            mag = np.array([data['mx'], data['my'], data['mz']])
            return accel, gyro, mag

    except Exception as e:
        return None

    return None


def debug_sensors(port='COM3', baudrate=115200, duration=10):
    """
    Debug raw sensor data to identify drift issues.
    """
    try:
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=1.0)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"Connected to {port}")
        print("=" * 80)
        print("Collecting raw sensor data for analysis...")
        print("Keep the IMU COMPLETELY STATIONARY!\n")

    except serial.SerialException as e:
        print(f"Failed to connect: {e}")
        return

    start_time = time.time()
    samples = []

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

                accel, gyro, mag = parsed
                samples.append({
                    'accel': accel,
                    'gyro': gyro,
                    'mag': mag,
                    'time': time.time() - start_time
                })

                # Print every 50 samples
                if len(samples) % 50 == 0:
                    print(f"[{len(samples):4d}] Accel: [{accel[0]:7.3f}, {accel[1]:7.3f}, {accel[2]:7.3f}] | "
                          f"Gyro: [{gyro[0]:7.3f}, {gyro[1]:7.3f}, {gyro[2]:7.3f}] | "
                          f"Mag: [{mag[0]:7.1f}, {mag[1]:7.1f}, {mag[2]:7.1f}]")

    except KeyboardInterrupt:
        print("\nStopped by user")

    ser.close()

    if len(samples) < 10:
        print("Not enough samples collected!")
        return

    # Analysis
    print("\n" + "=" * 80)
    print("SENSOR ANALYSIS (stationary IMU)")
    print("=" * 80)

    accels = np.array([s['accel'] for s in samples])
    gyros = np.array([s['gyro'] for s in samples])
    mags = np.array([s['mag'] for s in samples])

    print("\n1. ACCELEROMETER (should read ~9.8 m/s² total, pointing down)")
    print(f"   Mean:   [{accels.mean(axis=0)[0]:7.3f}, {accels.mean(axis=0)[1]:7.3f}, {accels.mean(axis=0)[2]:7.3f}]")
    print(f"   StdDev: [{accels.std(axis=0)[0]:7.3f}, {accels.std(axis=0)[1]:7.3f}, {accels.std(axis=0)[2]:7.3f}]")
    print(f"   Magnitude: {np.linalg.norm(accels.mean(axis=0)):.3f} m/s²")

    if abs(np.linalg.norm(accels.mean(axis=0)) - 9.81) > 1.0:
        print("   ⚠️  WARNING: Accelerometer not calibrated! Expected ~9.81 m/s²")

    print("\n2. GYROSCOPE (should read ~0 rad/s when stationary)")
    print(f"   Mean:   [{gyros.mean(axis=0)[0]:7.4f}, {gyros.mean(axis=0)[1]:7.4f}, {gyros.mean(axis=0)[2]:7.4f}] rad/s")
    print(f"   StdDev: [{gyros.std(axis=0)[0]:7.4f}, {gyros.std(axis=0)[1]:7.4f}, {gyros.std(axis=0)[2]:7.4f}] rad/s")

    gyro_bias = gyros.mean(axis=0)
    if np.any(np.abs(gyro_bias) > 0.1):
        print(f"   ⚠️  WARNING: Large gyro bias detected! This will cause drift.")
        print(f"   💡 SOLUTION: Subtract this bias: [{gyro_bias[0]:.4f}, {gyro_bias[1]:.4f}, {gyro_bias[2]:.4f}]")

    print("\n3. MAGNETOMETER (should be consistent, magnitude varies by location)")
    print(f"   Mean:   [{mags.mean(axis=0)[0]:7.1f}, {mags.mean(axis=0)[1]:7.1f}, {mags.mean(axis=0)[2]:7.1f}]")
    print(f"   StdDev: [{mags.std(axis=0)[0]:7.1f}, {mags.std(axis=0)[1]:7.1f}, {mags.std(axis=0)[2]:7.1f}]")
    print(f"   Magnitude: {np.linalg.norm(mags.mean(axis=0)):.1f}")

    mag_std_max = mags.std(axis=0).max()
    if mag_std_max > 50:
        print(f"   ⚠️  WARNING: High magnetometer noise! Possible magnetic interference nearby.")
        print(f"   💡 SOLUTION: Move away from computers, motors, magnets")

    print("\n4. NOISE LEVELS")
    print(f"   Accel noise: {accels.std(axis=0).mean():.4f} m/s²")
    print(f"   Gyro noise:  {gyros.std(axis=0).mean():.6f} rad/s")
    print(f"   Mag noise:   {mags.std(axis=0).mean():.2f}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)

    # Gyro bias correction
    if np.any(np.abs(gyro_bias) > 0.01):
        print(f"\n1. Add gyro bias correction to your ESP32 code:")
        print(f"   gyro_bias = [{gyro_bias[0]:.6f}, {gyro_bias[1]:.6f}, {gyro_bias[2]:.6f}]")
        print(f"   gyro_corrected = gyro_raw - gyro_bias")

    # Noise parameters
    print(f"\n2. Update UKF noise parameters in imu.py:")
    accel_noise = max(accels.std(axis=0).mean(), 0.01)
    gyro_noise = max(gyros.std(axis=0).mean(), 0.001)
    mag_noise = max(mags.std(axis=0).mean(), 1.0)
    print(f"   self.ukf.R[0:3, 0:3] *= {accel_noise:.4f}  # Accelerometer")
    print(f"   self.ukf.R[3:6, 3:6] *= {gyro_noise:.6f}  # Gyroscope")
    print(f"   self.ukf.R[6:9, 6:9] *= {mag_noise:.4f}  # Magnetometer")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    ports = find_serial_ports()

    if not ports:
        print("No serial ports found!")
        exit(1)

    PORT = ports[0]
    print(f"\nUsing port: {PORT}")
    print("Testing for 10 seconds...\n")

    debug_sensors(port=PORT, baudrate=115200, duration=10)
