"""
Complete IMU solution with all calibrations applied.
MPU6050 (accel + gyro) + QMC5883L (magnetometer)
"""

import serial
import numpy as np
import time
import json
from imu import IMU_UKF


# ============================================================================
# CALIBRATION PARAMETERS (from calibration scripts)
# ============================================================================

# Gyroscope bias (from debug_sensors.py)
GYRO_BIAS = np.array([-0.093434, 0.038554, -0.004698])

# Accelerometer scale (1.0 if in m/s², 9.81 if in g)
ACCEL_SCALE = 1.0

# Magnetometer calibration (from calibrate_magnetometer.py)
MAG_OFFSET = np.array([0.0857, -0.7558, -0.6930])
MAG_A = np.array([
    [0.995363, 0.000293, 0.026286],
    [0.000293, 0.950979, -0.001707],
    [0.026286, -0.001707, 1.057143]
])

# ============================================================================


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


def run_imu_filter(port='COM3', baudrate=115200, duration=None):
    """
    Run the complete IMU filter with all calibrations.

    Args:
        port: Serial port
        baudrate: Baud rate
        duration: How long to run (None = infinite)
    """
    # Create UKF
    ukf = IMU_UKF(dt=0.01)  # 100Hz

    # Connect to serial
    try:
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=1.0)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"Connected to {port}")
    except serial.SerialException as e:
        print(f"Failed to connect: {e}")
        return

    print("\n" + "="*80)
    print("IMU ATTITUDE ESTIMATION")
    print("="*80)
    print("\nCalibrations applied:")
    print(f"  - Gyro bias correction: {GYRO_BIAS}")
    print(f"  - Accelerometer scale: {ACCEL_SCALE}")
    print(f"  - Magnetometer hard-iron offset: {MAG_OFFSET}")
    print(f"  - Magnetometer soft-iron correction: Enabled")
    print("\n" + "="*80)
    print("\nReading IMU data... Press Ctrl+C to stop\n")

    start_time = time.time()
    sample_count = 0

    try:
        while True:
            # Check duration
            if duration and (time.time() - start_time) > duration:
                break

            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore')
                except:
                    continue

                parsed = parse_imu_data(line)
                if parsed is None:
                    continue

                accel_raw, gyro_raw, mag_raw = parsed

                # ============================================================
                # APPLY ALL CALIBRATIONS
                # ============================================================

                # 1. Accelerometer scaling
                accel = accel_raw * ACCEL_SCALE

                # 2. Gyroscope bias correction
                gyro = gyro_raw - GYRO_BIAS

                # 3. Magnetometer calibration (hard-iron + soft-iron)
                mag = (mag_raw - MAG_OFFSET) @ MAG_A.T

                # ============================================================
                # UPDATE FILTER
                # ============================================================

                ukf.update(accel, gyro, mag)

                # Get orientation
                euler = ukf.get_orientation_euler()

                sample_count += 1
                timestamp = time.time() - start_time

                # Print every 10 samples (10Hz display)
                if sample_count % 10 == 0:
                    print(f"[{timestamp:6.1f}s] "
                          f"Roll: {euler[0]:7.2f}° | "
                          f"Pitch: {euler[1]:7.2f}° | "
                          f"Yaw: {euler[2]:7.2f}°")

    except KeyboardInterrupt:
        print(f"\n\nStopped. Processed {sample_count} samples in {time.time()-start_time:.1f}s")

    ser.close()

    # Final state
    print("\n" + "="*80)
    print("FINAL STATE:")
    print("="*80)
    print(f"Orientation (Euler): {ukf.get_orientation_euler()}")
    print(f"Orientation (Quat):  {ukf.get_orientation_quaternion()}")
    print(f"Gyro bias estimate:  {ukf.get_gyro_bias()}")
    print("="*80)


if __name__ == "__main__":
    # Find ports
    ports = find_serial_ports()

    if not ports:
        print("No serial ports found!")
        exit(1)

    PORT = ports[0]
    BAUDRATE = 115200

    print(f"\nUsing port: {PORT}")
    print(f"Baud rate: {BAUDRATE}\n")

    # Run the filter
    run_imu_filter(port=PORT, baudrate=BAUDRATE, duration=None)
