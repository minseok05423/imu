import serial
import numpy as np
import time
import json
from imu import IMU_UKF


class ESP32_IMU_Reader:
    """
    Read IMU data from ESP32 via serial and process with UKF.
    Expects data at 100Hz.
    """

    def __init__(self, port='COM3', baudrate=115200, accel_scale=9.81, gyro_bias=None):
        """
        Initialize serial connection to ESP32.

        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Baud rate (must match ESP32 configuration)
            accel_scale: Scale factor for accelerometer (9.81 if in g, 1.0 if in m/s²)
            gyro_bias: Gyro bias correction [bx, by, bz] in rad/s
        """
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.ukf = IMU_UKF(dt=0.01)  # 100Hz = 0.01s

        # Calibration parameters
        self.accel_scale = accel_scale
        self.gyro_bias = np.array(gyro_bias) if gyro_bias is not None else np.zeros(3)

    def connect(self):
        """Establish serial connection."""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1.0
            )
            time.sleep(2)  # Wait for ESP32 to reset after serial connection
            print(f"Connected to {self.port} at {self.baudrate} baud")

            # Flush any old data
            self.serial.reset_input_buffer()
            return True
        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Close serial connection."""
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("Serial connection closed")

    def parse_imu_data(self, line):
        """
        Parse IMU data from serial line.

        Expected format options:
        1. CSV: "ax,ay,az,gx,gy,gz,mx,my,mz"
        2. JSON: {"ax":..., "ay":..., ...}

        Returns:
            tuple: (accel, gyro, mag) as numpy arrays, or None if parse fails
        """
        try:
            # Remove whitespace
            line = line.strip()

            # Try CSV format first
            if ',' in line and '{' not in line:
                values = [float(x) for x in line.split(',')]
                if len(values) == 9:
                    accel = np.array(values[0:3])
                    gyro = np.array(values[3:6])
                    mag = np.array(values[6:9])
                    return accel, gyro, mag

            # Try JSON format
            elif line.startswith('{'):
                data = json.loads(line)
                accel = np.array([data['ax'], data['ay'], data['az']])
                gyro = np.array([data['gx'], data['gy'], data['gz']])
                mag = np.array([data['mx'], data['my'], data['mz']])
                return accel, gyro, mag

        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"Parse error: {e} - Line: {line}")
            return None

        return None

    def read_and_process(self, duration=None, callback=None):
        """
        Read data from ESP32 and process with UKF.

        Args:
            duration: How long to read in seconds (None = infinite)
            callback: Optional function called with (timestamp, euler_angles, accel, gyro, mag)
        """
        if not self.serial or not self.serial.is_open:
            print("Serial port not open. Call connect() first.")
            return

        start_time = time.time()
        sample_count = 0

        print("Reading IMU data... Press Ctrl+C to stop")

        try:
            while True:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break

                # Read line from serial
                if self.serial.in_waiting > 0:
                    try:
                        line = self.serial.readline().decode('utf-8', errors='ignore')
                    except UnicodeDecodeError:
                        continue

                    # Parse the data
                    parsed = self.parse_imu_data(line)
                    if parsed is None:
                        continue

                    accel, gyro, mag = parsed

                    # Apply calibration corrections
                    accel = accel * self.accel_scale  # Scale accelerometer
                    gyro = gyro - self.gyro_bias      # Remove gyro bias

                    # Update UKF
                    self.ukf.update(accel, gyro, mag)

                    # Get orientation
                    euler = self.ukf.get_orientation_euler()

                    sample_count += 1
                    timestamp = time.time() - start_time

                    # Call callback if provided
                    if callback:
                        callback(timestamp, euler, accel, gyro, mag)

                    # Print every 10 samples (10Hz display)
                    if sample_count % 10 == 0:
                        print(f"[{timestamp:.2f}s] Roll: {euler[0]:6.2f}° | "
                              f"Pitch: {euler[1]:6.2f}° | Yaw: {euler[2]:6.2f}°")

        except KeyboardInterrupt:
            print(f"\nStopped. Processed {sample_count} samples in {time.time()-start_time:.2f}s")
        except Exception as e:
            print(f"Error during reading: {e}")

    def get_ukf(self):
        """Get the UKF instance for direct access."""
        return self.ukf


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


# Example usage
if __name__ == "__main__":
    # Find available ports
    ports = find_serial_ports()

    if not ports:
        print("\nNo serial ports found!")
        exit(1)

    # Select port (change to your ESP32 port)
    # Common: COM3, COM4 on Windows; /dev/ttyUSB0, /dev/ttyACM0 on Linux
    PORT = ports[0] if ports else 'COM3'
    BAUDRATE = 115200  # Match your ESP32 configuration

    # Gyro bias from calibration (update these values from debug_sensors.py)
    GYRO_BIAS = [-0.093434, 0.038554, -0.004698]

    # Accelerometer scale (1.0 if already in m/s², 9.81 if in g)
    ACCEL_SCALE = 1.0

    print(f"\nUsing port: {PORT}")
    print(f"Gyro bias correction: {GYRO_BIAS}")

    # Create reader with calibration
    reader = ESP32_IMU_Reader(
        port=PORT,
        baudrate=BAUDRATE,
        accel_scale=ACCEL_SCALE,
        gyro_bias=GYRO_BIAS
    )

    # Connect
    if reader.connect():
        try:
            # Read for 30 seconds (or until Ctrl+C)
            reader.read_and_process(duration=30)

            # Get final state
            print("\nFinal UKF state:")
            print(f"  Orientation: {reader.ukf.get_orientation_euler()}")
            print(f"  Gyro bias: {reader.ukf.get_gyro_bias()}")

        finally:
            reader.disconnect()
