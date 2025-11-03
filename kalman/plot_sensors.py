import serial
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque


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


class IMUPlotter:
    def __init__(self, port='COM3', baudrate=115200, window_size=500):
        """
        Initialize IMU plotter.

        Args:
            port: Serial port
            baudrate: Baud rate
            window_size: Number of samples to show in the plot
        """
        self.port = port
        self.baudrate = baudrate
        self.window_size = window_size

        # Data buffers
        self.times = deque(maxlen=window_size)
        self.accel_x = deque(maxlen=window_size)
        self.accel_y = deque(maxlen=window_size)
        self.accel_z = deque(maxlen=window_size)
        self.gyro_x = deque(maxlen=window_size)
        self.gyro_y = deque(maxlen=window_size)
        self.gyro_z = deque(maxlen=window_size)
        self.mag_x = deque(maxlen=window_size)
        self.mag_y = deque(maxlen=window_size)
        self.mag_z = deque(maxlen=window_size)

        self.start_time = None
        self.serial = None

    def connect(self):
        """Connect to serial port."""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1
            )
            time.sleep(2)
            self.serial.reset_input_buffer()
            print(f"Connected to {self.port}")
            self.start_time = time.time()
            return True
        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            return False

    def read_data(self):
        """Read one sample from serial."""
        if self.serial and self.serial.in_waiting > 0:
            try:
                line = self.serial.readline().decode('utf-8', errors='ignore')
                parsed = parse_imu_data(line)

                if parsed is not None:
                    accel, gyro, mag = parsed

                    # Add to buffers
                    current_time = time.time() - self.start_time
                    self.times.append(current_time)

                    self.accel_x.append(accel[0])
                    self.accel_y.append(accel[1])
                    self.accel_z.append(accel[2])

                    self.gyro_x.append(gyro[0])
                    self.gyro_y.append(gyro[1])
                    self.gyro_z.append(gyro[2])

                    self.mag_x.append(mag[0])
                    self.mag_y.append(mag[1])
                    self.mag_z.append(mag[2])

                    return True
            except:
                pass
        return False

    def animate(self, frame):
        """Animation update function."""
        # Read multiple samples per frame for smoothness
        for _ in range(10):
            self.read_data()

        if len(self.times) == 0:
            return

        times = list(self.times)

        # Update accelerometer plot
        self.ax1.clear()
        self.ax1.plot(times, list(self.accel_x), 'r-', label='X', linewidth=1)
        self.ax1.plot(times, list(self.accel_y), 'g-', label='Y', linewidth=1)
        self.ax1.plot(times, list(self.accel_z), 'b-', label='Z', linewidth=1)
        self.ax1.set_ylabel('Accel (m/s²)', fontsize=10)
        self.ax1.set_title('Accelerometer', fontsize=12, fontweight='bold')
        self.ax1.legend(loc='upper right', fontsize=8)
        self.ax1.grid(True, alpha=0.3)

        # Update gyroscope plot
        self.ax2.clear()
        self.ax2.plot(times, list(self.gyro_x), 'r-', label='X', linewidth=1)
        self.ax2.plot(times, list(self.gyro_y), 'g-', label='Y', linewidth=1)
        self.ax2.plot(times, list(self.gyro_z), 'b-', label='Z', linewidth=1)
        self.ax2.set_ylabel('Gyro (rad/s)', fontsize=10)
        self.ax2.set_title('Gyroscope', fontsize=12, fontweight='bold')
        self.ax2.legend(loc='upper right', fontsize=8)
        self.ax2.grid(True, alpha=0.3)

        # Update magnetometer plot
        self.ax3.clear()
        self.ax3.plot(times, list(self.mag_x), 'r-', label='X', linewidth=1)
        self.ax3.plot(times, list(self.mag_y), 'g-', label='Y', linewidth=1)
        self.ax3.plot(times, list(self.mag_z), 'b-', label='Z', linewidth=1)
        self.ax3.set_ylabel('Mag (µT)', fontsize=10)
        self.ax3.set_xlabel('Time (s)', fontsize=10)
        self.ax3.set_title('Magnetometer', fontsize=12, fontweight='bold')
        self.ax3.legend(loc='upper right', fontsize=8)
        self.ax3.grid(True, alpha=0.3)

    def plot(self):
        """Start plotting."""
        # Create figure with 3 subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.suptitle('ESP32 IMU Sensor Data (Real-time)', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Start animation
        self.anim = FuncAnimation(
            self.fig,
            self.animate,
            interval=50,  # Update every 50ms
            blit=False,
            cache_frame_data=False
        )

        print("\nPlotting... Close the window to stop.")
        plt.show()

    def close(self):
        """Close serial connection."""
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("Serial connection closed")


if __name__ == "__main__":
    # Find available ports
    ports = find_serial_ports()

    if not ports:
        print("No serial ports found!")
        exit(1)

    PORT = ports[0]
    BAUDRATE = 115200

    print(f"\nUsing port: {PORT}")
    print(f"Baud rate: {BAUDRATE}")
    print(f"Window size: 500 samples (~5 seconds at 100Hz)\n")

    # Create plotter
    plotter = IMUPlotter(port=PORT, baudrate=BAUDRATE, window_size=500)

    # Connect and plot
    if plotter.connect():
        try:
            plotter.plot()
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            plotter.close()
    else:
        print("Failed to connect to serial port!")
