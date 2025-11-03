"""
Test magnetometer to see if it's suitable for your environment.
This will show you if there's magnetic interference.
"""

import serial
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Magnetometer calibration
MAG_OFFSET = np.array([0.0857, -0.7558, -0.6930])
MAG_A = np.array([
    [0.995363, 0.000293, 0.026286],
    [0.000293, 0.950979, -0.001707],
    [0.026286, -0.001707, 1.057143]
])


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
                return np.array(values[6:9])  # Just magnetometer

        elif line.startswith('{'):
            data = json.loads(line)
            return np.array([data['mx'], data['my'], data['mz']])

    except:
        return None

    return None


class MagnetometerTester:
    def __init__(self, port='COM3', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None

        # Data buffers
        self.mag_history = []
        self.max_samples = 500

        self.setup_plot()

    def setup_plot(self):
        self.fig = plt.figure(figsize=(16, 8))

        # 3D plot
        self.ax1 = self.fig.add_subplot(131, projection='3d')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax1.set_title('Magnetometer 3D Path', fontsize=12, fontweight='bold')

        # XY plane
        self.ax2 = self.fig.add_subplot(132)
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Y')
        self.ax2.set_title('Magnetometer XY (Horizontal Plane)', fontsize=12)
        self.ax2.grid(True)
        self.ax2.axis('equal')

        # Text display
        self.ax3 = self.fig.add_subplot(133)
        self.ax3.axis('off')
        self.text_display = self.ax3.text(0.1, 0.5, '', fontsize=14, family='monospace',
                                         verticalalignment='center')

        plt.tight_layout()

    def connect(self):
        try:
            self.serial = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=0.01)
            time.sleep(2)
            self.serial.reset_input_buffer()
            print(f"Connected to {self.port}")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            return False

    def read_mag(self):
        if self.serial and self.serial.in_waiting > 0:
            try:
                line = self.serial.readline().decode('utf-8', errors='ignore')
                mag_raw = parse_imu_data(line)

                if mag_raw is not None:
                    # Apply calibration
                    mag = (mag_raw - MAG_OFFSET) @ MAG_A.T

                    self.mag_history.append(mag)
                    if len(self.mag_history) > self.max_samples:
                        self.mag_history.pop(0)

                    return True
            except:
                pass
        return False

    def update_plot(self, frame):
        # Read data
        for _ in range(10):
            self.read_mag()

        if len(self.mag_history) < 10:
            return

        mag_array = np.array(self.mag_history)

        # Clear plots
        self.ax1.cla()
        self.ax2.cla()

        # 3D scatter
        self.ax1.scatter(mag_array[:, 0], mag_array[:, 1], mag_array[:, 2],
                        c=np.arange(len(mag_array)), cmap='viridis', s=2)
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax1.set_title('Magnetometer 3D Path (Should be on sphere)', fontsize=12)

        # Draw sphere
        if len(mag_array) > 50:
            magnitude = np.linalg.norm(mag_array, axis=1).mean()
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            x = magnitude * np.outer(np.cos(u), np.sin(v))
            y = magnitude * np.outer(np.sin(u), np.sin(v))
            z = magnitude * np.outer(np.ones(np.size(u)), np.cos(v))
            self.ax1.plot_surface(x, y, z, alpha=0.1, color='gray')

        # XY plane
        self.ax2.scatter(mag_array[:, 0], mag_array[:, 1],
                        c=np.arange(len(mag_array)), cmap='viridis', s=2)
        self.ax2.set_xlabel('X (North)')
        self.ax2.set_ylabel('Y (East)')
        self.ax2.set_title('Horizontal Plane (Should be on circle)', fontsize=12)
        self.ax2.grid(True)
        self.ax2.axis('equal')

        # Draw circle
        if len(mag_array) > 50:
            mag_horizontal = mag_array[:, :2]
            radius = np.linalg.norm(mag_horizontal, axis=1).mean()
            circle = plt.Circle((0, 0), radius, color='red', fill=False,
                              linestyle='--', linewidth=2)
            self.ax2.add_patch(circle)

        # Calculate statistics
        current_mag = mag_array[-1]
        mag_magnitude = np.linalg.norm(current_mag)
        mag_std = np.std(np.linalg.norm(mag_array, axis=1))

        # Calculate heading consistency
        headings = np.arctan2(mag_array[:, 1], mag_array[:, 0]) * 180 / np.pi
        heading_std = np.std(headings)

        # Diagnosis
        diagnosis = "GOOD ✓"
        issues = []

        if mag_std > 0.3:
            diagnosis = "UNSTABLE ⚠"
            issues.append("Magnitude varies too much")

        if heading_std > 15 and len(mag_array) > 100:
            diagnosis = "INTERFERENCE ✗"
            issues.append("Heading jumps around")

        text = f"""
╔══════════════════════════════════════╗
║   MAGNETOMETER ENVIRONMENT TEST      ║
╚══════════════════════════════════════╝

INSTRUCTIONS:
────────────────────────────
1. Keep IMU STILL on desk
2. Slowly rotate it 360° horizontally
3. Watch the pattern

CURRENT VALUES:
────────────────────────────
  Mag X: {current_mag[0]:7.2f}
  Mag Y: {current_mag[1]:7.2f}
  Mag Z: {current_mag[2]:7.2f}

  Magnitude: {mag_magnitude:.2f}
  Mag Std:   {mag_std:.3f}

HORIZONTAL HEADING:
────────────────────────────
  Current: {np.arctan2(current_mag[1], current_mag[0])*180/np.pi:6.1f}°
  Std Dev: {heading_std:6.1f}°

DIAGNOSIS: {diagnosis}
────────────────────────────
"""

        if issues:
            text += "\nISSUES DETECTED:\n"
            for issue in issues:
                text += f"  ✗ {issue}\n"

        text += f"""

WHAT TO LOOK FOR:
────────────────────────────
✓ 3D path forms smooth sphere
✓ XY path forms smooth circle
✓ Heading changes smoothly
  when you rotate IMU

✗ Path is distorted/ellipse
✗ Heading jumps erratically
✗ Points scattered randomly

Samples: {len(mag_array)}/{self.max_samples}
"""

        self.text_display.set_text(text)

    def run(self):
        if not self.connect():
            print("Failed to connect!")
            return

        print("\n" + "="*60)
        print("MAGNETOMETER ENVIRONMENT TEST")
        print("="*60)
        print("\nInstructions:")
        print("1. Keep the IMU still on your desk")
        print("2. Slowly rotate it 360° horizontally")
        print("3. Watch the XY plot - it should form a circle")
        print("\nIf the circle is distorted or heading jumps:")
        print("  → You have magnetic interference")
        print("  → Use 6-DOF mode instead of 9-DOF")
        print("\nClose window when done.")
        print("="*60 + "\n")

        anim = FuncAnimation(self.fig, self.update_plot, interval=50,
                           blit=False, cache_frame_data=False)
        plt.show()

        if self.serial:
            self.serial.close()


if __name__ == "__main__":
    ports = find_serial_ports()
    if not ports:
        print("No serial ports found!")
        exit(1)

    tester = MagnetometerTester(port=ports[0], baudrate=115200)
    tester.run()
