"""
Record real IMU data from ESP32 with timestamps.
Press ENTER to start recording, press ENTER again to stop.
"""

import serial
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


def record_imu_data(port='COM3', baudrate=115200, output_file='real_imu_data.csv'):
    """
    Record IMU data from ESP32 to CSV file.
    ESP32 format: timestamp_us,ax,ay,az,gx,gy,gz,mx,my,mz,pressure_pa,temperature_c,baro_altitude_m,lat,lon,speed_kmh,altitude_m,gps_time,satellites,hdop,heading_deg

    Args:
        port: Serial port
        baudrate: Baud rate
        output_file: Output filename
    """
    try:
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=1.0)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"Connected to {port}")
    except serial.SerialException as e:
        print(f"Failed to connect: {e}")
        return

    print("\n" + "="*80)
    print("REAL IMU DATA RECORDER")
    print("="*80)
    print(f"\nRecording to: {output_file}")
    print("ESP32 should output: timestamp_us,ax,ay,az,gx,gy,gz,mx,my,mz,pressure_pa,temperature_c,baro_altitude_m,lat,lon,speed_kmh,altitude_m,gps_time,satellites,hdop,heading_deg")
    print("\n" + "="*80)

    input("\nPress ENTER to start recording...")

    print("\n" + "="*80)
    print("RECORDING... Press CTRL+C to stop")
    print("="*80 + "\n")

    data = []
    start_time = time.time() 
    last_print_time = start_time
    recording = True

    try:
        with open(output_file, 'w') as f:
            # Write header
            f.write("# Real IMU Data from ESP32\n")
            f.write("# Format: timestamp_us,ax,ay,az,gx,gy,gz,mx,my,mz,pressure_pa,temperature_c,baro_altitude_m,lat,lon,speed_kmh,altitude_m,gps_time,satellites,hdop,heading_deg\n")
            f.write(f"# Recorded: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            while recording:
                elapsed = time.time() - start_time

                if ser.in_waiting > 0:
                    try:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()

                        # Parse CSV format: timestamp_us,ax,ay,az,gx,gy,gz,mx,my,mz,pressure_pa,temperature_c,baro_altitude_m,lat,lon,speed_kmh,altitude_m,gps_time,satellites,hdop,heading_deg
                        if ',' in line and '{' not in line and not line.startswith('#'):
                            values = [x for x in line.split(',')]

                            if len(values) >= 10:
                                # ESP32 already provides timestamp
                                # Write to file exactly as received
                                f.write(line + '\n')
                                f.flush()  # Ensure data is written immediately
                                data.append(values)

                                # Progress update every 2 seconds
                                if (time.time() - last_print_time) > 2:
                                    print(f"[{elapsed:.1f}s] Recorded {len(data)} samples (t={values[0]:.3f})")
                                    last_print_time = time.time()

                    except ValueError as e:
                        # Skip lines that can't be parsed
                        if time.time() - last_print_time > 3:
                            print(f"[{elapsed:.1f}s] Parse error (skipping): {e}")
                            last_print_time = time.time()
                    except Exception as e:
                        if time.time() - last_print_time > 3:
                            print(f"[{elapsed:.1f}s] Error: {e}")
                            last_print_time = time.time()
                else:
                    time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n\nStopping recording...")
        recording = False

    elapsed = time.time() - start_time
    ser.close()

    print("\n" + "="*80)
    print("RECORDING COMPLETE!")
    print("="*80)

    if len(data) > 0:
        # Get timestamp range from ESP32 data
        timestamps = [d[0] for d in data]
        duration = timestamps[-1] - timestamps[0]
        avg_sample_rate = (len(data) - 1) / duration if duration > 0 else 0

        print(f"\nRecorded {len(data)} samples")
        print(f"PC elapsed time: {elapsed:.1f} seconds")
        print(f"ESP32 timestamp range: {timestamps[0]:.3f}s to {timestamps[-1]:.3f}s")
        print(f"ESP32 duration: {duration:.3f} seconds")
        print(f"Average sample rate: {avg_sample_rate:.1f} Hz")
        print(f"Output file: {output_file}")
    else:
        print("\nWARNING: No data recorded!")
        print("Make sure ESP32 is outputting: timestamp,ax,ay,az,gx,gy,gz,mx,my,mz")

    print("\n" + "="*80)
    print("\nNext steps:")
    print(f"  1. Process with UKF: python offline_orientation.py {output_file}")
    print(f"  2. Visualize: python visualize_orientations.py {output_file.replace('.csv', '_orientations.npz')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys

    ports = find_serial_ports()
    if not ports:
        print("No serial ports found!")
        sys.exit(1)

    PORT = ports[0]
    BAUDRATE = 115200

    # Optional: specify output filename
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        output_file = 'real_imu_data.csv'

    print(f"\nUsing port: {PORT}")
    print(f"Baud rate: {BAUDRATE}")
    print(f"Output file: {output_file}\n")

    record_imu_data(PORT, BAUDRATE, output_file)
