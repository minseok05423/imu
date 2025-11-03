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


if __name__ == "__main__":
    ports = find_serial_ports()

    if not ports:
        print("No serial ports found!")
        exit(1)

    PORT = ports[0]
    BAUDRATE = 115200

    print(f"\nUsing port: {PORT}")
    print(f"Baud rate: {BAUDRATE}")
    print("\nShowing raw data from ESP32...")
    print("="*80 + "\n")

    try:
        ser = serial.Serial(port=PORT, baudrate=BAUDRATE, timeout=1.0)
        time.sleep(2)
        ser.reset_input_buffer()

        print("Reading lines for 15 seconds (press Ctrl+C to stop early):\n")

        start_time = time.time()
        line_count = 0

        while time.time() - start_time < 15:  # Run for 15 seconds
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore')
                    print(f"[{line_count:3d}] {line.strip()}")
                    line_count += 1
                except Exception as e:
                    print(f"Error: {e}")
            else:
                time.sleep(0.01)

        print(f"\n{line_count} lines received in 15 seconds")
        ser.close()

    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "="*80)
    print("Now tell me:")
    print("1. What format is the data in? (CSV with commas? Space-separated?)")
    print("2. How many values per line?")
    print("3. What order are they in? (ax,ay,az,gx,gy,gz,mx,my,mz?)")
    print("="*80)
