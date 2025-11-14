"""
Record and decode binary IMU payloads from ESP32 and write CSV.

Protocol (little-endian):
Header: 0xA5 0x5A [len_lo] [len_hi]
Payload (len bytes):
uint64_t timestamp_us (8)
float ax,ay,az (3*4)
float gx,gy,gz (3*4)
float mx,my,mz (3*4)
float pressure_pa (4)
float temperature_c (4)
float baro_altitude_m (4)
int32_t lat_scaled (4) // lat * 1e7
int32_t lon_scaled (4) // lon * 1e7
uint16_t speed_x100 (2) // speed_kmh * 100, 0xFFFF = invalid
float altitude_m (4)
uint32_t gps_time_packed (4) // hh<<24 | mm<<16 | ss<<8 | centisecond
uint8_t satellites (1)
uint16_t hdop_x100 (2) // hdop * 100, 0xFFFF = invalid
uint16_t heading_x100 (2) // heading_deg * 100, 0xFFFF = invalid

Total expected payload length: 79 bytes (but code reads length from header).

Usage:
    python kalman/record_imu_binary.py --port COM3 --baud 115200 --out real_imu_binary_decoded.csv

The script writes a CSV with the decoded fields and human-friendly units.
"""

import argparse
import serial
import struct
import time
import sys

HEADER = b"\xA5\x5A"
EXPECTED_PAYLOAD_LEN = 79

CSV_HEADER = (
    "timestamp_us,ax,ay,az,gx,gy,gz,mx,my,mz,pressure_pa,temperature_c,baro_altitude_m," \
    "lat,lon,speed_kmh,altitude_m,gps_h,gps_m,gps_s,gps_cs,satellites,hdop,heading_deg\n"
)

STRUCT_FMT = "<Q12fiiHfIBHH"  # matches 79 bytes
STRUCT_SIZE = struct.calcsize(STRUCT_FMT)


def find_header(buffer):
    return buffer.find(HEADER)


def parse_payload(payload_bytes):
    if len(payload_bytes) < STRUCT_SIZE:
        raise ValueError(f"Payload too short: {len(payload_bytes)} bytes, expected >= {STRUCT_SIZE}")

    # Unpack according to format
    vals = struct.unpack(STRUCT_FMT, payload_bytes[:STRUCT_SIZE])

    # Map values
    idx = 0
    timestamp_us = vals[idx]; idx += 1

    # 12 floats: ax ay az gx gy gz mx my mz pressure temperature baro_altitude (12 floats)
    ax = vals[idx]; ay = vals[idx+1]; az = vals[idx+2]
    gx = vals[idx+3]; gy = vals[idx+4]; gz = vals[idx+5]
    mx = vals[idx+6]; my = vals[idx+7]; mz = vals[idx+8]
    pressure_pa = vals[idx+9]; temperature_c = vals[idx+10]; baro_altitude_m = vals[idx+11]
    idx += 12

    lat_scaled = vals[idx]; idx += 1
    lon_scaled = vals[idx]; idx += 1
    speed_x100 = vals[idx]; idx += 1
    altitude_m = vals[idx]; idx += 1
    gps_time_packed = vals[idx]; idx += 1
    satellites = vals[idx]; idx += 1
    hdop_x100 = vals[idx]; idx += 1
    heading_x100 = vals[idx]; idx += 1

    # Convert scalings
    lat = lat_scaled / 1e7
    lon = lon_scaled / 1e7

    if speed_x100 == 0xFFFF:
        speed_kmh = ""
    else:
        speed_kmh = speed_x100 / 100.0

    if hdop_x100 == 0xFFFF:
        hdop = ""
    else:
        hdop = hdop_x100 / 100.0

    if heading_x100 == 0xFFFF:
        heading_deg = ""
    else:
        heading_deg = heading_x100 / 100.0

    # Decode gps_time_packed
    gps_h = (gps_time_packed >> 24) & 0xFF
    gps_m = (gps_time_packed >> 16) & 0xFF
    gps_s = (gps_time_packed >> 8) & 0xFF
    gps_cs = gps_time_packed & 0xFF

    return {
        'timestamp_us': int(timestamp_us),
        'ax': float(ax), 'ay': float(ay), 'az': float(az),
        'gx': float(gx), 'gy': float(gy), 'gz': float(gz),
        'mx': float(mx), 'my': float(my), 'mz': float(mz),
        'pressure_pa': float(pressure_pa), 'temperature_c': float(temperature_c), 'baro_altitude_m': float(baro_altitude_m),
        'lat': lat, 'lon': lon,
        'speed_kmh': speed_kmh,
        'altitude_m': float(altitude_m),
        'gps_h': int(gps_h), 'gps_m': int(gps_m), 'gps_s': int(gps_s), 'gps_cs': int(gps_cs),
        'satellites': int(satellites),
        'hdop': hdop,
        'heading_deg': heading_deg
    }


def open_serial(port, baudrate, timeout=1.0):
    try:
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"Opened serial port {port} @ {baudrate}")
        return ser
    except Exception as e:
        print(f"Failed to open serial port {port}: {e}")
        sys.exit(1)


def record_loop(ser, output_file, max_packets=None):
    buffer = bytearray()
    count = 0
    start_time = time.time()
    last_print = start_time

    with open(output_file, 'w', newline='') as f:
        f.write(CSV_HEADER)
        f.flush()

        try:
            while True:
                chunk = ser.read(256)
                if chunk:
                    buffer.extend(chunk)

                # Search for header
                hpos = find_header(buffer)
                if hpos == -1:
                    # Keep buffer from growing too large
                    if len(buffer) > 4096:
                        buffer = buffer[-1024:]
                    continue

                # Need at least header + 2 length bytes
                if len(buffer) < hpos + 4:
                    continue

                # Read length
                payload_len = buffer[hpos+2] | (buffer[hpos+3] << 8)

                # Check if full payload present
                total_len = hpos + 4 + payload_len
                if len(buffer) < total_len:
                    continue

                # Extract payload
                payload = bytes(buffer[hpos+4: total_len])

                # Remove consumed bytes from buffer
                del buffer[:total_len]

                # Parse
                try:
                    if payload_len < STRUCT_SIZE:
                        print(f"Warning: payload length {payload_len} < expected {STRUCT_SIZE}, skipping")
                        continue

                    decoded = parse_payload(payload)

                    # Write CSV line
                    # Columns: timestamp_us,ax,ay,az,gx,gy,gz,mx,my,mz,pressure_pa,temperature_c,baro_altitude_m,lat,lon,speed_kmh,altitude_m,gps_h,gps_m,gps_s,gps_cs,satellites,hdop,heading_deg
                    line = (
                        f"{decoded['timestamp_us']},{decoded['ax']:.6f},{decoded['ay']:.6f},{decoded['az']:.6f},"
                        f"{decoded['gx']:.6f},{decoded['gy']:.6f},{decoded['gz']:.6f},"
                        f"{decoded['mx']:.6f},{decoded['my']:.6f},{decoded['mz']:.6f},"
                        f"{decoded['pressure_pa']:.3f},{decoded['temperature_c']:.3f},{decoded['baro_altitude_m']:.3f},"
                        f"{decoded['lat']:.7f},{decoded['lon']:.7f},"
                        f"{decoded['speed_kmh'] if decoded['speed_kmh']!='' else ''},"
                        f"{decoded['altitude_m']:.3f},"
                        f"{decoded['gps_h']},{decoded['gps_m']},{decoded['gps_s']},{decoded['gps_cs']},"
                        f"{decoded['satellites']},{decoded['hdop'] if decoded['hdop']!='' else ''},{decoded['heading_deg'] if decoded['heading_deg']!='' else ''}\n"
                    )

                    f.write(line)
                    f.flush()

                    count += 1

                    # Print progress every 2 seconds
                    now = time.time()
                    if now - last_print > 2.0:
                        elapsed = now - start_time
                        print(f"Recorded {count} packets, elapsed {elapsed:.1f}s")
                        last_print = now

                    if max_packets and count >= max_packets:
                        print("Reached max packets, stopping")
                        break

                except Exception as e:
                    print(f"Error parsing payload: {e}")
                    continue

        except KeyboardInterrupt:
            print("\nUser requested stop (KeyboardInterrupt)")

    print(f"Finished. Total packets recorded: {count}")


def main():
    parser = argparse.ArgumentParser(description="Record and decode binary IMU packets from serial to CSV.")
    parser.add_argument('--port', '-p', default=None, help='Serial port (e.g. COM3 or /dev/ttyUSB0)')
    parser.add_argument('--baud', '-b', default=115200, type=int, help='Baud rate')
    parser.add_argument('--out', '-o', default='real_imu_binary_decoded.csv', help='Output CSV file')
    parser.add_argument('--max', '-m', default=0, type=int, help='Max packets to record (0 = unlimited)')

    args = parser.parse_args()

    if args.port is None:
        print("Please specify a serial port with --port")
        print("Available serial ports:")
        try:
            import serial.tools.list_ports
            for p in serial.tools.list_ports.comports():
                print(f"  {p.device} - {p.description}")
        except Exception:
            pass
        sys.exit(1)

    ser = open_serial(args.port, args.baud)
    try:
        record_loop(ser, args.out, max_packets=(args.max if args.max>0 else None))
    finally:
        ser.close()


if __name__ == '__main__':
    main()
