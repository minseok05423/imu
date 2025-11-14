"""
Automated pipeline: Process IMU data with UKF and visualize results.

Usage:
    python process_and_visualize.py <imu_data.csv> [playback_speed]

Example:
    python process_and_visualize.py real_imu_data.csv
    python process_and_visualize.py real_imu_data.csv 2.0
"""

import sys
import subprocess
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_and_visualize.py <imu_data.csv> [playback_speed]")
        print("\nExample:")
        print("  python process_and_visualize.py real_imu_data.csv")
        print("  python process_and_visualize.py real_imu_data.csv 2.0  # 2x speed")
        sys.exit(1)

    imu_data_file = sys.argv[1]
    playback_speed = sys.argv[2] if len(sys.argv) > 2 else "1.0"

    # Check if input file exists
    if not os.path.exists(imu_data_file):
        print(f"ERROR: File not found: {imu_data_file}")
        sys.exit(1)

    # Derive output filename
    orientation_file = imu_data_file.replace('.csv', '_orientations.csv')

    print("="*80)
    print("AUTOMATED IMU PROCESSING PIPELINE")
    print("="*80)
    print(f"\nInput:  {imu_data_file}")
    print(f"Output: {orientation_file}")
    print(f"Playback speed: {playback_speed}x\n")

    # Step 1: Process with UKF
    print("\n" + "="*80)
    print("STEP 1: Processing IMU data with UKF + RTS Smoother")
    print("="*80 + "\n")

    try:
        result = subprocess.run(
            ["python", "kalman/offline_orientation.py", imu_data_file],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: UKF processing failed with exit code {e.returncode}")
        sys.exit(1)

    # Check if output was created
    if not os.path.exists(orientation_file):
        print(f"\nERROR: Expected output file not found: {orientation_file}")
        sys.exit(1)

    # Step 2: Visualize results
    print("\n" + "="*80)
    print("STEP 2: Visualizing orientation data")
    print("="*80 + "\n")

    try:
        result = subprocess.run(
            ["python", "kalman/visualize_replay.py", orientation_file, playback_speed],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Visualization failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nVisualization stopped by user")

    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - {orientation_file}")
    print(f"  - {orientation_file.replace('.csv', '.npz')}")
    print(f"  - {imu_data_file.replace('.csv', '_comparison.png')}")
    print(f"  - {imu_data_file.replace('.csv', '_results.png')}")

if __name__ == "__main__":
    main()
