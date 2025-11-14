"""
Real-time 3D visualization of recorded orientation data.
Replays the data with proper timing based on recorded timestamps.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys

def load_orientation_data(filename):
    """Load recorded orientation data from CSV."""
    try:
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        return data[:, 0], data[:, 1:4]  # timestamps, [roll, pitch, yaw]
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        sys.exit(1)

def rotate_points(points, angles_deg):
    """Rotate points by given Euler angles (degrees)."""
    # Convert to radians
    angles = np.radians(angles_deg)
    roll, pitch, yaw = angles

    # Create rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Apply rotations in order: yaw, pitch, roll
    R = Rz @ Ry @ Rx
    return points @ R.T

def create_cube_points():
    """Create points for a cube visualization."""
    # Define cube vertices
    points = np.array([
        [1, 0.4, 0.2],   # Front-top-right
        [1, -0.4, 0.2],  # Front-bottom-right
        [-1, -0.4, 0.2], # Front-bottom-left
        [-1, 0.4, 0.2],  # Front-top-left
        [1, 0.4, -0.2],  # Back-top-right
        [1, -0.4, -0.2], # Back-bottom-right
        [-1, -0.4, -0.2],# Back-bottom-left
        [-1, 0.4, -0.2], # Back-top-left
    ])
    
    # Scale down the cube
    points *= 0.5
    
    # Define edges as pairs of vertex indices
    edges = [
        [0,1], [1,2], [2,3], [3,0],  # Front face
        [4,5], [5,6], [6,7], [7,4],  # Back face
        [0,4], [1,5], [2,6], [3,7],  # Connecting edges
    ]
    
    return points, edges

def update_plot(ax, points, edges, angles, elapsed):
    """Update the 3D plot with new orientation."""
    ax.cla()
    
    # Draw coordinate frame
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', alpha=0.5, length=1.0)
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', alpha=0.5, length=1.0)
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', alpha=0.5, length=1.0)
    
    # Rotate and draw cube
    rotated_points = rotate_points(points, angles)
    
    # Draw edges
    for edge in edges:
        p1, p2 = rotated_points[edge]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', linewidth=2)
    
    # Draw arrows on the cube to show orientation
    origin = rotated_points[0]  # Use front-top-right vertex as origin
    length = 0.8
    # Forward arrow (roll axis)
    ax.quiver(origin[0], origin[1], origin[2], 
             length, 0, 0, color='r', alpha=0.8)
    # Right arrow (pitch axis)
    ax.quiver(origin[0], origin[1], origin[2],
             0, length, 0, color='g', alpha=0.8)
    # Up arrow (yaw axis)
    ax.quiver(origin[0], origin[1], origin[2],
             0, 0, length, color='b', alpha=0.8)
    
    # Set plot properties
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Show angles and time in title
    ax.set_title(f'Time: {elapsed:.2f}s\nRoll: {angles[0]:6.1f}°  '
                f'Pitch: {angles[1]:6.1f}°  Yaw: {angles[2]:6.1f}°')

def replay_orientation(filename, playback_speed=1.0):
    """Replay orientation data with 3D visualization.

    Args:
        filename: Path to orientation CSV file
        playback_speed: Speed multiplier (1.0 = real-time, 2.0 = 2x speed, etc.)
    """
    print(f"Loading orientation data from {filename}...")
    timestamps, euler_angles = load_orientation_data(filename)

    # Calculate timing
    dt = np.diff(timestamps).mean()
    print(f"Average sample period: {dt:.3f}s ({1/dt:.1f} Hz)")
    print(f"Playback speed: {playback_speed}x")

    # Downsample to ~30 FPS for smooth visualization (skip frames if needed)
    target_fps = 30
    frame_skip = max(1, int((1/dt) / target_fps))
    indices = np.arange(0, len(euler_angles), frame_skip)
    timestamps = timestamps[indices]
    euler_angles = euler_angles[indices]
    print(f"Downsampled to {len(timestamps)} frames for smooth playback")

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create cube visualization
    points, edges = create_cube_points()

    print("\nStarting replay...")
    print("Close the window to stop")
    plt.ion()  # Enable interactive mode

    try:
        t_start = time.time()
        for i, angles in enumerate(euler_angles):
            # Target time adjusted by playback speed
            t_target = (timestamps[i] - timestamps[0]) / playback_speed
            t_elapsed = time.time() - t_start

            # Try to maintain timing (but don't wait too long if we're behind)
            if t_elapsed < t_target:
                time.sleep(min(0.05, t_target - t_elapsed))  # Cap sleep at 50ms

            # Update visualization
            actual_time = timestamps[i] - timestamps[0]
            update_plot(ax, points, edges, angles, actual_time)
            plt.draw()
            plt.pause(0.001)  # Required for interactive update

            # Print status periodically
            if i == 0 or i % 30 == 0:  # Print every ~1 second
                print(f"Time: {actual_time:6.2f}s  |  "
                      f"Roll: {angles[0]:7.2f}°  "
                      f"Pitch: {angles[1]:7.2f}°  "
                      f"Yaw: {angles[2]:7.2f}°")

    except KeyboardInterrupt:
        print("\nReplay stopped by user")
    finally:
        plt.ioff()
        plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_replay.py <orientation_data.csv> [speed]")
        print("\nExample:")
        print("  python visualize_replay.py real_imu_data_orientations.csv")
        print("  python visualize_replay.py real_imu_data_orientations.csv 2.0  # 2x speed")
        sys.exit(1)

    data_file = sys.argv[1]
    playback_speed = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    replay_orientation(data_file, playback_speed)