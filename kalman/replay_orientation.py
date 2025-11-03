"""
Replay saved orientation data from offline_orientation.py with real-time visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import sys

def load_orientation_data(filename):
    """Load saved orientation data."""
    if filename.endswith('.npz'):
        data = np.load(filename)
        return data['timestamps'], data['euler_angles']
    elif filename.endswith('.csv'):
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        return data[:,0], data[:,1:4]
    else:
        raise ValueError("Unknown file format. Expected .npz or .csv")

def setup_3d_plot():
    """Setup 3D plot for visualization."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw coordinate frame
    length = 1.0
    ax.quiver(0, 0, 0, length, 0, 0, color='r', alpha=0.5)  # X axis
    ax.quiver(0, 0, 0, 0, length, 0, color='g', alpha=0.5)  # Y axis
    ax.quiver(0, 0, 0, 0, 0, length, color='b', alpha=0.5)  # Z axis
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return fig, ax

def create_oriented_box():
    """Create vertices and edges for a box."""
    # Define box vertices
    vertices = np.array([
        [ 0.5,  0.2,  0.1],  # Front-top-right
        [ 0.5, -0.2,  0.1],  # Front-bottom-right
        [-0.5, -0.2,  0.1],  # Front-bottom-left
        [-0.5,  0.2,  0.1],  # Front-top-left
        [ 0.5,  0.2, -0.1],  # Back-top-right
        [ 0.5, -0.2, -0.1],  # Back-bottom-right
        [-0.5, -0.2, -0.1],  # Back-bottom-left
        [-0.5,  0.2, -0.1],  # Back-top-left
    ])
    
    # Define edges as pairs of vertex indices
    edges = [
        [0,1], [1,2], [2,3], [3,0],  # Front face
        [4,5], [5,6], [6,7], [7,4],  # Back face
        [0,4], [1,5], [2,6], [3,7],  # Connecting edges
    ]
    
    return vertices, edges

def rotate_points(points, angles):
    """Rotate points by given Euler angles (in degrees)."""
    # Convert to radians
    roll, pitch, yaw = np.radians(angles)
    
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
    
    # Apply rotations
    R = Rz @ Ry @ Rx
    return points @ R.T

def update_plot(ax, vertices, edges, angles):
    """Update plot with new orientation."""
    ax.cla()
    
    # Draw coordinate frame
    length = 1.0
    ax.quiver(0, 0, 0, length, 0, 0, color='r', alpha=0.5, label='X')
    ax.quiver(0, 0, 0, 0, length, 0, color='g', alpha=0.5, label='Y')
    ax.quiver(0, 0, 0, 0, 0, length, color='b', alpha=0.5, label='Z')
    
    # Rotate and draw box
    rotated_vertices = rotate_points(vertices, angles)
    
    # Draw edges
    for edge in edges:
        v1, v2 = rotated_vertices[edge]
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 'k-')
    
    # Draw orientation axes on box
    axis_length = 0.8
    origin = rotated_vertices[0]  # Use front-top-right vertex as origin
    box_axes = np.array([
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ])
    rotated_axes = rotate_points(box_axes, angles)
    
    ax.quiver(origin[0], origin[1], origin[2], 
             rotated_axes[0,0], rotated_axes[0,1], rotated_axes[0,2],
             color='r', alpha=0.8)
    ax.quiver(origin[0], origin[1], origin[2],
             rotated_axes[1,0], rotated_axes[1,1], rotated_axes[1,2],
             color='g', alpha=0.8)
    ax.quiver(origin[0], origin[1], origin[2],
             rotated_axes[2,0], rotated_axes[2,1], rotated_axes[2,2],
             color='b', alpha=0.8)
    
    # Set plot properties
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Roll: {angles[0]:.1f}°  Pitch: {angles[1]:.1f}°  Yaw: {angles[2]:.1f}°')

def replay_orientation(data_file):
    """Replay orientation data with 3D visualization."""
    print(f"Loading orientation data from {data_file}...")
    timestamps, euler_angles = load_orientation_data(data_file)
    
    # Calculate timing
    dt = np.diff(timestamps).mean()
    print(f"Average sample period: {dt:.3f}s ({1/dt:.1f} Hz)")
    
    # Create 3D plot
    fig, ax = setup_3d_plot()
    vertices, edges = create_oriented_box()
    
    print("\nStarting replay...")
    print("Close the window to stop")
    
    try:
        for i, angles in enumerate(euler_angles):
            t_start = time.time()
            
            update_plot(ax, vertices, edges, angles)
            plt.pause(0.001)  # Required for interactive update
            
            # Try to maintain timing
            t_elapsed = time.time() - t_start
            if t_elapsed < dt:
                time.sleep(dt - t_elapsed)
            
            if i % 100 == 0:
                print(f"Time: {timestamps[i]:.2f}s  |  "
                      f"Roll: {angles[0]:7.2f}°  "
                      f"Pitch: {angles[1]:7.2f}°  "
                      f"Yaw: {angles[2]:7.2f}°")
    
    except KeyboardInterrupt:
        print("\nReplay stopped by user")
    
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python replay_orientation.py <orientation_data.npz/csv>")
        print("\nExample:")
        print("  python replay_orientation.py imu_data_orientations.npz")
        print("  python replay_orientation.py imu_data_orientations.csv")
        sys.exit(1)
    
    data_file = sys.argv[1]
    replay_orientation(data_file)