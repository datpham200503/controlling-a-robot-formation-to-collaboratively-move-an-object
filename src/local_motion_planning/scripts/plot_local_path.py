#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse  # Added: For ellipsoid plotting
import json
import rospy
from std_msgs.msg import Float64MultiArray
import threading
import time

# === Tải dữ liệu ===
global_json_path = '/home/dat/catkin_ws/src/global_path_planning/config/global.json'
global_path_json_path = '/home/dat/catkin_ws/src/global_path_planning/config/global_path.json'

with open(global_json_path, 'r') as f:
    global_data = json.load(f)
    
map_size = global_data['map']
z_init = global_data['initial_configuration']
obstacles = global_data['obstacles']
object_radius = global_data['object_radius']
a_i = global_data['robot_shape']['a_i']
l_r = global_data['robot_shape']['l_r']
w_r = global_data['robot_shape']['w_r']

with open(global_path_json_path, 'r') as f:
    path_data = json.load(f)

P = {
    'A': [np.array(p['A']) for p in path_data['polytopes']],
    'b': [np.array(p['b']) for p in path_data['polytopes']]
}
G = {'V': path_data['z_values'], 'E': []}
T = path_data['T']

ru = [
    object_radius * np.cos(0.0), object_radius * np.sin(0.0),
    object_radius * np.cos(2 * np.pi / 3), object_radius * np.sin(2 * np.pi / 3),
    object_radius * np.cos(4 * np.pi / 3), object_radius * np.sin(4 * np.pi / 3),
    a_i, a_i, a_i,
    l_r, w_r
]
robot_dims = [l_r, w_r]

# === Biến đồng bộ hình vẽ ===
latest_zg = None
latest_polytope = None
latest_dynamic_obstacle = None  # Added
lock = threading.Lock()
dynamic_obstacle_lock = threading.Lock()  # Added

def compute_formation_vertices(z, ru, robot_dims):
    t_x, t_y, theta = z[0], z[1], z[2]
    l_r, w_r = robot_dims
    three_angles = [theta, 2 * np.pi / 3 + theta, 4 * np.pi / 3 + theta]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    vertices = []

    for i in range(3):
        x_local = ru[2 * i]
        y_local = ru[2 * i + 1]
        x_global = t_x + cos_theta * x_local - sin_theta * y_local
        y_global = t_y + sin_theta * x_local + cos_theta * y_local
        vertices.append((x_global, y_global))

    for i in range(3):
        theta_i = z[3 + i] + three_angles[i]
        cos_theta_i = np.cos(theta_i)
        sin_theta_i = np.sin(theta_i)
        x_g, y_g = vertices[i]
        a_i = ru[6 + i]
        x_center = x_g + (a_i + l_r / 2) * cos_theta_i
        y_center = y_g + (a_i + l_r / 2) * sin_theta_i
        local_corners = [(l_r / 2, w_r / 2), (-l_r / 2, w_r / 2), (-l_r / 2, -w_r / 2), (l_r / 2, -w_r / 2)]
        for x_local, y_local in local_corners:
            x_rotated = x_center + cos_theta_i * x_local - sin_theta_i * y_local
            y_rotated = y_center + sin_theta_i * x_local + cos_theta_i * y_local
            vertices.append((x_rotated, y_rotated))
    return vertices

def formation_callback(msg):
    global latest_zg
    rospy.loginfo("Received formation_goal: %s", msg.data)
    if len(msg.data) != 6:
        rospy.logwarn("Malformed formation_goal received")
        return
    with lock:
        latest_zg = np.array(msg.data)

def dynamic_obstacle_callback(msg):  # Added
    """Callback để nhận tọa độ vật cản động."""
    global latest_dynamic_obstacle
    if len(msg.data) != 8:
        rospy.logwarn("Received malformed dynamic obstacle data: expected 8 values, got %d", len(msg.data))
        with dynamic_obstacle_lock:
            latest_dynamic_obstacle = None
        return
    try:
        coords = np.array(msg.data).reshape(4, 2).T  # Shape (2, 4): [[x1, x2, x3, x4], [y1, y2, y3, y4]]
        with dynamic_obstacle_lock:
            latest_dynamic_obstacle = coords
        rospy.loginfo("Received dynamic obstacle: %s", coords)
    except Exception as e:
        rospy.logwarn("Failed to process dynamic obstacle data: %s", str(e))
        with dynamic_obstacle_lock:
            latest_dynamic_obstacle = None

def polytope_callback(msg):
    global latest_polytope
    rospy.loginfo("Received polytope_data: %s", msg.data)
    if not msg.data:
        rospy.logwarn("Empty polytope data received")
        with lock:
            latest_polytope = None
        return
    try:
        A_rows = int(msg.data[0])
        A_cols = int(msg.data[1])
        A_flat = msg.data[2:2 + A_rows * A_cols]
        b_flat = msg.data[2 + A_rows * A_cols:2 + A_rows * A_cols + A_rows]
        ellipsoid_D = msg.data[2 + A_rows * A_cols + A_rows:2 + A_rows * A_cols + A_rows + 2]  # Added
        ellipsoid_C_flat = msg.data[2 + A_rows * A_cols + A_rows + 2:2 + A_rows * A_cols + A_rows + 6]  # Added
        A = np.array(A_flat).reshape(A_rows, A_cols) if A_rows > 0 else None
        b = np.array(b_flat) if A_rows > 0 else None
        ellipsoid_D = np.array(ellipsoid_D)
        ellipsoid_C = np.array(ellipsoid_C_flat).reshape(2, 2) if len(ellipsoid_C_flat) == 4 else np.eye(2)  # Added
        with lock:
            latest_polytope = {'A': A, 'b': b, 'ellipsoid_D': ellipsoid_D, 'ellipsoid_C': ellipsoid_C}  # Added
    except Exception as e:
        rospy.logwarn(f"Failed to process polytope data: {str(e)}")
        with lock:
            latest_polytope = None

def draw_static_elements(ax):
    ax.clear()
    x_min, y_min = map_size[0]
    x_max, y_max = map_size[1]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_title("Robot Formations and Polytopes")
    ax.grid(True)

    # Static Obstacles
    for obs in obstacles:
        coords = list(zip(obs['coordinates'][0], obs['coordinates'][1]))
        polygon = Polygon(coords, closed=True, edgecolor='black', facecolor='gray')
        ax.add_patch(polygon)

    # Dynamic Obstacle
    with dynamic_obstacle_lock:  # Added
        if latest_dynamic_obstacle is not None:  # Added
            coords = list(zip(latest_dynamic_obstacle[0], latest_dynamic_obstacle[1]))  # Added
            polygon = Polygon(coords, closed=True, edgecolor='black', facecolor='gray')  # Added
            ax.add_patch(polygon)  # Added

    # Static Polytopes
    x = np.linspace(x_min - 1, x_max + 1, 400)
    y = np.linspace(y_min - 1, y_max + 1, 400)
    X, Y = np.meshgrid(x, y)
    for A, b in zip(P['A'], P['b']):
        Z = np.ones_like(X)
        for i in range(len(A)):
            Z *= (A[i, 0] * X + A[i, 1] * Y <= b[i])
        ax.contourf(X, Y, Z, levels=[0.5, 1], colors=['lightgray'], alpha=0.3)

    # Initial and goal formation
    for idx, z in enumerate([G['V'][0], G['V'][-1]]):
        color = 'orange' if idx == 0 else 'red'
        verts = compute_formation_vertices(z, ru, robot_dims)
        ax.add_patch(Polygon(verts[:3], closed=True, edgecolor=color, facecolor=color, alpha=0.5))
        for i in range(3):
            start = 3 + 4 * i
            ax.add_patch(Polygon(verts[start:start + 4], closed=True, edgecolor=color, facecolor=color, alpha=0.5))

def main():
    global latest_zg, latest_polytope
    rospy.init_node('formation_visualizer', anonymous=True)
    rospy.Subscriber('/formation_goal', Float64MultiArray, formation_callback, queue_size=10)
    rospy.Subscriber('/polytope_data', Float64MultiArray, polytope_callback, queue_size=10)
    rospy.Subscriber('/dynamic_obstacle', Float64MultiArray, dynamic_obstacle_callback, queue_size=10)  # Added

    fig, ax = plt.subplots()
    plt.ion()
    plt.show()

    draw_static_elements(ax)
    plt.pause(0.01)

    last_zg = None
    last_polytope = None
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        with lock:
            z = latest_zg.copy() if latest_zg is not None else None
            polytope = latest_polytope.copy() if latest_polytope is not None else None

        if z is not None and (last_zg is None or not np.allclose(z, last_zg)):
            draw_static_elements(ax)
            # Draw dynamic polytope if available
            if polytope is not None and polytope['A'] is not None and polytope['b'] is not None:
                x = np.linspace(map_size[0][0] - 1, map_size[1][0] + 1, 400)
                y = np.linspace(map_size[0][1] - 1, map_size[1][1] + 1, 400)
                X, Y = np.meshgrid(x, y)
                Z = np.ones_like(X)
                for i in range(len(polytope['A'])):
                    Z *= (polytope['A'][i, 0] * X + polytope['A'][i, 1] * Y <= polytope['b'][i])
                ax.contourf(X, Y, Z, levels=[0.5, 1], colors=['lightblue'], alpha=0.5)
                # ax.plot(polytope['start_point'][0], polytope['start_point'][1], 'go', label='Start Point' if last_polytope is None else '')  # Commented out: Replaced with ellipsoid
            # Draw formation
            verts = compute_formation_vertices(z, ru, robot_dims)
            ax.add_patch(Polygon(verts[:3], closed=True, edgecolor='blue', facecolor='blue', alpha=0.5))
            for i in range(3):
                start = 3 + 4 * i
                robot_vertices = verts[start:start + 4]
                ax.add_patch(Polygon(robot_vertices, closed=True, edgecolor='blue', facecolor='blue', alpha=0.5))
                x_center = sum(v[0] for v in robot_vertices) / 4
                y_center = sum(v[1] for v in robot_vertices) / 4
                ax.text(x_center, y_center, str(i + 1), color='white', fontsize=12, ha='center', va='center', weight='bold')
            # Add legend only once
            if last_zg is None:
                ax.legend()
            last_zg = z
            last_polytope = polytope
            plt.pause(0.01)

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Formation visualizer node terminated.")