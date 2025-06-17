#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import json
import rospy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped
import threading
from tf.transformations import euler_from_quaternion
from scipy.optimize import minimize

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

latest_poses = [None, None, None] 
latest_polytope = None
latest_dynamic_obstacles = []
centroid_trail = []  
robot_trails = [[], [], []]  
MAX_TRAIL_LENGTH = 100  
lock = threading.Lock()
dynamic_obstacle_lock = threading.Lock()

def compute_formation_vertices(z, ru, robot_dims):
    t_x, t_y, theta = z[0], z[1], z[2]
    l_r, w_r = robot_dims
    three_angles = [theta, 2 * np.pi / 3 + theta, 4 * np.pi / 3 + theta]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    vertices = []
    robot_centers = [] 

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
        robot_centers.append((x_center, y_center))  
        local_corners = [(l_r / 2, w_r / 2), (-l_r / 2, w_r / 2), (-l_r / 2, -w_r / 2), (l_r / 2, -w_r / 2)]
        for x_local, y_local in local_corners:
            x_rotated = x_center + cos_theta_i * x_local - sin_theta_i * y_local
            y_rotated = y_center + sin_theta_i * x_local + cos_theta_i * y_local
            vertices.append((x_rotated, y_rotated))
    return vertices, robot_centers

def amcl_pose_callback(msg, robot_id):
    global latest_poses
    rospy.loginfo("Received amcl_pose for robot %d: position (%f, %f), orientation (%f, %f, %f, %f)",
                  robot_id, msg.pose.pose.position.x, msg.pose.pose.position.y,
                  msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                  msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
    with lock:
        latest_poses[robot_id - 1] = msg.pose.pose

def invert_formation(x_centers, y_centers, thetas_i, ru):

    l_r, w_r = ru[9], ru[10]


    x_g_list, y_g_list = [], []

    a_list = []
    for i in range(3):
        x_c, y_c = x_centers[i], y_centers[i]
        theta_i = thetas_i[i]

        a_i = 0
        x_g = x_c - (a_i + l_r/2) * np.cos(theta_i)
        y_g = y_c - (a_i + l_r/2) * np.sin(theta_i)
        x_g_list.append(x_g)
        y_g_list.append(y_g)
        a_list.append(a_i)


    x0_g, y0_g = x_g_list[0], y_g_list[0]
    x0_l, y0_l = ru[0], ru[1]

    def residuals(params):
        t_x, t_y, theta = params
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        total_res = 0
        for i in range(3):
            x_l = ru[2 * i]
            y_l = ru[2 * i + 1]
            x_g_pred = t_x + cos_theta * x_l - sin_theta * y_l
            y_g_pred = t_y + sin_theta * x_l + cos_theta * y_l
            dx = x_g_pred - x_g_list[i]
            dy = y_g_pred - y_g_list[i]
            total_res += dx**2 + dy**2
        return total_res

    init_guess = [x0_g - x0_l, y0_g - y0_l, 0.0]
    res = minimize(residuals, init_guess)
    t_x, t_y, theta = res.x


    z = [t_x, t_y, theta]
    for i in range(3):
        offset = i * 2 * np.pi / 3
        z_i = thetas_i[i] - (theta + offset)
        z.append(z_i)

    return np.array(z)

def compute_formation_params():

    with lock:
        if None in latest_poses:
            return None
        poses = latest_poses.copy()

    x_max = map_size[1][0] 
    x_centers = []
    y_centers = []
    thetas_i = []
    for i, pose in enumerate(poses):
        x_map = pose.position.x
        y_map = pose.position.y

        x_center = x_max - y_map
        y_center = x_map
        quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        _, _, yaw = euler_from_quaternion(quaternion)

        theta_i = yaw - np.pi / 2
        theta_i = np.arctan2(np.sin(theta_i), np.cos(theta_i))
        x_centers.append(x_center)
        y_centers.append(y_center)
        thetas_i.append(theta_i)
        rospy.loginfo("Robot %d: x_center = %f, y_center = %f, theta_i = %f", i + 1, x_center, y_center, theta_i)

    z = invert_formation(x_centers, y_centers, thetas_i, ru)
    rospy.loginfo("Computed formation parameters: %s", z)
    return z

def dynamic_obstacle_callback(msg):

    global latest_dynamic_obstacles
    if len(msg.data) % 9 != 0:
        rospy.logwarn("Received malformed dynamic obstacle data: expected multiple of 9 values, got %d", len(msg.data))
        with dynamic_obstacle_lock:
            latest_dynamic_obstacles = []
        return

    try:
        with dynamic_obstacle_lock:
            current_obstacles = latest_dynamic_obstacles.copy()

        num_obstacles = len(msg.data) // 9
        for i in range(num_obstacles):
            obstacle_data = msg.data[i * 9:(i + 1) * 9]
            obstacle_id = obstacle_data[0]
            coords = np.array(obstacle_data[1:]).reshape(4, 2).T

            if np.all(coords == 0):
                current_obstacles = [obs for obs in current_obstacles if obs["id"] != obstacle_id]
                rospy.loginfo("Dynamic obstacle ID %d cleared (all zeros received)", obstacle_id)
                continue


            x_min, y_min = map_size[0]
            x_max, y_max = map_size[1]
            if not (x_min <= coords[0].min() <= coords[0].max() <= x_max and
                    y_min <= coords[1].min() <= coords[1].max() <= y_max):
                rospy.logwarn("Obstacle ID %d: Coordinates out of map bounds: %s", obstacle_id, coords)
                continue


            current_obstacles = [obs for obs in current_obstacles if obs["id"] != obstacle_id]
            current_obstacles.append({"id": obstacle_id, "coords": coords})
            rospy.loginfo("Received dynamic obstacle ID %d: %s", obstacle_id, coords)

        with dynamic_obstacle_lock:
            latest_dynamic_obstacles = current_obstacles

        if not latest_dynamic_obstacles:
            rospy.loginfo("No valid dynamic obstacles remain. Cleared obstacle list.")

    except (ValueError, TypeError) as e:
        rospy.logwarn("Failed to process dynamic obstacle data: %s", str(e))
        with dynamic_obstacle_lock:
            latest_dynamic_obstacles = []

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
        A = np.array(A_flat).reshape(A_rows, A_cols) if A_rows > 0 else None
        b = np.array(b_flat) if A_rows > 0 else None
        with lock:
            latest_polytope = {'A': A, 'b': b}
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


    for obs in obstacles:
        coords = list(zip(obs['coordinates'][0], obs['coordinates'][1]))
        polygon = Polygon(coords, closed=True, edgecolor='black', facecolor='gray')
        ax.add_patch(polygon)

    with dynamic_obstacle_lock:
        for obstacle in latest_dynamic_obstacles:
            coords = list(zip(obstacle["coords"][0], obstacle["coords"][1]))
            polygon = Polygon(coords, closed=True, edgecolor='black', facecolor='gray', alpha=0.7)
            ax.add_patch(polygon)
            centroid_x = np.mean(obstacle["coords"][0])
            centroid_y = np.mean(obstacle["coords"][1])
            ax.text(centroid_x, centroid_y, f"ID: {int(obstacle['id'])}", color='white', fontsize=8, ha='center', va='center')


    x = np.linspace(x_min - 1, x_max + 1, 400)
    y = np.linspace(y_min - 1, y_max + 1, 400)
    X, Y = np.meshgrid(x, y)
    for A, b in zip(P['A'], P['b']):
        Z = np.ones_like(X)
        for i in range(len(A)):
            Z *= (A[i, 0] * X + A[i, 1] * Y <= b[i])
        ax.contourf(X, Y, Z, levels=[0.5, 1], colors=['lightgray'], alpha=0.3)


    with lock:

        if len(centroid_trail) > 1:
            trail_x, trail_y = zip(*centroid_trail)
            ax.plot(trail_x, trail_y, 'g-', linewidth=1)

        for i in range(3):
            if len(robot_trails[i]) > 1:
                trail_x, trail_y = zip(*robot_trails[i])
                ax.plot(trail_x, trail_y, 'b-', linewidth=1)


    for idx, z in enumerate([G['V'][0], G['V'][-1]]):
        color = 'orange' if idx == 0 else 'red'
        verts, _ = compute_formation_vertices(z, ru, robot_dims)
        ax.add_patch(Polygon(verts[:3], closed=True, edgecolor=color, facecolor=color, alpha=0.5))
        for i in range(3):
            start = 3 + 4 * i
            ax.add_patch(Polygon(verts[start:start + 4], closed=True, edgecolor=color, facecolor=color, alpha=0.5))


    path_x = []
    path_y = []
    for z in G['V']:
        verts, _ = compute_formation_vertices(z, ru, robot_dims)
        centroid_x = sum(v[0] for v in verts[:3]) / 3
        centroid_y = sum(v[1] for v in verts[:3]) / 3
        path_x.append(centroid_x)
        path_y.append(centroid_y)
    ax.plot(path_x, path_y, 'k-', linewidth=2)

def main():
    global latest_poses, latest_polytope, centroid_trail, robot_trails
    rospy.init_node('formation_visualizer', anonymous=True)
    rospy.Subscriber('/robot1/amcl_pose', PoseWithCovarianceStamped, lambda msg: amcl_pose_callback(msg, 1), queue_size=10)
    rospy.Subscriber('/robot2/amcl_pose', PoseWithCovarianceStamped, lambda msg: amcl_pose_callback(msg, 2), queue_size=10)
    rospy.Subscriber('/robot3/amcl_pose', PoseWithCovarianceStamped, lambda msg: amcl_pose_callback(msg, 3), queue_size=10)
    rospy.Subscriber('/polytope_data', Float64MultiArray, polytope_callback, queue_size=10)
    rospy.Subscriber('/dynamic_obstacle', Float64MultiArray, dynamic_obstacle_callback, queue_size=10)

    fig, ax = plt.subplots()
    plt.ion()
    plt.show()

    draw_static_elements(ax)
    plt.pause(0.01)

    last_z = None
    last_polytope = None
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        z = compute_formation_params()
        rospy.loginfo("Current formation parameters: %s", z)
        with lock:
            polytope = latest_polytope.copy() if latest_polytope is not None else None

        if z is not None and (last_z is None or not np.allclose(z, last_z)):

            verts, robot_centers = compute_formation_vertices(z, ru, robot_dims)

            with lock:
                centroid_trail.append((z[0], z[1]))  
                if len(centroid_trail) > MAX_TRAIL_LENGTH:
                    centroid_trail.pop(0)
                for i in range(3):
                    robot_trails[i].append(robot_centers[i])
                    if len(robot_trails[i]) > MAX_TRAIL_LENGTH:
                        robot_trails[i].pop(0)

            draw_static_elements(ax)
            if polytope is not None and polytope['A'] is not None and polytope['b'] is not None:
                x = np.linspace(map_size[0][0] - 1, map_size[1][0] + 1, 400)
                y = np.linspace(map_size[0][1] - 1, map_size[1][1] + 1, 400)
                X, Y = np.meshgrid(x, y)
                Z = np.ones_like(X)
                for i in range(len(polytope['A'])):
                    Z *= (polytope['A'][i, 0] * X + polytope['A'][i, 1] * Y <= polytope['b'][i])
                ax.contourf(X, Y, Z, levels=[0.5, 1], colors=['lightblue'], alpha=0.5)

            verts, robot_centers = compute_formation_vertices(z, ru, robot_dims)
            ax.add_patch(Polygon(verts[:3], closed=True, edgecolor='blue', facecolor='blue', alpha=0.5))
            for i in range(3):
                start = 3 + 4 * i
                robot_vertices = verts[start:start + 4]
                ax.add_patch(Polygon(robot_vertices, closed=True, edgecolor='blue', facecolor='blue', alpha=0.5))
                x_center = sum(v[0] for v in robot_vertices) / 4
                y_center = sum(v[1] for v in robot_vertices) / 4
                ax.text(x_center, y_center, str(i + 1), color='white', fontsize=12, ha='center', va='center', weight='bold')
            last_z = z
            last_polytope = polytope
            plt.pause(0.01)

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Formation visualizer node terminated.")