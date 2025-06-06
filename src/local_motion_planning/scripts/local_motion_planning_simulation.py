#!/home/dat/env/bin/python
import rospy
import irispy
import numpy as np
import json
from snopt import formation
from planning_functions import compute_polytope, compute_formation_vertices
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Float64MultiArray
import threading

# === Biến đồng bộ cho vật cản động ===
latest_dynamic_obstacle = None
dynamic_obstacle_lock = threading.Lock()

def load_json(file_path):
    """Đọc dữ liệu từ file JSON."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def convert_obstacles_to_numpy(obstacles):
    converted_obstacles = []
    for obs in obstacles:
        coords = np.array(obs['coordinates'])
        converted_obstacles.append(coords)
    return converted_obstacles

def angle_difference(angle2, angle1):
    """Tính chênh lệch góc ngắn nhất giữa hai góc (rad)."""
    diff = angle2 - angle1
    return np.arctan2(np.sin(diff), np.cos(diff))

def interpolate_path(z_curr, z_next, K):
    """Tạo các điểm nội suy giữa z_curr và z_next với K điểm trung gian."""
    interpolated_points = []
    for k in range(K + 2):
        alpha = k / (K + 1)
        t_k = z_curr[:2] + alpha * (z_next[:2] - z_curr[:2])
        theta_k = z_curr[2:] + alpha * angle_difference(z_next[2:], z_curr[2:])
        t_k = np.concatenate((t_k, theta_k))
        interpolated_points.append(t_k)
    return interpolated_points

def create_interpolated_path(T, z_values, distance_threshold=0.1):
    """Tạo danh sách các điểm nội suy với khoảng cách ~10 cm."""
    interpolated_path = []
    K_values = []
    
    for i in range(len(T) - 1):
        z_curr = np.array(z_values[i])
        z_next = np.array(z_values[i + 1])
        dist = np.linalg.norm(z_next[:2] - z_curr[:2])
        K = int(np.ceil(dist / distance_threshold))
        K_values.append(K)
        points = interpolate_path(z_curr, z_next, K)
        interpolated_path.extend(points[:-1])
    interpolated_path.append(z_values[-1])
    
    return interpolated_path, K_values

def create_pose_msg(x, y, theta, frame_id="map"):
    """Tạo PoseStamped message từ x, y, theta."""
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = frame_id
    pose_msg.header.stamp = rospy.Time.now()
    
    pose_msg.pose.position.x = x
    pose_msg.pose.position.y = y
    pose_msg.pose.position.z = 0.0
    
    q = quaternion_from_euler(0, 0, theta)
    pose_msg.pose.orientation.x = q[0]
    pose_msg.pose.orientation.y = q[1]
    pose_msg.pose.orientation.z = q[2]
    pose_msg.pose.orientation.w = q[3]
    
    return pose_msg

def get_robot_poses(z, ru, map_size):
    """Tính toán vị trí trung tâm và hướng của từng robot từ các đỉnh đội hình."""
    robot_poses = []
    three_angles = [z[2], 2 * np.pi / 3 + z[2], 4 * np.pi / 3 + z[2]]
    
    t_x, t_y, theta = z[0], z[1], z[2]
    l_r, w_r = ru[9], ru[10]

    x_min, y_min = map_size[0]
    x_max, y_max = map_size[1]

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    for i in range(3):
        x_local = ru[2 * i]
        y_local = ru[2 * i + 1]
        x_global = t_x + cos_theta * x_local - sin_theta * y_local
        y_global = t_y + sin_theta * x_local + cos_theta * y_local

        theta_i = z[3 + i] + three_angles[i]
        cos_theta_i = np.cos(theta_i)
        sin_theta_i = np.sin(theta_i)
        x_g = x_global 
        y_g = y_global
        a_i = ru[6 + i]

        x_center = x_g + (a_i + l_r / 2) * cos_theta_i
        y_center = y_g + (a_i + l_r / 2) * sin_theta_i
        theta_g = z[3 + i] + three_angles[i]

        x_new = y_center
        y_new = x_max - x_center
        theta_new = theta_g - np.pi / 2
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

        robot_poses.append((x_new, y_new, theta_new))

    return robot_poses

def compute_ellipsoid_from_vertices(vertices):
    """Tạo elip nhỏ nhất chứa các đỉnh được cung cấp."""
    vertices = np.array(vertices)
    # Tính tâm elip (trung bình các đỉnh)
    centroid = np.mean(vertices, axis=0)
    # Tính bán kính elip (khoảng cách lớn nhất từ tâm đến đỉnh)
    distances = np.linalg.norm(vertices - centroid, axis=1)
    radius = np.max(distances)
    # Tạo elip sử dụng irispy.Ellipsoid.fromNSphere
    ellipsoid = irispy.Ellipsoid.fromNSphere(centroid, radius)
    return ellipsoid

def dynamic_obstacle_callback(msg):
    """Callback để nhận tọa độ vật cản động."""
    global latest_dynamic_obstacle
    if len(msg.data) != 8:
        rospy.logwarn("Received malformed dynamic obstacle data: expected 8 values, got %d", len(msg.data))
        return
    try:
        # Định dạng: [x1, y1, x2, y2, x3, y3, x4, y4]
        coords = np.array(msg.data).reshape(4, 2).T  # Shape (2, 4): [[x1, x2, x3, x4], [y1, y2, y3, y4]]
        with dynamic_obstacle_lock:
            latest_dynamic_obstacle = coords
        rospy.loginfo("Received dynamic obstacle: %s", coords)
    except Exception as e:
        rospy.logwarn("Failed to process dynamic obstacle data: %s", str(e))
        with dynamic_obstacle_lock:
            latest_dynamic_obstacle = None

def local_motion_planning():
    """Thuật toán local motion planning."""
    rospy.init_node('local_motion_planning_node', anonymous=True)
    rate = rospy.Rate(2)

    pub_robots = [
        rospy.Publisher(f'/robot{i+1}/goal_pose', PoseStamped, queue_size=10)
        for i in range(3)
    ]

    pub_formation = rospy.Publisher('/formation_goal', Float64MultiArray, queue_size=10)
    pub_polytope = rospy.Publisher('/polytope_data', Float64MultiArray, queue_size=10)

    rospy.Subscriber('/dynamic_obstacle', Float64MultiArray, dynamic_obstacle_callback, queue_size=10)

    while pub_formation.get_num_connections() == 0:
        rospy.loginfo("Waiting for subscriber to connect to /formation_goal...")
        rospy.sleep(0.5)

    data_config = load_json('/home/dat/catkin_ws/src/global_path_planning/config/global.json')
    data_path = load_json('/home/dat/catkin_ws/src/global_path_planning/config/global_path.json')

    T = data_path['T']
    z_values = data_path['z_values']

    obstacles = convert_obstacles_to_numpy(data_config['obstacles'])
    initial_config = data_config['initial_configuration']
    start_centroid = np.array(data_config['start_centroid'])
    goal_centroid = np.array(data_config['goal_centroid'])
    object_radius = data_config['object_radius']
    robot_shape = data_config['robot_shape']
    map_size = data_config['map']

    bounds = irispy.Polyhedron.from_bounds(map_size[0], map_size[1])

    a_i = robot_shape['a_i']
    l_r = robot_shape['l_r']
    w_r = robot_shape['w_r']

    ru = [
        object_radius * np.cos(0.0), object_radius * np.sin(0.0),
        object_radius * np.cos(2 * np.pi / 3), object_radius * np.sin(2 * np.pi / 3),
        object_radius * np.cos(4 * np.pi / 3), object_radius * np.sin(4 * np.pi / 3),
        a_i, a_i + 0.0001, a_i + 0.0002, #avoid same value in array
        l_r, w_r
    ]
    
    interpolated_path, K_values = create_interpolated_path(T, z_values, distance_threshold=0.1)
    rospy.loginfo(f"Starting local motion planning with {len(interpolated_path)} points.")
    z_pre = z_values[0]
    path_index = 1
    k_count = 1
    k_index = 0
    points_to_process = None
    
    while not rospy.is_shutdown():
        if path_index < len(interpolated_path):
            points_to_process = interpolated_path[path_index]
            vertices = compute_formation_vertices(points_to_process, ru)
            ellipsoid = compute_ellipsoid_from_vertices(vertices)
            rospy.loginfo(f"Computing polytope for point {path_index} with ellipsoid center={ellipsoid.getD()}, radius={ellipsoid.getC()[0,0]}")
            
            # Tạo danh sách obstacles bao gồm cả vật cản động nếu có
            current_obstacles = obstacles.copy()
            with dynamic_obstacle_lock:
                if latest_dynamic_obstacle is not None:
                    current_obstacles.append(latest_dynamic_obstacle)
                    rospy.loginfo("Added dynamic obstacle to obstacles list: %s", latest_dynamic_obstacle)
            
            A, b = compute_polytope(current_obstacles, ellipsoid, bounds)
            if A is None or b is None:
                rospy.logwarn("Failed to compute polytope A and b")
                polytope_msg = Float64MultiArray()
                polytope_msg.data = []
                pub_polytope.publish(polytope_msg)
            else:
                polytope_msg = Float64MultiArray()
                A_rows, A_cols = A.shape if A is not None else (0, 0)
                A_flat = A.flatten() if A is not None else np.array([])
                b_flat = b.flatten() if b is not None else np.array([])
                polytope_msg.data = [float(A_rows), float(A_cols)] + A_flat.tolist() + b_flat.tolist()
                pub_polytope.publish(polytope_msg)
        else:
            rospy.loginfo("All points processed.")
            break

        if path_index + 1 <= len(interpolated_path):
            points_to_process = interpolated_path[path_index]
        
        rospy.loginfo(f"Processing point {path_index} polytope A=\n{A}\nb={b}")
        zinit = np.array([points_to_process[0], points_to_process[1], points_to_process[2], points_to_process[3], points_to_process[4], points_to_process[5]])
        status_g, zg = formation(zinit, points_to_process[:2], A, b)
        
        if status_g in range(1, 7) and zg is not None:
            zg_msg = Float64MultiArray()
            zg_msg.data = zg.tolist()
            pub_formation.publish(zg_msg)
            rospy.loginfo(f"Published formation_goal: zg={zg.tolist()}")

            robot_poses = get_robot_poses(zg, ru, map_size)

            for i, (x, y, theta) in enumerate(robot_poses):
                pose_msg = create_pose_msg(x, y, theta)
                pub_robots[i].publish(pose_msg)
                rospy.loginfo(f"Published robot_{i+1}/goal_pose: x={x}, y={y}, theta={theta}")

            z_pre = zg
            path_index += 1
            k_count += 1
        else:
            # rospy.logwarn(f"Formation failed for point {path_index}: status={status_g}, zg={zg}")  # Commented out: Modified logging
            rospy.logwarn(f"Formation failed for point {path_index}: status={status_g}, zg={zg}. Continuing to calculate.")  # Added
            # break  # Commented out: Removed to continue loop
        
        rate.sleep()

if __name__ == '__main__':
    try:
        local_motion_planning()
    except rospy.ROSInterruptException:
        rospy.loginfo("Local motion planning node terminated.")