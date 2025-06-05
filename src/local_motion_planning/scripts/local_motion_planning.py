#!/home/dat/env/bin/python
import rospy
import irispy
import numpy as np
import json
from snopt import formation
from planning_functions import (compute_polytope, process_new_polytope)
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Float64MultiArray

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
        # Tính khoảng cách giữa hai cấu hình
        dist = np.linalg.norm(z_next[:2] - z_curr[:2])
        # Tính số điểm nội suy
        K = int(np.ceil(dist / distance_threshold))
        K_values.append(K)
        # Tạo điểm nội suy
        points = interpolate_path(z_curr, z_next, K)
        interpolated_path.extend(points[:-1])  # Tránh lặp điểm cuối
    interpolated_path.append(z_values[-1])  # Thêm điểm cuối
    
    return interpolated_path, K_values

def create_pose_msg(x, y, theta, frame_id="map"):
    """Tạo PoseStamped message từ x, y, theta."""
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = frame_id
    pose_msg.header.stamp = rospy.Time.now()
    
    pose_msg.pose.position.x = x
    pose_msg.pose.position.y = y
    pose_msg.pose.position.z = 0.0
    
    # Chuyển đổi góc theta thành quaternion
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

        # Chuyển đổi sang hệ tọa độ mới
        x_new = y_center  # Trục x mới là y cũ
        y_new = x_max - x_center  # Trục y mới là x_max - x cũ
        theta_new = theta_g - np.pi / 2  # Điều chỉnh góc: trừ π/2 để tham chiếu từ trục x mới (hướng lên)
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))  # Chuẩn hóa góc về [-π, π]

        robot_poses.append((x_new, y_new, theta_new))

    return robot_poses

def local_motion_planning():
    """Thuật toán local motion planning."""
    rospy.init_node('local_motion_planning_node', anonymous=True)
    rate = rospy.Rate(1)

    pub_robots = [
        rospy.Publisher(f'/robot{i+1}/goal_pose', PoseStamped, queue_size=10)
        for i in range(3)
    ]

    pub_formation = rospy.Publisher('/formation_goal', Float64MultiArray, queue_size=10)

    while pub_formation.get_num_connections() == 0:
        rospy.loginfo("Waiting for subscriber to connect to /formation_goal...")
        rospy.sleep(0.5)

    data_config = load_json('/home/dat/catkin_ws/src/global_path_planning/config/global.json')
    data_path = load_json('/home/dat/catkin_ws/src/global_path_planning/config/global_path.json')

    T = data_path['T']
    z_values = data_path['z_values']
    polytopes = data_path['polytopes']

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
        a_i, a_i, a_i,
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
        if k_count <= K_values[k_index] + 1:
            A = np.array(polytopes[k_index]['A'])
            b = np.array(polytopes[k_index]['b'])
        else:
            k_index += 1
            k_count = 1
            if k_index >= len(K_values):
                rospy.loginfo("All points processed.")
                break
            A = np.array(polytopes[k_index]['A'])
            b = np.array(polytopes[k_index]['b'])

        if path_index + 1 <= len(interpolated_path):
            points_to_process = interpolated_path[path_index]

        zinit = np.array([points_to_process[0], points_to_process[1], points_to_process[2], points_to_process[3], points_to_process[4], points_to_process[5]])
        status_g, zg = formation(zinit, points_to_process[:2], A, b)
        
        if status_g == 1 and zg is not None:
            # Publish zg as Float64MultiArray
            zg_msg = Float64MultiArray()
            zg_msg.data = zg.tolist()  # Convert numpy array to list
            pub_formation.publish(zg_msg)
            rospy.loginfo(f"Published formation_goal: zg={zg.tolist()}")

            robot_poses = get_robot_poses(zg, ru, map_size)

            # Publish PoseStamped cho từng robot
            for i, (x, y, theta) in enumerate(robot_poses):
                pose_msg = create_pose_msg(x, y, theta)
                pub_robots[i].publish(pose_msg)
                rospy.loginfo(f"Published robot_{i+1}/goal_pose: x={x}, y={y}, theta={theta}")

            # Cập nhật z_pre thành zg cho điểm tiếp theo
            z_pre = zg
        else:
            rospy.logwarn(f"Formation failed for point {path_index}: status={status_g}, zg={zg}")
            break

        path_index += 1
        k_count += 1
        
        rate.sleep()

if __name__ == '__main__':
    try:
        local_motion_planning()
    except rospy.ROSInterruptException:
        rospy.loginfo("Local motion planning node terminated.")