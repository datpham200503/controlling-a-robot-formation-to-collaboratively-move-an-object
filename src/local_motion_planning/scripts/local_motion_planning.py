#!/home/dat/env/bin/python
import rospy
import irispy
import numpy as np
import json
from snopt import formation
from planning_functions import (compute_polytope, process_new_polytope)
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float64MultiArray
from tf.transformations import quaternion_from_euler

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
        theta_new = theta_g + np.pi / 2
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

        robot_poses.append((x_new, y_new, theta_new))

    return robot_poses

class LocalMotionPlanner:
    def __init__(self):
        rospy.init_node('local_motion_planning_node', anonymous=True)
        self.rate = rospy.Rate(0.5)

        self.pub_robots = [
            rospy.Publisher(f'/robot_{i+1}/goal_pose', PoseStamped, queue_size=10)
            for i in range(3)
        ]
        self.pub_formation = rospy.Publisher('/formation_goal', Float64MultiArray, queue_size=10)

        # Subscriber for dynamic obstacles
        self.dynamic_obstacle_A = None
        self.dynamic_obstacle_b = None
        self.dynamic_obstacle_sub = rospy.Subscriber('/dynamic_obstacles', Float64MultiArray, self.dynamic_obstacle_callback)

        # Subscriber for all goals reached signal
        self.all_goals_reached = False
        self.all_goals_reached_sub = rospy.Subscriber('/all_goals_reached', Bool, self.all_goals_reached_callback)

        # Load configuration and path data
        self.data_config = load_json('/home/dat/catkin_ws/src/global_path_planning/config/global.json')
        self.data_path = load_json('/home/dat/catkin_ws/src/global_path_planning/config/global_path.json')

        self.T = self.data_path['T']
        self.z_values = self.data_path['z_values']
        self.polytopes = self.data_path['polytopes']

        self.obstacles = convert_obstacles_to_numpy(self.data_config['obstacles'])
        self.initial_config = self.data_config['initial_configuration']
        self.start_centroid = np.array(self.data_config['start_centroid'])
        self.goal_centroid = np.array(self.data_config['goal_centroid'])
        self.object_radius = self.data_config['object_radius']
        self.robot_shape = self.data_config['robot_shape']
        self.map_size = self.data_config['map']

        self.bounds = irispy.Polyhedron.from_bounds(self.map_size[0], self.map_size[1])

        a_i = self.robot_shape['a_i']
        l_r = self.robot_shape['l_r']
        w_r = self.robot_shape['w_r']

        self.ru = [
            self.object_radius * np.cos(0.0), self.object_radius * np.sin(0.0),
            self.object_radius * np.cos(2 * np.pi / 3), self.object_radius * np.sin(2 * np.pi / 3),
            self.object_radius * np.cos(4 * np.pi / 3), self.object_radius * np.sin(4 * np.pi / 3),
            a_i, a_i, a_i,
            l_r, w_r
        ]

        self.interpolated_path, self.K_values = create_interpolated_path(self.T, self.z_values, distance_threshold=0.1)
        self.z_pre = self.z_values[0]
        self.path_index = 1
        self.k_count = 1
        self.k_index = 0
        self.last_publish_time = rospy.Time.now()

    def dynamic_obstacle_callback(self, msg):
        """Callback for dynamic obstacles (Ax < b)."""
        data = np.array(msg.data)
        # Assume data format: [A elements (m x n), b elements (m)]
        m = 4  # Assume 4 constraints for a square
        n = 2  # Assume 2D position constraints
        if len(data) != m * n + m:
            rospy.logwarn("Invalid dynamic obstacle data length: %d", len(data))
            return
        self.dynamic_obstacle_A = data[:m*n].reshape(m, n)
        self.dynamic_obstacle_b = data[m*n:]
        rospy.loginfo("Received dynamic obstacle: A=%s, b=%s", self.dynamic_obstacle_A, self.dynamic_obstacle_b)

    def all_goals_reached_callback(self, msg):
        """Callback for all goals reached signal."""
        self.all_goals_reached = msg.data
        rospy.loginfo("Received all_goals_reached: %s", self.all_goals_reached)

    def combine_constraints(self, A_polytope, b_polytope):
        """Combine polytope constraints with dynamic obstacle constraints."""
        if self.dynamic_obstacle_A is None or self.dynamic_obstacle_b is None:
            return A_polytope, b_polytope

        epsilon = 0.01  # Small margin to ensure strict inequality
        A_obstacle = self.dynamic_obstacle_A
        b_obstacle = self.dynamic_obstacle_b - epsilon  # Convert Ax < b to Ax <= b - epsilon

        # Combine all constraints
        A_combined = np.vstack([A_polytope, A_obstacle])
        b_combined = np.concatenate([b_polytope, b_obstacle])

        return A_combined, b_combined

    def run(self):
        """Main loop for local motion planning."""
        while self.pub_robots[0].get_num_connections() == 0:
            rospy.loginfo("Waiting for subscriber to connect...")
            rospy.sleep(0.5)

        # Send first goal
        points_to_process = self.interpolated_path[0]
        zinit = np.array([points_to_process[0], points_to_process[1], points_to_process[2], points_to_process[3], points_to_process[4], points_to_process[5]])
        A = np.array(self.polytopes[0]['A'])
        b = np.array(self.polytopes[0]['b'])
        A_combined, b_combined = self.combine_constraints(A, b)
        status_g, zg = formation(zinit, points_to_process[:2], A_combined, b_combined)
        
        if status_g == 1 and zg is not None:
            zg_msg = Float64MultiArray()
            zg_msg.data = zg.tolist()
            self.pub_formation.publish(zg_msg)
            rospy.loginfo(f"Published first formation_goal: zg={zg.tolist()}")

            robot_poses = get_robot_poses(zg, self.ru, self.map_size)
            for i, (x, y, theta) in enumerate(robot_poses):
                pose_msg = create_pose_msg(x, y, theta)
                self.pub_robots[i].publish(pose_msg)
                rospy.loginfo(f"Published first robot_{i+1}/goal_pose: x={x}, y={y}, theta={theta}")
        else:
            rospy.logwarn(f"Formation failed for first point: status={status_g}, zg={zg}")
            return

        rospy.loginfo("Waiting for all robots to reach first goals...")
        while not rospy.is_shutdown() and not self.all_goals_reached:
            self.rate.sleep()

        # After all robots reach first goals, proceed with subsequent goals every 2 seconds
        rospy.loginfo("Starting timed goal publishing every 2 seconds...")
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            if (current_time - self.last_publish_time).to_sec() >= 2.0 and self.path_index < len(self.interpolated_path):
                if self.k_count <= self.K_values[self.k_index] + 1:
                    A = np.array(self.polytopes[self.k_index]['A'])
                    b = np.array(self.polytopes[self.k_index]['b'])
                else:
                    self.k_index += 1
                    self.k_count = 1
                    if self.k_index >= len(self.K_values):
                        rospy.loginfo("All points processed.")
                        break
                    A = np.array(self.polytopes[self.k_index]['A'])
                    b = np.array(self.polytopes[self.k_index]['b'])

                A_combined, b_combined = self.combine_constraints(A, b)
                points_to_process = self.interpolated_path[self.path_index]
                zinit = np.array([points_to_process[0], points_to_process[1], points_to_process[2], points_to_process[3], points_to_process[4], points_to_process[5]])
                status_g, zg = formation(zinit, points_to_process[:2], A_combined, b_combined)
                
                if status_g == 1 and zg is not None:
                    zg_msg = Float64MultiArray()
                    zg_msg.data = zg.tolist()
                    self.pub_formation.publish(zg_msg)
                    rospy.loginfo(f"Published formation_goal: zg={zg.tolist()}")

                    robot_poses = get_robot_poses(zg, self.ru, self.map_size)
                    for i, (x, y, theta) in enumerate(robot_poses):
                        pose_msg = create_pose_msg(x, y, theta)
                        self.pub_robots[i].publish(pose_msg)
                        rospy.loginfo(f"Published robot_{i+1}/goal_pose: x={x}, y={y}, theta={theta}")

                    self.z_pre = zg
                    self.last_publish_time = current_time
                else:
                    rospy.logwarn(f"Formation failed for point %d: status=%d, zg=%s", self.path_index, status_g, zg)
                    break

                self.path_index += 1
                self.k_count += 1

            self.rate.sleep()

if __name__ == '__main__':
    try:
        planner = LocalMotionPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass