#!/home/dat/env/bin/python
import rospy
import irispy
import numpy as np
import json
from snopt import formation
from planning_functions import compute_polytope
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Float64MultiArray
from collections import deque
import tf
import threading

class CombinedMotionPlanner:
    def __init__(self):

        rospy.init_node('combined_motion_planner', anonymous=True)
        self.rate = rospy.Rate(5)

        self.distance_threshold_x = rospy.get_param('~distance_threshold_x', 0.1)
        self.distance_threshold_y = rospy.get_param('~distance_threshold_y', 0.1)
        self.angle_threshold = rospy.get_param('~angle_threshold', 0.2)

        self.num_robots = 3
        self.current_poses = [None] * self.num_robots
        self.current_goals = [None] * self.num_robots
        self.goal_queues = [deque() for _ in range(self.num_robots)]
        self.last_goal_times = [rospy.Time.now() for _ in range(self.num_robots)]
        self.is_first_goal = True
        self.poses_initialized = [False] * self.num_robots

        self.goal_pubs = [rospy.Publisher(f'/robot{i+1}/goal', PoseStamped, queue_size=10) 
                         for i in range(self.num_robots)]
        self.pub_formation = rospy.Publisher('/formation_goal', Float64MultiArray, queue_size=10)
        self.pub_polytope = rospy.Publisher('/polytope_data', Float64MultiArray, queue_size=10)

        self.amcl_pose_subs = []
        self.goal_pose_subs = []
        for i in range(self.num_robots):
            robot_id = i + 1
            self.amcl_pose_subs.append(
                rospy.Subscriber(f'/robot{robot_id}/amcl_pose', 
                               PoseWithCovarianceStamped, 
                               lambda msg, idx=i: self.amcl_pose_callback(msg, idx))
            )
            self.goal_pose_subs.append(
                rospy.Subscriber(f'/robot{robot_id}/goal_pose', 
                               PoseStamped, 
                               lambda msg, idx=i: self.goal_pose_callback(msg, idx))
            )

        self.latest_dynamic_obstacles = []
        
        self.dynamic_obstacle_lock = threading.Lock()
        rospy.Subscriber('/dynamic_obstacle', Float64MultiArray, self.dynamic_obstacle_callback, queue_size=10)

        self.tf_listener = tf.TransformListener()

        self.data_config = self.load_json('/home/dat/catkin_ws/src/global_path_planning/config/global.json')
        self.data_path = self.load_json('/home/dat/catkin_ws/src/global_path_planning/config/global_path.json')

        self.T = self.data_path['T']
        self.z_values = self.data_path['z_values']
        self.obstacles = self.convert_obstacles_to_numpy(self.data_config['obstacles'])
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


        self.interpolated_path, self.K_values = self.create_interpolated_path(self.T, self.z_values, distance_threshold=0.1)
        self.path_index = 0
        self.z_pre = self.z_values[0]

        self.last_update_time = rospy.Time.now()


        rospy.sleep(10.0)

    def load_json(self, file_path):

        with open(file_path, 'r') as f:
            return json.load(f)

    def convert_obstacles_to_numpy(self, obstacles):

        return [np.array(obs['coordinates']) for obs in obstacles]

    def angle_difference(self, angle2, angle1):

        diff = angle2 - angle1
        return np.arctan2(np.sin(diff), np.cos(diff))

    def interpolate_path(self, z_curr, z_next, K):

        interpolated_points = []
        for k in range(K + 2):
            alpha = k / (K + 1)
            t_k = z_curr[:2] + alpha * (z_next[:2] - z_curr[:2])
            theta_k = z_curr[2:] + alpha * self.angle_difference(z_next[2:], z_curr[2:])
            t_k = np.concatenate((t_k, theta_k))
            interpolated_points.append(t_k)
        return interpolated_points

    def create_interpolated_path(self, T, z_values, distance_threshold=0.1):

        interpolated_path = []
        K_values = []
        for i in range(len(T) - 1):
            z_curr = np.array(z_values[i])
            z_next = np.array(z_values[i + 1])
            dist = np.linalg.norm(z_next[:2] - z_curr[:2])
            K = int(np.ceil(dist / distance_threshold))
            K_values.append(K)
            points = self.interpolate_path(z_curr, z_next, K)
            interpolated_path.extend(points[:-1])
        interpolated_path.append(z_values[-1])
        return interpolated_path, K_values

    def create_pose_msg(self, x, y, theta, frame_id="map"):

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

    def get_robot_poses(self, z, ru, map_size):

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

    def compute_ellipsoid_from_vertices(self, vertices):

        vertices = np.array(vertices)
        centroid = np.mean(vertices, axis=0)
        distances = np.linalg.norm(vertices - centroid, axis=1)
        radius = max(np.max(distances), 0.1)
        ellipsoid = irispy.Ellipsoid.fromNSphere(centroid, radius)
        return ellipsoid

    def get_vertices_from_poses(self):

        vertices = []
        x_max = self.map_size[1][0]
        for i in range(self.num_robots):
            if self.current_poses[i] is not None:
                pose = self.current_poses[i]

                x_map = pose.position.x
                y_map = pose.position.y
                yaw = self.get_yaw(pose.orientation)

                x_center = x_max - y_map
                y_center = x_map

                vertices.append([x_center, y_center])
            else:
                rospy.logwarn("Robot %d: AMCL pose is None, using goal pose for ellipsoid", i + 1)
                if self.current_goals[i] is not None:
                    pose = self.current_goals[i].pose

                    x_map = pose.position.x
                    y_map = pose.position.y
                    x_center = x_max - y_map
                    y_center = x_map
                    vertices.append([x_center, y_center])
                else:
                    rospy.logwarn("Robot %d: No goal pose available, skipping ellipsoid computation", i + 1)
                    return None
        return vertices

    def dynamic_obstacle_callback(self, msg):

        if len(msg.data) % 9 != 0:
            rospy.logwarn("Received malformed dynamic obstacle data: expected multiple of 9 values, got %d", len(msg.data))
            with self.dynamic_obstacle_lock:
                self.latest_dynamic_obstacles = []
            return

        try:

            with self.dynamic_obstacle_lock:
                current_obstacles = self.latest_dynamic_obstacles.copy()


            num_obstacles = len(msg.data) // 9
            new_obstacles = []
            for i in range(num_obstacles):
                obstacle_data = msg.data[i * 9:(i + 1) * 9]
                obstacle_id = obstacle_data[0]

                coords = np.array(obstacle_data[1:]).reshape(4, 2).T 


                if np.all(coords == 0):

                    current_obstacles = [obs for obs in current_obstacles if obs["id"] != obstacle_id]
                    rospy.loginfo("Obstacle ID %d: Removed (all coordinates are zero)", obstacle_id)
                    continue

                x_min, y_min = self.map_size[0]
                x_max, y_max = self.map_size[1]
                if not (x_min <= coords[0].min() <= coords[0].max() <= x_max and
                        y_min <= coords[1].min() <= coords[1].max() <= y_max):
                    rospy.logwarn("Obstacle ID %d: Coordinates out of map bounds: %s", obstacle_id, coords)
                    continue


                if any(obs["id"] == obstacle_id for obs in current_obstacles):
                    rospy.logwarn("Obstacle ID %d: Duplicate ID detected, updating existing obstacle", obstacle_id)
                    current_obstacles = [obs for obs in current_obstacles if obs["id"] != obstacle_id]


                current_obstacles.append({"id": obstacle_id, "coords": coords})
                rospy.loginfo("Received obstacle ID %d: %s", obstacle_id, coords)


            with self.dynamic_obstacle_lock:
                self.latest_dynamic_obstacles = current_obstacles

            if not self.latest_dynamic_obstacles:
                rospy.loginfo("No valid dynamic obstacles remain. Cleared obstacle list.")

        except (ValueError, TypeError) as e:
            rospy.logwarn("Failed to process dynamic obstacle data: %s", str(e))
            with self.dynamic_obstacle_lock:
                self.latest_dynamic_obstacles = []

    def amcl_pose_callback(self, msg, robot_idx):

        self.current_poses[robot_idx] = msg.pose.pose
        self.poses_initialized[robot_idx] = True
        rospy.loginfo("Robot %d: AMCL pose updated: x=%f, y=%f", 
                      robot_idx + 1, msg.pose.pose.position.x, msg.pose.pose.position.y)

    def goal_pose_callback(self, msg, robot_idx):

        self.goal_queues[robot_idx].append(msg)
        rospy.loginfo("Robot %d: New goal received at x=%f, y=%f, queue size: %d", 
                      robot_idx + 1, msg.pose.position.x, msg.pose.position.y, len(self.goal_queues[robot_idx]))

    def get_yaw(self, quaternion):

        try:
            euler = tf.transformations.euler_from_quaternion(
                [quaternion.x, quaternion.y, quaternion.z, quaternion.w])
            return euler[2]
        except Exception as e:
            rospy.logerr("Quaternion conversion error: %s", e)
            return 0.0

    def has_reached_goal(self, robot_idx):

        if self.current_poses[robot_idx] is None or self.current_goals[robot_idx] is None:
            rospy.logwarn("Robot %d: Pose or goal is None (pose=%s, goal=%s)", 
                          robot_idx + 1, self.current_poses[robot_idx], self.current_goals[robot_idx])
            return False
        try:
            goal_pose_stamped = PoseStamped()
            goal_pose_stamped.header.frame_id = "map"
            goal_pose_stamped.header.stamp = rospy.Time(0)
            goal_pose_stamped.pose = self.current_goals[robot_idx].pose
            try:
                self.tf_listener.waitForTransform(f"robot{robot_idx + 1}/base_footprint", "map", 
                                                 rospy.Time(0), rospy.Duration(5.0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logwarn("Robot %d: TF transform lookup failed: %s", robot_idx + 1, e)
                return False
            transformed_goal = self.tf_listener.transformPose(f"robot{robot_idx + 1}/base_footprint", 
                                                              goal_pose_stamped)
            dx = transformed_goal.pose.position.x
            dy = transformed_goal.pose.position.y
            angle_error = self.get_yaw(transformed_goal.pose.orientation)
            if (abs(dx) < self.distance_threshold_x and 
                abs(dy) < self.distance_threshold_y and 
                abs(angle_error) < self.angle_threshold):
                rospy.loginfo("Robot %d: Goal reached: dx=%f, dy=%f, angle_error=%f", 
                              robot_idx + 1, dx, dy, angle_error)
                return True
            else:
                rospy.loginfo("Robot %d: Not yet reached: dx=%f, dy=%f, angle_error=%f", 
                              robot_idx + 1, dx, dy, angle_error)
            return False
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("Robot %d: TF transform failed: %s", robot_idx + 1, e)
            return False

    def all_robots_reached_goals(self):

        if not all(self.poses_initialized):
            rospy.logwarn("Not all robot poses are initialized. Waiting for AMCL poses.")
            return False
        all_reached = True
        for i in range(self.num_robots):
            if self.current_goals[i] is not None:
                reached = self.has_reached_goal(i)
                rospy.loginfo("Robot %d: Goal check result: %s", i + 1, reached)
                if not reached:
                    all_reached = False
            else:
                rospy.logwarn("Robot %d: Current goal is None", i + 1)
                all_reached = False
        return all_reached

    def compute_and_publish(self, points_to_process, update_goals=False):

        vertices = self.get_vertices_from_poses()
        if vertices is None:
            rospy.logwarn("Cannot compute ellipsoid due to missing poses")
            return False

        ellipsoid = self.compute_ellipsoid_from_vertices(vertices)
        rospy.loginfo(f"Computed ellipsoid with center={ellipsoid.getD()}, radius={ellipsoid.getC()[0,0]}")

        
        current_obstacles = self.obstacles.copy()
        with self.dynamic_obstacle_lock:
            for obstacle in self.latest_dynamic_obstacles:
                current_obstacles.append(obstacle["coords"])
                rospy.loginfo("Added dynamic obstacle ID %d: %s", obstacle["id"], obstacle["coords"])

        A, b = compute_polytope(current_obstacles, ellipsoid, self.bounds)
        rospy.loginfo(f"Polytope A shape: {A.shape if A is not None else None}, b shape: {b.shape if b is not None else None}")
        if A is None or b is None:
            rospy.logwarn("Failed to compute polytope A and b")
            polytope_msg = Float64MultiArray()
            polytope_msg.data = []
            self.pub_polytope.publish(polytope_msg)
            return False
        else:
            polytope_msg = Float64MultiArray()
            A_rows, A_cols = A.shape if A is not None else (0, 0)
            A_flat = A.flatten() if A is not None else np.array([])
            b_flat = b.flatten() if b is not None else np.array([])
            polytope_msg.data = [float(A_rows), float(A_cols)] + A_flat.tolist() + b_flat.tolist()
            self.pub_polytope.publish(polytope_msg)

        zinit = np.array([points_to_process[0], points_to_process[1], points_to_process[2], 
                          points_to_process[3], points_to_process[4], points_to_process[5]])
        status_g, zg = formation(zinit, points_to_process[:2], A, b)

        if status_g in range(1, 7) and zg is not None:
            zg_msg = Float64MultiArray()
            zg_msg.data = zg.tolist()
            self.pub_formation.publish(zg_msg)
            rospy.loginfo(f"Published formation_goal: zg={zg.tolist()}")

            if update_goals:
                robot_poses = self.get_robot_poses(zg, self.ru, self.map_size)
                for i, (x, y, theta) in enumerate(robot_poses):
                    current_pose = self.current_poses[i]
                    if current_pose is None:
                        rospy.logwarn(f"Robot {i+1}: current pose is None, skipping intermediate goal planning.")
                        continue
                    current_yaw = self.get_yaw(current_pose.orientation)
                    angle_diff = self.angle_difference(theta, current_yaw)
                    
                    if abs(angle_diff) > np.deg2rad(20):
                        num_steps = int(np.ceil(abs(angle_diff) / np.deg2rad(10)))
                        rospy.loginfo(f"Robot {i+1}: splitting rotation into {num_steps} steps (total rotation: {np.rad2deg(angle_diff):.2f}°)")
                        for step in range(1, num_steps + 1):
                            interp_theta = current_yaw + (angle_diff / num_steps) * step
                            interp_theta = np.arctan2(np.sin(interp_theta), np.cos(interp_theta))  # normalize
                            pose_msg = self.create_pose_msg(x, y, interp_theta)
                            self.goal_queues[i].append(pose_msg)
                            rospy.loginfo(f"Robot {i+1}: Added intermediate goal at θ={np.rad2deg(interp_theta):.2f}°")
                        self.current_goals[i] = self.goal_queues[i][-1]
                        self.goal_pubs[i].publish(self.current_goals[i])
                    else:
                        pose_msg = self.create_pose_msg(x, y, theta)
                        self.goal_queues[i].append(pose_msg)
                        self.current_goals[i] = pose_msg
                        self.goal_pubs[i].publish(pose_msg)
                        rospy.loginfo(f"Published robot_{i+1}/goal: x={x}, y={y}, θ={np.rad2deg(theta):.2f}°")
                self.z_pre = zg
                self.path_index += 1
            return True
        else:
            rospy.logwarn(f"Formation failed: status={status_g}, zg={zg}")
            return False

    def run(self):

        rospy.loginfo(f"Starting motion planning with {len(self.interpolated_path)} points.")
        

        if self.path_index < len(self.interpolated_path):
            points_to_process = self.interpolated_path[self.path_index]
            zinit = np.array([points_to_process[0], points_to_process[1], points_to_process[2], 
                              points_to_process[3], points_to_process[4], points_to_process[5]])
            zg = zinit
            zg_msg = Float64MultiArray()
            zg_msg.data = zg.tolist()
            self.pub_formation.publish(zg_msg)
            rospy.loginfo(f"Published first formation_goal: zg={zg.tolist()}")

            robot_poses = self.get_robot_poses(zg, self.ru, self.map_size)
            for i, (x, y, theta) in enumerate(robot_poses):
                pose_msg = self.create_pose_msg(x, y, theta)
                self.goal_queues[i].append(pose_msg)
                self.current_goals[i] = pose_msg
                self.goal_pubs[i].publish(pose_msg)
                rospy.loginfo(f"Published robot_{i+1}/goal: x={x}, y={y}, theta={theta}")
            self.z_pre = zg
            self.path_index += 1
            self.is_first_goal = False

            rospy.loginfo("All robots have received their first goals. Waiting for user input (Enter)...")
            input("Press Enter to continue to next goals...")

        while not rospy.is_shutdown():

            for i in range(self.num_robots):
                if self.current_goals[i] is not None:
                    self.goal_pubs[i].publish(self.current_goals[i])

            if not self.is_first_goal and self.all_robots_reached_goals():
                for i in range(self.num_robots):
                    self.current_goals[i] = None
                if self.path_index < len(self.interpolated_path):
                    rospy.loginfo("All robots reached goals, computing next point...")
                    points_to_process = self.interpolated_path[self.path_index]
                    if self.compute_and_publish(points_to_process, update_goals=True):
                        pass
                    else:
                        self.path_index += 1
                else:
                    rospy.loginfo("All points processed.")
                    break

            self.rate.sleep()

if __name__ == '__main__':
    try:
        planner = CombinedMotionPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Combined motion planner node terminated.")