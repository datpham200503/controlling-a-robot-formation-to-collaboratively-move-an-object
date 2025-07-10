#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import json
import rospy
import time
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
import threading
from tf.transformations import euler_from_quaternion
from scipy.optimize import minimize


global_json_path = '/home/dat/catkin_ws/src/global_path_planning/config/global.json'
global_path_json_path = '/home/dat/catkin_ws/src/global_path_planning/config/global_path.json'

with open(global_json_path, 'r') as f:
    global_data = json.load(f)

map_size = global_data['map']
object_radius = global_data['object_radius']
a_i = global_data['robot_shape']['a_i']
l_r = global_data['robot_shape']['l_r']
w_r = global_data['robot_shape']['w_r']

with open(global_path_json_path, 'r') as f:
    path_data = json.load(f)

ru = [
    object_radius * np.cos(0.0), object_radius * np.sin(0.0),
    object_radius * np.cos(2 * np.pi / 3), object_radius * np.sin(2 * np.pi / 3),
    object_radius * np.cos(4 * np.pi / 3), object_radius * np.sin(4 * np.pi / 3),
    a_i, a_i, a_i,
    l_r, w_r
]


latest_poses = [None, None, None]
latest_goals = [None, None, None]
latest_formation_goal = None
lock = threading.Lock()


tx_trail = []  
ty_trail = []  
tx_goal_trail = [] 
ty_goal_trail = [] 
theta_trail = []  
theta_goal_trail = []  
time_trail = []

robot_x_trails = [[], [], []]  
robot_y_trails = [[], [], []]  
robot_x_goal_trails = [[], [], []]  
robot_y_goal_trails = [[], [], []]  
robot_theta_trails = [[], [], []]  
robot_theta_goal_trails = [[], [], []]  
robot_time_trails = [[], [], []]

MAX_TRAIL_LENGTH = 100
start_time = time.time()

def amcl_pose_callback(msg, robot_id):
    global latest_poses
    with lock:
        latest_poses[robot_id - 1] = msg.pose.pose

def robot_goal_callback(msg, robot_id):
    global latest_goals
    with lock:
        latest_goals[robot_id - 1] = msg.pose

def invert_formation(x_centers, y_centers, thetas_i, ru):
    l_r, w_r = ru[9], ru[10]
    x_g_list, y_g_list = [], []
    for i in range(3):
        x_c, y_c = x_centers[i], y_centers[i]  
        theta_i = thetas_i[i]
        a_i = ru[6 + i]  
        x_g = x_c - (a_i + l_r/2) * np.cos(theta_i)
        y_g = y_c - (a_i + l_r/2) * np.sin(theta_i)
        x_g_list.append(x_g)
        y_g_list.append(y_g)

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
    if not res.success:
        rospy.logwarn("Optimization in invert_formation failed!")
        return None
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
    for pose in poses:
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

    z = invert_formation(x_centers, y_centers, thetas_i, ru)
    return z

def formation_goal_callback(msg):
    global latest_formation_goal
    if len(msg.data) >= 3:
        with lock:
            latest_formation_goal = [msg.data[0], msg.data[1], msg.data[2]]

def compute_robot_metrics():
    metrics = []
    with lock:
        for i in range(3):
            if latest_poses[i] is None or latest_goals[i] is None:
                metrics.append((None, None, None, None, None, None))
                continue
            

            actual_pose = latest_poses[i]
            x_actual = actual_pose.position.x
            y_actual = actual_pose.position.y
            actual_quat = (actual_pose.orientation.x, actual_pose.orientation.y,
                         actual_pose.orientation.z, actual_pose.orientation.w)
            _, _, actual_yaw = euler_from_quaternion(actual_quat)
            

            goal_pose = latest_goals[i]
            x_goal = goal_pose.position.x
            y_goal = goal_pose.position.y
            goal_quat = (goal_pose.orientation.x, goal_pose.orientation.y,
                       goal_pose.orientation.z, goal_pose.orientation.w)
            _, _, goal_yaw = euler_from_quaternion(goal_quat)
            
            metrics.append((x_actual, y_actual, actual_yaw, x_goal, y_goal, goal_yaw))
    return metrics

def main():
    global tx_trail, ty_trail, tx_goal_trail, ty_goal_trail, theta_trail, theta_goal_trail, time_trail
    global robot_x_trails, robot_y_trails, robot_x_goal_trails, robot_y_goal_trails
    global robot_theta_trails, robot_theta_goal_trails, robot_time_trails
    
    rospy.init_node('formation_visualizer', anonymous=True)
    

    rospy.Subscriber('/robot1/amcl_pose', PoseWithCovarianceStamped, lambda msg: amcl_pose_callback(msg, 1), queue_size=10)
    rospy.Subscriber('/robot2/amcl_pose', PoseWithCovarianceStamped, lambda msg: amcl_pose_callback(msg, 2), queue_size=10)
    rospy.Subscriber('/robot3/amcl_pose', PoseWithCovarianceStamped, lambda msg: amcl_pose_callback(msg, 3), queue_size=10)
    rospy.Subscriber('/robot1/goal', PoseStamped, lambda msg: robot_goal_callback(msg, 1), queue_size=10)
    rospy.Subscriber('/robot2/goal', PoseStamped, lambda msg: robot_goal_callback(msg, 2), queue_size=10)
    rospy.Subscriber('/robot3/goal', PoseStamped, lambda msg: robot_goal_callback(msg, 3), queue_size=10)
    rospy.Subscriber('/formation_goal', Float64MultiArray, formation_goal_callback, queue_size=10)


    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3)
    

    ax_tx = fig.add_subplot(gs[0, 0])
    ax_ty = fig.add_subplot(gs[0, 1])
    

    ax_theta_formation = fig.add_subplot(gs[0, 2])
    ax_theta_robot1 = fig.add_subplot(gs[1, 0])
    ax_theta_robot2 = fig.add_subplot(gs[1, 1])
    ax_theta_robot3 = fig.add_subplot(gs[1, 2])
    

    ax_robot1_x = fig.add_subplot(gs[2, 0])
    ax_robot1_y = fig.add_subplot(gs[2, 1])
    ax_robot2_x = fig.add_subplot(gs[2, 2])
    ax_robot2_y = fig.add_subplot(gs[3, 0])
    ax_robot3_x = fig.add_subplot(gs[3, 1])
    ax_robot3_y = fig.add_subplot(gs[3, 2])
    
    plt.ion()
    fig.suptitle("Formation and Robot Metrics Visualization")
    fig.canvas.manager.set_window_title('Formation Monitoring Dashboard')
    

    plt.tight_layout(pad=6.0)  
    fig.subplots_adjust(top=0.9, hspace=0.6, wspace=0.6)  
    
    plt.show()

    rate = rospy.Rate(5)  
    while not rospy.is_shutdown():
        current_time = time.time() - start_time
        
        z = compute_formation_params()
        robot_metrics = compute_robot_metrics()


        with lock:
            if z is not None and latest_formation_goal is not None and len(latest_formation_goal) >= 3:

                tx_trail.append(z[0])
                ty_trail.append(z[1])
                tx_goal_trail.append(latest_formation_goal[0])
                ty_goal_trail.append(latest_formation_goal[1])
                theta_trail.append(z[2])
                theta_goal_trail.append(latest_formation_goal[2])
                time_trail.append(current_time)
                

                if len(time_trail) > MAX_TRAIL_LENGTH:
                    time_trail.pop(0)
                    tx_trail.pop(0)
                    ty_trail.pop(0)
                    tx_goal_trail.pop(0)
                    ty_goal_trail.pop(0)
                    theta_trail.pop(0)
                    theta_goal_trail.pop(0)
                

                ax_tx.clear()
                ax_ty.clear()
                ax_theta_formation.clear()
                

                ax_tx.set_title("Formation X Position")
                ax_tx.set_xlabel("Time (s)")
                ax_tx.set_ylabel("X (m)")
                ax_tx.grid(True)
                ax_tx.plot(time_trail, tx_trail, 'b-', label='Formation x')
                ax_tx.plot(time_trail, tx_goal_trail, 'r--', label='Goal x')
                ax_tx.legend()
                
                ax_ty.set_title("Formation Y Position")
                ax_ty.set_xlabel("Time (s)")
                ax_ty.set_ylabel("Y (m)")
                ax_ty.grid(True)
                ax_ty.plot(time_trail, ty_trail, 'g-', label='Formation y')
                ax_ty.plot(time_trail, ty_goal_trail, 'm--', label='Goal y')
                ax_ty.legend()
                
                ax_theta_formation.set_title("Formation Orientation")
                ax_theta_formation.set_xlabel("Time (s)")
                ax_theta_formation.set_ylabel("θ (rad)")
                ax_theta_formation.grid(True)
                ax_theta_formation.plot(time_trail, theta_trail, 'k-', label='Formation θ')
                ax_theta_formation.plot(time_trail, theta_goal_trail, 'r--', label='Goal θ')
                ax_theta_formation.legend()


        for i in range(3):
            if robot_metrics[i][0] is not None:

                robot_time_trails[i].append(current_time)
                robot_x_trails[i].append(robot_metrics[i][0])
                robot_y_trails[i].append(robot_metrics[i][1])
                robot_x_goal_trails[i].append(robot_metrics[i][3])
                robot_y_goal_trails[i].append(robot_metrics[i][4])
                robot_theta_trails[i].append(robot_metrics[i][2])
                robot_theta_goal_trails[i].append(robot_metrics[i][5])
                

                if len(robot_time_trails[i]) > MAX_TRAIL_LENGTH:
                    robot_time_trails[i].pop(0)
                    robot_x_trails[i].pop(0)
                    robot_y_trails[i].pop(0)
                    robot_x_goal_trails[i].pop(0)
                    robot_y_goal_trails[i].pop(0)
                    robot_theta_trails[i].pop(0)
                    robot_theta_goal_trails[i].pop(0)
                

                if i == 0:
                    ax_x = ax_robot1_x
                    ax_y = ax_robot1_y
                    ax_theta = ax_theta_robot1
                elif i == 1:
                    ax_x = ax_robot2_x
                    ax_y = ax_robot2_y
                    ax_theta = ax_theta_robot2
                else:
                    ax_x = ax_robot3_x
                    ax_y = ax_robot3_y
                    ax_theta = ax_theta_robot3
                

                ax_x.clear()
                ax_y.clear()
                ax_theta.clear()
                

                ax_x.set_title(f"Robot {i+1} X Position")
                ax_x.set_xlabel("Time (s)")
                ax_x.set_ylabel("X (m)")
                ax_x.grid(True)
                ax_x.plot(robot_time_trails[i], robot_x_trails[i], 'b-', label=f'Robot {i+1} X')
                ax_x.plot(robot_time_trails[i], robot_x_goal_trails[i], 'r--', label=f'Goal X')
                ax_x.legend()
                
                ax_y.set_title(f"Robot {i+1} Y Position")
                ax_y.set_xlabel("Time (s)")
                ax_y.set_ylabel("Y (m)")
                ax_y.grid(True)
                ax_y.plot(robot_time_trails[i], robot_y_trails[i], 'g-', label=f'Robot {i+1} Y')
                ax_y.plot(robot_time_trails[i], robot_y_goal_trails[i], 'm--', label=f'Goal Y')
                ax_y.legend()
                
                ax_theta.set_title(f"Robot {i+1} Orientation")
                ax_theta.set_xlabel("Time (s)")
                ax_theta.set_ylabel("θ (rad)")
                ax_theta.grid(True)
                ax_theta.plot(robot_time_trails[i], robot_theta_trails[i], 'k-', label=f'Robot {i+1} θ')
                ax_theta.plot(robot_time_trails[i], robot_theta_goal_trails[i], 'r--', label='Goal θ')
                ax_theta.legend()

        plt.pause(0.01)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Formation visualizer node terminated.")