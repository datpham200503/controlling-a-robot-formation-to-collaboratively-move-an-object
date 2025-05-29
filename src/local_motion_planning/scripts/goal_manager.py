#!/home/dat/env/bin/python

import rospy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import tf
import math
from collections import deque

class RobotGoalManager:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('robot_goal_manager', anonymous=True)
        self.rate = rospy.Rate(5)  # 5Hz

        # Parameters for goal checking (similar to cal_robot_move.py)
        self.distance_threshold_x = rospy.get_param('~distance_threshold_x', 0.05)
        self.distance_threshold_y = rospy.get_param('~distance_threshold_y', 0.05)
        self.angle_threshold = rospy.get_param('~angle_threshold', 0.1)

        # Initialize variables for each robot
        self.num_robots = 3
        self.current_poses = [None] * self.num_robots
        self.current_goals = [None] * self.num_robots
        self.goal_queues = [deque() for _ in range(self.num_robots)]
        self.last_goal_times = [rospy.Time.now() for _ in range(self.num_robots)]

        # TF listener for coordinate transformations
        self.tf_listener = tf.TransformListener()

        # Subscribers for each robot
        self.amcl_pose_subs = []
        self.goal_pose_subs = []
        for i in range(self.num_robots):
            robot_id = i + 1
            # Subscribe to /robot_i/amcl_pose
            self.amcl_pose_subs.append(
                rospy.Subscriber(f'/robot_{robot_id}/amcl_pose', 
                               PoseWithCovarianceStamped, 
                               lambda msg, idx=i: self.amcl_pose_callback(msg, idx))
            )
            # Subscribe to /robot_i/goal_pose
            self.goal_pose_subs.append(
                rospy.Subscriber(f'/robot_{robot_id}/goal_pose', 
                               PoseStamped, 
                               lambda msg, idx=i: self.goal_pose_callback(msg, idx))
            )

        # Publishers for current goals of each robot
        self.goal_pubs = [rospy.Publisher(f'/robot_{i+1}/goal', PoseStamped, queue_size=10) 
                         for i in range(self.num_robots)]

    def amcl_pose_callback(self, msg, robot_idx):
        """Callback for AMCL pose updates of a specific robot."""
        self.current_poses[robot_idx] = msg.pose.pose

    def goal_pose_callback(self, msg, robot_idx):
        """Callback for new goal pose of a specific robot."""
        current_time = rospy.Time.now()
        # If a current goal is being processed, add new goal to queue
        if self.current_goals[robot_idx] is not None:
            self.goal_queues[robot_idx].append(msg)
            rospy.loginfo("Robot %d: New goal received, added to queue. Queue size: %d", 
                         robot_idx + 1, len(self.goal_queues[robot_idx]))
        else:
            self.current_goals[robot_idx] = msg
            self.last_goal_times[robot_idx] = current_time
            rospy.loginfo("Robot %d: New goal set: x=%f, y=%f", 
                         robot_idx + 1, msg.pose.position.x, msg.pose.position.y)
            # Publish new goal to /robot_i/goal topic
            self.goal_pubs[robot_idx].publish(msg)

    def get_yaw(self, quaternion):
        """Convert quaternion to yaw angle."""
        try:
            euler = tf.transformations.euler_from_quaternion(
                [quaternion.x, quaternion.y, quaternion.z, quaternion.w])
            return euler[2]
        except Exception as e:
            rospy.logerr("Quaternion conversion error: %s", e)
            return 0.0

    def has_reached_goal(self, robot_idx):
        """Check if the specified robot has reached its current goal."""
        if self.current_poses[robot_idx] is None or self.current_goals[robot_idx] is None:
            return False

        try:
            # Transform goal pose to base_footprint frame
            goal_pose_stamped = PoseStamped()
            goal_pose_stamped.header.frame_id = "map"
            goal_pose_stamped.header.stamp = rospy.Time(0)
            goal_pose_stamped.pose = self.current_goals[robot_idx].pose

            # Assume base_footprint frame for each robot is /robot_i/base_footprint
            transformed_goal = self.tf_listener.transformPose(f"robot_{robot_idx + 1}/base_footprint", 
                                                            goal_pose_stamped)

            dx = transformed_goal.pose.position.x
            dy = transformed_goal.pose.position.y
            angle_error = self.get_yaw(transformed_goal.pose.orientation)

            # Check if robot is within thresholds
            if (abs(dx) < self.distance_threshold_x and 
                abs(dy) < self.distance_threshold_y and 
                abs(angle_error) < self.angle_threshold):
                rospy.loginfo("Robot %d: Goal reached: dx=%f, dy=%f, angle_error=%f", 
                             robot_idx + 1, dx, dy, angle_error)
                return True
            return False

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("Robot %d: TF transform failed: %s", robot_idx + 1, e)
            return False

    def all_robots_reached_goals(self):
        """Check if all robots have reached their current goals."""
        for i in range(self.num_robots):
            if self.current_goals[i] is not None and not self.has_reached_goal(i):
                return False
        return True

    def run(self):
        """Main loop running at 5Hz."""
        while not rospy.is_shutdown():
            # Publish current goals to /robot_i/goal topics
            for i in range(self.num_robots):
                if self.current_goals[i] is not None:
                    self.goal_pubs[i].publish(self.current_goals[i])

            # Process new goals only when all robots have reached their current goals
            if self.all_robots_reached_goals():
                for i in range(self.num_robots):
                    if self.current_goals[i] is not None:
                        # Clear current goal
                        self.current_goals[i] = None
                        # Get next goal from queue if available
                        if self.goal_queues[i]:
                            self.current_goals[i] = self.goal_queues[i].popleft()
                            self.last_goal_times[i] = rospy.Time.now()
                            rospy.loginfo("Robot %d: Processing next goal from queue: x=%f, y=%f", 
                                         i + 1, 
                                         self.current_goals[i].pose.position.x, 
                                         self.current_goals[i].pose.position.y)
                            # Publish new goal to /robot_i/goal
                            self.goal_pubs[i].publish(self.current_goals[i])

            self.rate.sleep()

if __name__ == '__main__':
    try:
        manager = RobotGoalManager()
        manager.run()
    except rospy.ROSInterruptException:
        pass