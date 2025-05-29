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

        # Initialize variables
        self.current_pose = None
        self.current_goal = None
        self.goal_queue = deque()  # Queue to store incoming goals
        self.last_goal_time = rospy.Time.now()

        # TF listener for coordinate transformations
        self.tf_listener = tf.TransformListener()

        # Subscribers
        self.amcl_pose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_pose_callback)
        self.goal_pose_sub = rospy.Subscriber('/robot_1/goal_pose', PoseStamped, self.goal_pose_callback)

        # Publisher for the current goal
        self.goal_pub = rospy.Publisher('/goal', PoseStamped, queue_size=10)

    def amcl_pose_callback(self, msg):
        """Callback for AMCL pose updates."""
        self.current_pose = msg.pose.pose

    def goal_pose_callback(self, msg):
        """Callback for new goal pose."""
        current_time = rospy.Time.now()
        # If a new goal arrives and the current goal is still being processed, add to queue
        if self.current_goal is not None:
            self.goal_queue.append(msg)
            rospy.loginfo("New goal received, added to queue. Queue size: %d", len(self.goal_queue))
        else:
            self.current_goal = msg
            self.last_goal_time = current_time
            rospy.loginfo("New goal set: x=%f, y=%f", msg.pose.position.x, msg.pose.position.y)
            # Publish the new goal to /goal topic
            self.goal_pub.publish(msg)

    def get_yaw(self, quaternion):
        """Convert quaternion to yaw angle."""
        try:
            euler = tf.transformations.euler_from_quaternion(
                [quaternion.x, quaternion.y, quaternion.z, quaternion.w])
            return euler[2]
        except Exception as e:
            rospy.logerr("Invalid quaternion input: %s", e)
            return 0.0

    def has_reached_goal(self):
        """Check if the robot has reached the current goal."""
        if self.current_pose is None or self.current_goal is None:
            return False

        try:
            # Transform goal pose to base_footprint frame
            goal_pose_stamped = PoseStamped()
            goal_pose_stamped.header.frame_id = "map"
            goal_pose_stamped.header.stamp = rospy.Time(0)
            goal_pose_stamped.pose = self.current_goal.pose

            transformed_goal = self.tf_listener.transformPose("base_footprint", goal_pose_stamped)

            dx = transformed_goal.pose.position.x
            dy = transformed_goal.pose.position.y
            angle_error = self.get_yaw(transformed_goal.pose.orientation)

            # Check if robot is within thresholds
            if (abs(dx) < self.distance_threshold_x and 
                abs(dy) < self.distance_threshold_y and 
                abs(angle_error) < self.angle_threshold):
                rospy.loginfo("Goal reached: dx=%f, dy=%f, angle_error=%f", dx, dy, angle_error)
                return True
            return False

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("TF transform failed: %s", e)
            return False

    def run(self):
        """Main loop running at 5Hz."""
        while not rospy.is_shutdown():
            if self.current_goal is not None:
                # Publish the current goal to /goal topic
                self.goal_pub.publish(self.current_goal)
                
                # Check if robot has reached the current goal
                if self.has_reached_goal():
                    # Clear current goal and process next goal in queue
                    self.current_goal = None
                    if self.goal_queue:
                        self.current_goal = self.goal_queue.popleft()
                        self.last_goal_time = rospy.Time.now()
                        rospy.loginfo("Processing next goal from queue: x=%f, y=%f", 
                                      self.current_goal.pose.position.x, 
                                      self.current_goal.pose.position.y)
                        # Publish the new goal to /goal topic
                        self.goal_pub.publish(self.current_goal)

            self.rate.sleep()

if __name__ == '__main__':
    try:
        manager = RobotGoalManager()
        manager.run()
    except rospy.ROSInterruptException:
        pass