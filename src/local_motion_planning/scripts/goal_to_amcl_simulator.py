#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
import numpy as np
import tf2_ros
import tf

class GoalToAmclSimulator:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('goal_to_amcl_simulator', anonymous=True)
        self.rate = rospy.Rate(5)  # 5 Hz for main loop

        # Number of robots
        self.num_robots = 3

        # Store latest goals for each robot
        self.latest_goals = [None] * self.num_robots

        # Track first goal for each robot
        self.first_goal_received = [False] * self.num_robots

        # Publishers for AMCL poses
        self.amcl_pubs = [
            rospy.Publisher(f'/robot{i+1}/amcl_pose', PoseWithCovarianceStamped, queue_size=10)
            for i in range(self.num_robots)
        ]

        # Subscribers for goals
        self.goal_subs = []
        for i in range(self.num_robots):
            robot_id = i + 1
            self.goal_subs.append(
                rospy.Subscriber(f'/robot{robot_id}/goal', PoseStamped,
                                 lambda msg, idx=i: self.goal_callback(msg, idx))
            )

        # TF broadcaster for publishing transforms
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Covariance matrix for AMCL pose (small values for simulation)
        self.covariance = np.diag([0.01, 0.01, 0.0, 0.0, 0.0, 0.001]).flatten().tolist()

        # Track last publish time
        self.last_publish_time = rospy.Time.now()

    def goal_callback(self, msg, robot_idx):
        """Callback for goal messages."""
        self.latest_goals[robot_idx] = msg
        rospy.loginfo("Robot %d: Received goal at x=%f, y=%f, z=%f",
                      robot_idx + 1, msg.pose.position.x, msg.pose.position.y,
                      msg.pose.orientation.z)
        # Publish immediately for first goal
        if not self.first_goal_received[robot_idx]:
            amcl_msg = self.create_amcl_pose_msg(msg)
            self.amcl_pubs[robot_idx].publish(amcl_msg)
            rospy.loginfo("Robot %d: Published initial AMCL pose at x=%f, y=%f",
                          robot_idx + 1, amcl_msg.pose.pose.position.x,
                          amcl_msg.pose.pose.position.y)
            transform_msg = self.create_transform_msg(msg, robot_idx)
            self.tf_broadcaster.sendTransform(transform_msg)
            rospy.loginfo("Robot %d: Published initial TF transform from map to robot%d/base_footprint",
                          robot_idx + 1, robot_idx + 1)
            self.first_goal_received[robot_idx] = True

    def create_amcl_pose_msg(self, goal_pose):
        """Create PoseWithCovarianceStamped from PoseStamped."""
        amcl_msg = PoseWithCovarianceStamped()
        amcl_msg.header.stamp = rospy.Time.now()
        amcl_msg.header.frame_id = "map"
        amcl_msg.pose.pose = goal_pose.pose
        amcl_msg.pose.covariance = self.covariance
        return amcl_msg

    def create_transform_msg(self, goal_pose, robot_idx):
        """Create TransformStamped from PoseStamped for map -> robot{i}/base_footprint."""
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "map"
        transform.child_frame_id = f"robot{robot_idx + 1}/base_footprint"
        transform.transform.translation.x = goal_pose.pose.position.x
        transform.transform.translation.y = goal_pose.pose.position.y
        transform.transform.translation.z = 0.0
        transform.transform.rotation = goal_pose.pose.orientation
        return transform

    def run(self):
        """Main loop to publish AMCL poses and TF transforms every 5 seconds."""
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            # Check if 5 seconds have passed since last publish
            if (current_time - self.last_publish_time).to_sec() >= 5.0:
                for i in range(self.num_robots):
                    if self.latest_goals[i] is not None:
                        # Publish AMCL pose
                        amcl_msg = self.create_amcl_pose_msg(self.latest_goals[i])
                        self.amcl_pubs[i].publish(amcl_msg)
                        rospy.loginfo("Robot %d: Published AMCL pose at x=%f, y=%f",
                                      i + 1, amcl_msg.pose.pose.position.x,
                                      amcl_msg.pose.pose.position.y)

                        # Publish TF transform
                        transform_msg = self.create_transform_msg(self.latest_goals[i], i)
                        self.tf_broadcaster.sendTransform(transform_msg)
                        rospy.loginfo("Robot %d: Published TF transform from map to robot%d/base_footprint",
                                      i + 1, i + 1)
                self.last_publish_time = current_time
            self.rate.sleep()

if __name__ == '__main__':
    try:
        simulator = GoalToAmclSimulator()
        simulator.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Goal to AMCL simulator node terminated.")