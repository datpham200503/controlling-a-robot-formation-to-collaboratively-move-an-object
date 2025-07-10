#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped
import tf
import math

class RobotPoseController:
    def __init__(self):
        rospy.init_node('pose_controller', anonymous=True)

        self.linear_speed_x = rospy.get_param('~linear_speed_x', 0.05)
        self.linear_speed_y = rospy.get_param('~linear_speed_y', 0.05)
        self.angular_speed = rospy.get_param('~angular_speed', 0.05)
        self.distance_threshold_x = rospy.get_param('~distance_threshold_x', 0.05)
        self.distance_threshold_y = rospy.get_param('~distance_threshold_y', 0.05)
        self.angle_threshold = rospy.get_param('~angle_threshold', 0.1)

        self.linear_kp = rospy.get_param('~linear_kp', 0.5)
        self.linear_ki = rospy.get_param('~linear_ki', 0.0)
        self.linear_kd = rospy.get_param('~linear_kd', 0.0)

        self.angular_kp = rospy.get_param('~angular_kp', 0.5)
        self.angular_ki = rospy.get_param('~angular_ki', 0.00)
        self.angular_kd = rospy.get_param('~angular_kd', 0.0)

        self.current_pose = None
        self.target_pose = None

        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.amcl_pose_sub = rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, self.amcl_pose_callback)
        self.goal_pose_sub = rospy.Subscriber('goal', PoseStamped, self.goal_pose_callback)

        self.rate = rospy.Rate(5)  # 5 Hz

        self.error_sum_x = 0.0
        self.last_error_x = 0.0
        self.error_sum_y = 0.0
        self.last_error_y = 0.0
        self.error_sum_theta = 0.0
        self.last_error_theta = 0.0

        self.tf_listener = tf.TransformListener()

    def amcl_pose_callback(self, msg):
        self.current_pose = msg.pose.pose
        
        #self.now_pose_msg.header = msg.header
        #self.now_pose_msg.pose = msg.pose
        

    def goal_pose_callback(self, msg):
        self.target_pose = msg.pose

    def get_yaw(self, quaternion):
        try:
            euler = tf.transformations.euler_from_quaternion(
                [quaternion.x, quaternion.y, quaternion.z, quaternion.w])
            return euler[2]
        except:
            rospy.logerr("Invalid quaternion input")
            return 0.0

    def calculate_velocity(self):
        cmd_vel = Twist()

        if self.current_pose is None or self.target_pose is None:
            return cmd_vel

        try:
            # Convert target pose to base_footprint frame
            target_pose_stamped = PoseStamped()
            target_pose_stamped.header.frame_id = "map"
            target_pose_stamped.header.stamp = rospy.Time(0)
            target_pose_stamped.pose = self.target_pose

            transformed_target = self.tf_listener.transformPose("robot1/base_footprint", target_pose_stamped)

            dx = transformed_target.pose.position.x
            dy = transformed_target.pose.position.y
            angle_error = self.get_yaw(transformed_target.pose.orientation)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("TF transform failed: %s", e)
            return cmd_vel

        # distance = math.sqrt(dx ** 2 + dy ** 2)
        if abs(dx) < self.distance_threshold_x and abs(dy) < self.distance_threshold_y and abs(angle_error) < self.angle_threshold:
            #self.now_pose_pub.publish(self.now_pose_msg)
            return cmd_vel

        dt = 1.0 / 5

        self.error_sum_x += dx * dt
        derivative_x = (dx - self.last_error_x) / dt
        cmd_vel.linear.x = self.linear_kp * dx + self.linear_ki * self.error_sum_x + self.linear_kd * derivative_x
        self.last_error_x = dx
        cmd_vel.linear.x = max(min(cmd_vel.linear.x, self.linear_speed_x), -self.linear_speed_x)

        self.error_sum_y += dy * dt
        derivative_y = (dy - self.last_error_y) / dt
        cmd_vel.linear.y = self.linear_kp * dy + self.linear_ki * self.error_sum_y + self.linear_kd * derivative_y
        self.last_error_y = dy
        cmd_vel.linear.y = max(min(cmd_vel.linear.y, self.linear_speed_y), -self.linear_speed_y)

        self.error_sum_theta += angle_error * dt
        derivative_theta = (angle_error - self.last_error_theta) / dt
        cmd_vel.angular.z = self.angular_kp * angle_error + self.angular_ki * self.error_sum_theta + self.angular_kd * derivative_theta
        self.last_error_theta = angle_error
        cmd_vel.angular.z = max(min(cmd_vel.angular.z, self.angular_speed), -self.angular_speed)

        rospy.loginfo("cmd_vel: %s", cmd_vel)
        rospy.loginfo("dx: %s, dy: %s, angle_error: %s", dx, dy, angle_error)

        return cmd_vel

    def run(self):
        while not rospy.is_shutdown():
            cmd_vel = self.calculate_velocity()
            self.cmd_vel_pub.publish(cmd_vel)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        controller = RobotPoseController()
        controller.run()
    except rospy.ROSInterruptException:
        pass


