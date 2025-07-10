#!/usr/bin/env python

import rospy
import math
import tf
from nav_msgs.msg import Odometry
from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion

class OdomPublisher:
    def __init__(self):
        # Robot parameters
        self.Lx = 0.289  # Half robot length (m)
        self.Ly = 0.210  # Half robot width (m)
        self.R = 0.0485  # Wheel radius (m)
        self.PULSES_PER_REVOLUTION = 3000  # Pulses per wheel revolution (adjust if needed)
        self.interval_pub = 0.4  # 400ms expected interval

        # Position and orientation
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.orientation_q = [0, 0, 0, 1]  # Initial quaternion

        # Encoder previous values
        self.prev_pul_left_st_0 = 0
        self.prev_pul_right_st_1 = 0
        self.prev_pul_left_nd_2 = 0
        self.prev_pul_right_nd_3 = 0
        self.first_reading = True  # Flag for first encoder reading

        # ROS node initialization
        rospy.init_node('odom_publisher', anonymous=True)
        self.odom_pub = rospy.Publisher('odom', Odometry, queue_size=10)
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.odom = Odometry()
        # Subscribers
        rospy.Subscriber('/robot1/encoder', Int16MultiArray, self.encoder_callback)
        rospy.Subscriber('/robot1/imu/data', Imu, self.imu_callback)

        # Transform message
        self.t = TransformStamped()
        self.t.header.frame_id = "robot1/odom"
        self.t.child_frame_id = "robot1/base_footprint"

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def calculate_distance(self, current_pul, prev_pul):
        # Calculate tick difference with overflow handling
        ticks = current_pul - prev_pul
        if ticks > 10000:
            ticks = -(65535 - ticks)
        elif ticks < -10000:
            ticks = 65535 - abs(ticks)
        # Calculate distance
        distance = (ticks * 2 * math.pi * self.R) / self.PULSES_PER_REVOLUTION
        return distance

    def imu_callback(self, msg):
        # Store IMU quaternion
        self.orientation_q = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]

    def encoder_callback(self, msg):
        # Get encoder pulses
        pul_left_st_0 = msg.data[0]  # Front-left
        pul_right_st_1 = msg.data[1]  # Front-right
        pul_left_nd_2 = msg.data[2]  # Rear-left
        pul_right_nd_3 = msg.data[3]  # Rear-right

        # Skip first reading to initialize previous values
        if self.first_reading:
            self.prev_pul_left_st_0 = pul_left_st_0
            self.prev_pul_right_st_1 = pul_right_st_1
            self.prev_pul_left_nd_2 = pul_left_nd_2
            self.prev_pul_right_nd_3 = pul_right_nd_3
            self.first_reading = False
            return

        # Calculate distances with overflow handling
        distance_left_st_0 = self.calculate_distance(pul_left_st_0, self.prev_pul_left_st_0)
        distance_right_st_1 = self.calculate_distance(pul_right_st_1, self.prev_pul_right_st_1)
        distance_left_nd_2 = self.calculate_distance(pul_left_nd_2, self.prev_pul_left_nd_2)
        distance_right_nd_3 = self.calculate_distance(pul_right_nd_3, self.prev_pul_right_nd_3)

        # Update previous pulse counts
        self.prev_pul_left_st_0 = pul_left_st_0
        self.prev_pul_right_st_1 = pul_right_st_1
        self.prev_pul_left_nd_2 = pul_left_nd_2
        self.prev_pul_right_nd_3 = pul_right_nd_3

        # Calculate velocities
        current_time = rospy.Time.now().to_sec()
        if hasattr(self, 'last_time'):
            delta_t = current_time - self.last_time
        else:
            delta_t = self.interval_pub
        self.last_time = current_time

        del_x = (distance_left_st_0 + distance_right_st_1 + distance_left_nd_2 + distance_right_nd_3) / 4.0
        del_y = (-distance_left_st_0 + distance_right_st_1 + distance_left_nd_2 - distance_right_nd_3) / 4.0
        del_z = (-distance_left_st_0 + distance_right_st_1 - distance_left_nd_2 + distance_right_nd_3) / (4.0 * (self.Lx + self.Ly))

        # Get yaw from IMU
        roll, pitch, yaw_imu = euler_from_quaternion(self.orientation_q)
        self.yaw = self.normalize_angle(yaw_imu)

        # Update position in global frame
        self.x += (del_x * math.cos(self.yaw) - del_y * math.sin(self.yaw))
        self.y += (del_x * math.sin(self.yaw) + del_y * math.cos(self.yaw))

        # Publish odometry
        self.odom.header.stamp = rospy.Time.now()
        self.odom.header.frame_id = "robot1/odom"
        self.odom.child_frame_id = "robot1/base_footprint"
        self.odom.pose.pose.position.x = self.x
        self.odom.pose.pose.position.y = self.y
        self.odom.pose.pose.position.z = 0
        q = quaternion_from_euler(0, 0, self.yaw)
        self.odom.pose.pose.orientation.x = q[0]
        self.odom.pose.pose.orientation.y = q[1]
        self.odom.pose.pose.orientation.z = q[2]
        self.odom.pose.pose.orientation.w = q[3]
        self.odom.twist.twist.linear.x = del_x / delta_t
        self.odom.twist.twist.linear.y = del_y / delta_t
        self.odom.twist.twist.angular.z = del_z / delta_t
        self.odom.pose.covariance = [
            0.01, 0, 0, 0, 0, 0,
            0, 0.01, 0, 0, 0, 0,
            0, 0, 999, 0, 0, 0,
            0, 0, 0, 999, 0, 0,
            0, 0, 0, 0, 999, 0,
            0, 0, 0, 0, 0, 0.1
        ]
        self.odom.twist.covariance = [
            0.1, 0, 0, 0, 0, 0,
            0, 0.1, 0, 0, 0, 0,
            0, 0, 999, 0, 0, 0,
            0, 0, 0, 999, 0, 0,
            0, 0, 0, 0, 999, 0,
            0, 0, 0, 0, 0, 0.2
        ]
        self.odom_pub.publish(self.odom)

        # Publish TF
        self.t.header = self.odom.header
        self.t.transform.translation.x = self.x
        self.t.transform.translation.y = self.y
        self.t.transform.translation.z = 0
        self.t.transform.rotation.x = q[0]
        self.t.transform.rotation.y = q[1]
        self.t.transform.rotation.z = q[2]
        self.t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(
            (self.t.transform.translation.x, self.t.transform.translation.y, self.t.transform.translation.z),
            (self.t.transform.rotation.x, self.t.transform.rotation.y, self.t.transform.rotation.z, self.t.transform.rotation.w),
            self.t.header.stamp,
            self.t.child_frame_id,
            self.t.header.frame_id
        )

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        odom_publisher = OdomPublisher()
        odom_publisher.run()
    except rospy.ROSInterruptException:
        pass
