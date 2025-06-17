#!/usr/bin/env python

import rospy
import tf
import math
import copy
from sensor_msgs.msg import LaserScan

OTHER_ROBOTS = ["robot1/base_link", "robot2/base_link", "robot3/base_link"]
ROBOT_RADIUS = 0.3  


RECTANGULAR_REGIONS = [
    (2.6, 3.1, 0.7, 1.3),  
    (1.4, 1.9, 2.55, 3.2),   
    (-0.5, 5, -0.5, 0.5),
    (-0.5, 5, 3.5, 4.0),
    (4.5, 5, -0.5, 4.0),
    (-0.5, 0.8, -0.5, 4.0),
]


class LidarFilter:
    def __init__(self):
        rospy.init_node('filtered_lidar_node')
        self.tf_listener = tf.TransformListener()

        self.scan_sub = rospy.Subscriber("scan_multi", LaserScan, self.scan_callback)
        self.scan_pub = rospy.Publisher("/scan_filtered", LaserScan, queue_size=1)

        rospy.loginfo("Nút lọc LiDAR đã khởi động")
        rospy.spin()

    def is_point_in_rectangle(self, x, y, rect):

        x_min, x_max, y_min, y_max = rect
        return x_min <= x <= x_max and y_min <= y <= y_max

    def scan_callback(self, msg):

        filtered_scan = copy.deepcopy(msg)
        filtered_scan.ranges = list(msg.ranges)  

        robot_positions = []

        for frame in OTHER_ROBOTS:
            try:
                t = self.tf_listener.getLatestCommonTime(msg.header.frame_id, frame)
                (trans, rot) = self.tf_listener.lookupTransform(msg.header.frame_id, frame, t)
                robot_positions.append((trans[0], trans[1], trans[2]))
            except (tf.Exception, tf.LookupException, tf.ConnectivityException) as e:
                rospy.logwarn("Không thể lấy TF của %s: %s", frame, str(e))
                continue

        try:
            t = self.tf_listener.getLatestCommonTime("map", msg.header.frame_id)
            (trans_map, rot_map) = self.tf_listener.lookupTransform("map", msg.header.frame_id, t)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException) as e:
            rospy.logwarn("Không thể lấy TF từ %s sang map: %s", msg.header.frame_id, str(e))
            trans_map, rot_map = None, None


        angle = msg.angle_min
        for i, r in enumerate(msg.ranges):
            if math.isinf(r) or math.isnan(r):
                angle += msg.angle_increment
                continue

            x_base = r * math.cos(angle)
            y_base = r * math.sin(angle)


            for (rx, ry, rz) in robot_positions:
                distance = math.hypot(x_base - rx, y_base - ry)
                if distance < ROBOT_RADIUS:
                    filtered_scan.ranges[i] = float('inf') 
                    break


            if trans_map and rot_map and not math.isinf(filtered_scan.ranges[i]):

                quaternion = (rot_map[0], rot_map[1], rot_map[2], rot_map[3])
                euler = tf.transformations.euler_from_quaternion(quaternion)
                yaw = euler[2]  
                x_map = trans_map[0] + x_base * math.cos(yaw) - y_base * math.sin(yaw)
                y_map = trans_map[1] + x_base * math.sin(yaw) + y_base * math.cos(yaw)


                for rect in RECTANGULAR_REGIONS:
                    if self.is_point_in_rectangle(x_map, y_map, rect):
                        filtered_scan.ranges[i] = float('inf')  
                        break

            angle += msg.angle_increment

        self.scan_pub.publish(filtered_scan)

if __name__ == "__main__":
    try:
        LidarFilter()
    except rospy.ROSInterruptException:
        pass