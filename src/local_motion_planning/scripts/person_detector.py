#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import Float64MultiArray
from matplotlib.patches import Polygon as MplPolygon

class DynamicObstacleGenerator:
    def __init__(self):
        rospy.init_node('dynamic_obstacle_generator', anonymous=True)
        self.rate = rospy.Rate(1)  # 1 Hz

        self.obstacle_pub = rospy.Publisher('/dynamic_obstacles', Float64MultiArray, queue_size=10)

        # Thay vì lấy từ người dùng, fix cứng 4 góc ở đây
        self.corners = [
            (1.0, 1.0),
            (3.0, 1.0),
            (3.0, 3.0),
            (1.0, 3.0)
        ]

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Dynamic Obstacle (Outside Region)')
        self.ax.grid(True)
        self.ax.set_aspect('equal')

    def points_to_constraints(self, corners):
        corners = np.array(corners)
        A = np.zeros((4, 2))
        b = np.zeros(4)

        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i+1)%4]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])  # Rotate 90 degrees CCW (inward)
            normal = normal / np.linalg.norm(normal)
            center = np.mean(corners, axis=0)
            if np.dot(normal, center - p1) < 0:
                normal = -normal  # Ensure inward
            # For outside region: flip normal (outward) and adjust b
            A[i] = normal
            b[i] = np.dot(normal, p1)

        return A, b

    def process_and_publish(self):
        if self.corners is None:
            return

        A, b = self.points_to_constraints(self.corners)
        rospy.loginfo("Generated constraints: A=%s, b=%s", A, b)

        msg = []
        msg.extend(A.flatten().tolist())
        msg.extend(b.tolist())
        obstacle_msg = Float64MultiArray(data=msg)
        self.obstacle_pub.publish(obstacle_msg)
        rospy.loginfo("Published to /dynamic_obstacles: %s", msg)

    def run(self):
        plt.ion()
        while not rospy.is_shutdown():
            self.process_and_publish()
            plt.show(block=False)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        generator = DynamicObstacleGenerator()
        generator.run()
    except rospy.ROSInterruptException:
        plt.close()
        pass
