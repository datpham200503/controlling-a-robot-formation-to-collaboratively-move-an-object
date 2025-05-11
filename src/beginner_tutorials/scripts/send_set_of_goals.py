#!/usr/bin/env python

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import Int32MultiArray

# Các tọa độ và góc để di chuyển
x = [0.01, 1.5, 1.71, 0.11, -0.5, 0.7]
y = [-0.5, -0.5, 1.09, 1.78, 0.5, 0.54]

x_pose = []
y_pose = []

path_received = False

def send_goals():
    global x_pose, y_pose, path_received

    rospy.init_node('navigation_goals')
    # Tạo client cho move_base
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    rospy.Subscriber('path', Int32MultiArray, callback)

    # Chờ cho server sẵn sàng
    rospy.loginfo("Waiting for move_base action server...")
    client.wait_for_server()   

    while not rospy.is_shutdown():
        # Kiểm tra nếu path đã được nhận
        if not path_received:
            rospy.loginfo("Waiting for path data...")
            rospy.sleep(1)
            continue

        # Gửi các mục tiêu sau khi nhận path
        for i in range(len(x_pose)):
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()

            # Thiết lập vị trí và hướng của robot
            goal.target_pose.pose.position.x = x_pose[i]
            goal.target_pose.pose.position.y = y_pose[i]
            goal.target_pose.pose.orientation.w = 1.0  # Giữ hướng ổn định (quaternion w = 1)

            # Gửi mục tiêu đến move_base
            rospy.loginfo(f"Sending goal {i+1}")
            client.send_goal(goal)

            # Chờ đợi kết quả
            client.wait_for_result()

            # Kiểm tra trạng thái
            if client.get_state() == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo(f"Hooray, the base moved to goal {i+1}")
            else:
                rospy.loginfo(f"The base failed to move to goal {i+1} for some reason")

        # Sau khi gửi hết các mục tiêu, reset lại dữ liệu
        rospy.loginfo("Finished current path. Waiting for new path data...")
        path_received = False
        x_pose = []  # Xóa lại danh sách vị trí
        y_pose = []  # Xóa lại danh sách vị trí
        rospy.sleep(1)  # Chờ một chút trước khi tiếp tục nhận dữ liệu mới

def callback(data):
    global path_received, x_pose, y_pose

    if path_received:
        rospy.logwarn("Path already received. Ignoring further updates.")
        return

    rospy.loginfo(f"Received path data: {data.data}")
    array_data = data.data

    for point in array_data:
        x_pose.append(x[point-1])
        y_pose.append(y[point-1])

    # Đánh dấu là đã nhận path
    path_received = True

if __name__ == '__main__':
    try:
        send_goals()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
