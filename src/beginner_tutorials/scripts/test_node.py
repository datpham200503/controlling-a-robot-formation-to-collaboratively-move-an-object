#!/usr/bin/env python
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Int32MultiArray


class MoveBaseSeq:

    def __init__(self):
        rospy.init_node('move_base_sequence')
        
        # Đăng ký subscriber cho topic 'path'
        rospy.Subscriber('path', Int32MultiArray, self.callback)

        # Thiết lập các điểm chuẩn
        self.points_setup = [[0.01, -0.5, 0.0], [1.5, -0.5, 0.0], [1.71, 1.09, 0.0], [0.11, 1.78, 0.0], [-0.5, 0.5, 0.0], [0.7, 0.54, 0.0]]
        self.pose_seq = []  # Danh sách các tọa độ mục tiêu
        self.goal_cnt = 0
        self.quaternion = Quaternion(0.0, 0.0, 0, 1)
        self.path_received = False  # Cờ kiểm tra xem đã nhận được path chưa
        self.goal_processing = False  # Cờ kiểm tra trạng thái của action client

        # Tạo action client
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        if not self.client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("Action server not available!")
            rospy.signal_shutdown("Action server not available!")
            return
        rospy.loginfo("Connected to move base server.")
        rospy.loginfo("Waiting for path data...")

    def callback(self, data):
        if self.path_received:
            rospy.logwarn("Path already received. Ignoring further updates.")
            return

        rospy.loginfo(f"Received path data: {data.data}")
        array_data = data.data

        # Giảm 1 đơn vị cho từng chỉ số trong array_data
        adjusted_data = [i - 1 for i in array_data]

        # Lọc các chỉ số không hợp lệ
        valid_indices = [i for i in adjusted_data if 0 <= i < len(self.points_setup)]
        if len(valid_indices) < len(adjusted_data):
            rospy.logwarn("Some indices are out of range and will be ignored.")

        # Tạo danh sách tọa độ mục tiêu từ các chỉ số hợp lệ
        try:
            points_seq = [self.points_setup[i] for i in valid_indices]
            rospy.loginfo(f"Points sequence: {points_seq}")

            # Tạo pose_seq từ points_seq
            self.pose_seq = [Pose(Point(*point), self.quaternion) for point in points_seq]
            rospy.loginfo(f"Pose sequence: {self.pose_seq}")

            # Đặt cờ và bắt đầu navigation
            if self.pose_seq:
                self.path_received = True
                rospy.loginfo("Path data processed. Starting navigation...")
                self.movebase_client()
            else:
                rospy.logerr("No valid poses generated. Navigation cannot start.")
        except Exception as e:
            rospy.logerr(f"Error in callback processing: {e}")

    def active_cb(self):
        rospy.loginfo(f"Goal pose {self.goal_cnt+1} is now being processed by the Action Server...")

    def feedback_cb(self, feedback):
        rospy.loginfo(f"Feedback for goal pose {self.goal_cnt+1} received")

    def done_cb(self, status, result):
        # Kiểm tra trạng thái Action Client để tránh gọi lại nhiều lần
        if self.goal_processing:
            rospy.logwarn(f"Goal pose {self.goal_cnt+1} received DONE twice.")
            return

        self.goal_processing = True

        if status == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo(f"Goal pose {self.goal_cnt+1} reached successfully.")
        else:
            rospy.logwarn(f"Goal pose {self.goal_cnt+1} failed with status {status}.")
        
        self.goal_cnt += 1
        if self.goal_cnt < len(self.pose_seq):
            self.send_next_goal()
        else:
            rospy.loginfo("All goal poses reached!")
            rospy.signal_shutdown("All goal poses reached!")

    def movebase_client(self):
        if not self.pose_seq:
            rospy.logerr("Pose sequence is empty. Cannot send goals.")
            rospy.signal_shutdown("Pose sequence is empty.")
            return
        self.send_next_goal()

    def send_next_goal(self):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = self.pose_seq[self.goal_cnt]
        rospy.loginfo(f"Sending goal pose {self.goal_cnt+1} to Action Server")
        rospy.loginfo(str(self.pose_seq[self.goal_cnt]))
        self.client.send_goal(goal, self.done_cb, self.active_cb, self.feedback_cb)


if __name__ == '__main__':
    try:
        MoveBaseSeq()
        rospy.spin()  # Duy trì node chờ dữ liệu từ callback
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation finished.")
