#!/usr/bin/env python

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

def move_base_callback(x):
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    if x==1:
        goal.target_pose.pose.position.x = 0.27
        goal.target_pose.pose.position.y = 1.97
        goal.target_pose.pose.orientation.z = 0.889
        goal.target_pose.pose.orientation.w = 0.456
    elif x==2:
        goal.target_pose.pose.position.x = 1.55
        goal.target_pose.pose.position.y = 2.12
        goal.target_pose.pose.orientation.z = -0.034
        goal.target_pose.pose.orientation.w = 0.99
    else:
        goal.target_pose.pose.position.x = -0.102
        goal.target_pose.pose.position.y = 0.122
        goal.target_pose.pose.orientation.z = -0.028
        goal.target_pose.pose.orientation.w = 0.999

    client.send_goal(goal)
    wait = client.wait_for_result()

    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")
    else:
        return client.get_result()
    
if __name__=="__main__":
    try:
        rospy.init_node('movebase_client')
        result = move_base_callback(2)
        rospy.loginfo("1 Goal execution done!")
        rospy.sleep(5)
        result = move_base_callback(1)
        rospy.loginfo("2 Goal execution done!")
        rospy.sleep(5)
        result = move_base_callback(3)
        rospy.loginfo("Home Goal execution done!")
        if result:
            rospy.loginfo("Goal execution done!" + str(result))

    except rospy.ROSInterruptException:
        rospy.loginfo("Navigaion test finished.")