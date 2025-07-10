#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import Tkinter as tk  # Python 2 dùng Tkinter viết hoa
from threading import Lock

class CmdVelArmGUI:
    def __init__(self):
        rospy.init_node('cmd_vel_arm_gui', anonymous=True)

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/robot1/cmd_vel', Twist, queue_size=10)
        self.setup_pub = rospy.Publisher('/robot1/setup', Bool, queue_size=10)
        self.gripper_pub = rospy.Publisher('/robot1/gripper', Bool, queue_size=10)

        # Cmd Vel variables
        self.linear_x = 0.0
        self.linear_y = 0.0
        self.angular_z = 0.0

        # Arm variables
        self.setup_state = False
        self.gripper_state = False

        self.lock = Lock()

        # GUI
        self.root = tk.Tk()
        self.root.title("Robot Control GUI")

        # CMD VEL UI
        self.label_x = tk.Label(self.root, text="Linear X: %.2f m/s" % self.linear_x)
        self.label_x.pack()
        self.slider_x = tk.Scale(self.root, from_=-0.1, to=0.1, resolution=0.005,
                                 orient=tk.HORIZONTAL, command=self.update_x,
                                 length=300, width=20, sliderlength=40)
        self.slider_x.pack()

        self.label_y = tk.Label(self.root, text="Linear Y: %.2f m/s" % self.linear_y)
        self.label_y.pack()
        self.slider_y = tk.Scale(self.root, from_=-0.1, to=0.1, resolution=0.005,
                                 orient=tk.HORIZONTAL, command=self.update_y,
                                 length=300, width=20, sliderlength=40)
        self.slider_y.pack()

        self.label_z = tk.Label(self.root, text="Angular Z: %.2f rad/s" % self.angular_z)
        self.label_z.pack()
        self.slider_z = tk.Scale(self.root, from_=-0.3, to=0.3, resolution=0.0001,
                                 orient=tk.HORIZONTAL, command=self.update_z,
                                 length=300, width=20, sliderlength=40)
        self.slider_z.pack()

        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop)
        self.stop_button.pack(pady=10)

        # Arm Control Buttons
        self.setup_button = tk.Button(self.root, text="Set Up: %s" % self.setup_state, command=self.toggle_setup)
        self.setup_button.pack(pady=10)

        self.gripper_button = tk.Button(self.root, text="Gripper: %s" % self.gripper_state, command=self.toggle_gripper)
        self.gripper_button.pack(pady=10)

        # ROS Timer
        rospy.Timer(rospy.Duration(0.5), self.publish_data)  # 2 Hz

    # Update slider values
    def update_x(self, value):
        with self.lock:
            self.linear_x = float(value)
            self.label_x.config(text="Linear X: %.2f m/s" % self.linear_x)

    def update_y(self, value):
        with self.lock:
            self.linear_y = float(value)
            self.label_y.config(text="Linear Y: %.2f m/s" % self.linear_y)

    def update_z(self, value):
        with self.lock:
            self.angular_z = float(value)
            self.label_z.config(text="Angular Z: %.2f rad/s" % self.angular_z)

    def stop(self):
        with self.lock:
            self.linear_x = 0.0
            self.linear_y = 0.0
            self.angular_z = 0.0
            self.slider_x.set(0.0)
            self.slider_y.set(0.0)
            self.slider_z.set(0.0)
            self.label_x.config(text="Linear X: 0.00 m/s")
            self.label_y.config(text="Linear Y: 0.00 m/s")
            self.label_z.config(text="Angular Z: 0.00 rad/s")

    def toggle_setup(self):
        with self.lock:
            self.setup_state = not self.setup_state
            self.setup_button.config(text="Set Up: %s" % self.setup_state)
            self.setup_pub.publish(Bool(self.setup_state))

    def toggle_gripper(self):
        with self.lock:
            self.gripper_state = not self.gripper_state
            self.gripper_button.config(text="Gripper: %s" % self.gripper_state)
            self.gripper_pub.publish(Bool(self.gripper_state))

    def publish_data(self, event):
        with self.lock:
            twist = Twist()
            twist.linear.x = self.linear_x
            twist.linear.y = self.linear_y
            twist.angular.z = self.angular_z
            self.cmd_vel_pub.publish(twist)

    def run(self):
        self.root.mainloop()

    def shutdown(self):
        with self.lock:
            self.stop()
            self.setup_pub.publish(Bool(False))
            self.gripper_pub.publish(Bool(False))
            self.cmd_vel_pub.publish(Twist())
        self.root.quit()


if __name__ == '__main__':
    try:
        gui = CmdVelArmGUI()
        rospy.on_shutdown(gui.shutdown)
        gui.run()
    except rospy.ROSInterruptException:
        pass

