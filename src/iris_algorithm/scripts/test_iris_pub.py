#!/home/dat/env/bin/python
import rospy
import irispy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

def test_random_obstacles_2d(pub=None):
    bounds = irispy.Polyhedron.from_bounds([0, 0], [1, 1])
    
    # Define two fixed obstacles
    obstacles = [
        # Obstacle 1
        np.array([
            [0.2, 0.4, 0.4, 0.2],  # x coordinates
            [0.2, 0.2, 0.4, 0.4]   # y coordinates
        ]),
        # Obstacle 2
        np.array([
            [0.8, 1.0, 1.0, 0.8],  # x coordinates
            [0.8, 0.8, 1.0, 1.0]   # y coordinates
        ])
    ]
    
    start = np.array([0.75, 0.25])

    region, debug = irispy.inflate_region(obstacles, start, bounds=bounds, return_debug_data=True)

    # Log and publish the region's polytope inequalities (A and b)
    A, b = None, None
    try:
        polyhedron = region.getPolyhedron()
        A = polyhedron.getA()  # Matrix A (n_l x 2)
        b = polyhedron.getB()  # Vector b (n_l x 1)
        rospy.loginfo("Region inequalities: A=\n%s\nb=%s", A, b)

        if pub is not None:
            # Prepare Float64MultiArray for A
            msg_A = Float64MultiArray()
            A_flat = A.flatten()  # Flatten A to 1D array
            msg_A.layout.dim = [
                MultiArrayDimension(label="rows", size=A.shape[0], stride=A.shape[0] * A.shape[1]),
                MultiArrayDimension(label="cols", size=A.shape[1], stride=A.shape[1])
            ]
            msg_A.data = A_flat.tolist()

            # Prepare Float64MultiArray for b
            msg_b = Float64MultiArray()
            msg_b.layout.dim = [
                MultiArrayDimension(label="rows", size=b.shape[0], stride=b.shape[0])
            ]
            msg_b.data = b.tolist()

            # Publish A and b
            pub.publish(msg_A)
            pub.publish(msg_b)
            rospy.loginfo("Published A and b to /iris_polytope")

    except AttributeError:
        rospy.loginfo("Region: %s (could not access A and b)", region)

    return A, b

def publish_polytope_callback(event, pub, A, b):
    if A is not None and b is not None and pub is not None:
        # Prepare Float64MultiArray for A
        msg_A = Float64MultiArray()
        A_flat = A.flatten()
        msg_A.layout.dim = [
            MultiArrayDimension(label="rows", size=A.shape[0], stride=A.shape[0] * A.shape[1]),
            MultiArrayDimension(label="cols", size=A.shape[1], stride=A.shape[1])
        ]
        msg_A.data = A_flat.tolist()

        # Prepare Float64MultiArray for b
        msg_b = Float64MultiArray()
        msg_b.layout.dim = [
            MultiArrayDimension(label="rows", size=b.shape[0], stride=b.shape[0])
        ]
        msg_b.data = b.tolist()

        # Publish A and b
        pub.publish(msg_A)
        pub.publish(msg_b)
        rospy.loginfo("Published A and b to /iris_polytope (periodic)")

def irispy_node():
    rospy.init_node('irispy_node', anonymous=True)
    # Create publisher for A and b
    pub = rospy.Publisher('/iris_polytope', Float64MultiArray, queue_size=10)
    rospy.loginfo("IRISpy node started")
    try:
        A, b = test_random_obstacles_2d(pub=pub)
        # Set up timer to publish every 2 seconds
        if A is not None and b is not None:
            rospy.Timer(rospy.Duration(2), lambda event: publish_polytope_callback(event, pub, A, b))
    except rospy.ROSInterruptException:
        rospy.logerr("Node interrupted")
    except Exception as e:
        rospy.logerr("Error: %s", str(e))
    
    rospy.spin()

if __name__ == '__main__':
    irispy_node()