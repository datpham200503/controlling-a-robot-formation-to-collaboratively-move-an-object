#!/home/dat/env/bin/python
import rospy
import irispy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def test_random_obstacles_2d(show=False):
    bounds = irispy.Polyhedron.from_bounds([0, 0], [5, 5])
    
    # Define two fixed obstacles
    obstacles = [
        # Obstacle 1
        np.array([
            [0.5, 0.9, 0.9, 0.5],  # x coordinates
            [0.75, 0.75, 1.05, 1.05]   # y coordinates
        ]),
        # Obstacle 2
        np.array([
            [2.9, 3.3, 3.3, 2.9],  # x coordinates
            [3.05, 3.05, 3.35, 3.35]   # y coordinates
        ]),
        np.array([
            [3.0 + 0.5, 3.0 + 0.9, 3.0 + 0.9, 3.0 + 0.5],  # x coordinates
            [0.75, 0.75, 1.05, 1.05]   # y coordinates
        ]),
    ]
    
    start = np.array([3.0, 1.0])

    region, debug = irispy.inflate_region(obstacles, start, bounds=bounds, return_debug_data=True)

    try:
        polyhedron = region.getPolyhedron()
        A = polyhedron.getA()
        b = polyhedron.getB()
        rospy.loginfo("Region inequalities: A=\n%s\nb=%s", A, b)
    except AttributeError:
        rospy.loginfo("Region: %s (could not access A and b)", region)

    # Draw the polytope, ellipsoid, and obstacles
    polyhedron.draw2d()
    region.getEllipsoid().draw2d()
    
    for obstacle in obstacles:
        points = list(zip(obstacle[0, :], obstacle[1, :]))
        polygon = Polygon(points, facecolor='gray', edgecolor='black', zorder=1)
        plt.gca().add_patch(polygon)
    
    plt.gca().set_xlim([0.0, 5.0])
    plt.gca().set_ylim([0.0, 5.0])

    plt.scatter(start[0], start[1], color='green', s=50, zorder=10)
    plt.gca().set_xlim([0.0, 5.0])
    plt.gca().set_ylim([0.0, 5.0])

    if show:
        plt.grid(True)
        plt.show()

def irispy_node():
    rospy.init_node('irispy_node', anonymous=True)
    rospy.loginfo("IRISpy node started")
    try:
        test_random_obstacles_2d(show=True)
    except rospy.ROSInterruptException:
        rospy.logerr("Node interrupted")
    except Exception as e:
        rospy.logerr("Error: %s", str(e))

if __name__ == '__main__':
    irispy_node()