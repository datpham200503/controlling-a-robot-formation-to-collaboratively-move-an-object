#!/home/dat/env/bin/python
import rospy
import irispy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
from scipy.spatial.distance import cdist

def create_rectangle(center, angle, width=0.2, height=0.2):

    points = np.array([
        [-width/2, -height/2],
        [width/2, -height/2],
        [width/2, height/2],
        [-width/2, height/2]
    ]).T
    

    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    

    points = rotation @ points + center[:, np.newaxis]
    return points

def check_overlap(center1, center2, width=0.3, height=0.2, min_distance=0.4):

    distance = np.linalg.norm(center1 - center2)

    diagonal = np.sqrt(width**2 + height**2)
    return distance > min_distance + diagonal

def test_random_obstacles_2d(show=False):
    bounds = irispy.Polyhedron.from_bounds([0, 0], [1, 1])
    

    obstacles = []
    width, height = 0.3, 0.2
    min_distance = 0.4 
    

    center1 = np.random.uniform([0.2, 0.2], [0.8, 0.8])
    angle1 = np.random.uniform(0, 2*np.pi)
    obstacles.append(create_rectangle(center1, angle1, width, height))
    

    center2 = np.random.uniform([0.2, 0.2], [0.8, 0.8])
    angle2 = np.random.uniform(0, 2*np.pi)
    obstacles.append(create_rectangle(center2, angle2, width, height))

    start = np.array([0.5, 0.5])


    region, debug = irispy.inflate_region(obstacles, start, bounds=bounds, return_debug_data=True)


    try:
        polyhedron = region.getPolyhedron()
        A = polyhedron.getA()
        b = polyhedron.getB()
        rospy.loginfo("Region inequalities: A=\n%s\nb=%s", A, b)
    except AttributeError:
        rospy.loginfo("Region: %s (could not access A and b)", region)
        A, b = None, None


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


    polyhedron.draw2d(ax=ax1)
    region.getEllipsoid().draw2d(ax=ax1)
    
    for obstacle in obstacles:
        points = list(zip(obstacle[0, :], obstacle[1, :]))
        polygon = Polygon(points, facecolor='gray', edgecolor='black', zorder=1)
        ax1.add_patch(polygon)
    
    ax1.scatter(start[0], start[1], color='green', s=50, zorder=10)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.grid(True)
    ax1.set_title('IRIS Region and Obstacles')


    if A is not None and b is not None:

        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x, y)
        points = np.vstack([X.ravel(), Y.ravel()]).T
        Z = np.all(A @ points.T <= b[:, np.newaxis], axis=0).reshape(X.shape)
        

        ax2.contourf(X, Y, Z, levels=[0.5, 1], colors=['green'], alpha=0.3)
        

        x = np.linspace(0, 1, 100)
        for i in range(len(b)):
            if abs(A[i, 1]) > 1e-6:
                y = (b[i] - A[i, 0] * x) / A[i, 1]
                ax2.plot(x, y, 'b-')
            else:
                x_val = b[i] / A[i, 0] if abs(A[i, 0]) > 1e-6 else 0
                ax2.axvline(x=x_val, color='b', linestyle='-')
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.0])
        ax2.grid(True)
        ax2.set_title('Polyhedron Constraints (Ax <= b)')

    if show:
        plt.tight_layout()
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