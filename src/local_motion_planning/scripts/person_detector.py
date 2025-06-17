#!/home/dat/env/bin/python
import rospy
import irispy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Thông số robot từ plot_local_path.py
object_radius = 0.15
a_i = 0.15
l_r = 0.3  # Chiều dài robot
w_r = 0.3  # Chiều rộng robot
ru = [
    object_radius * np.cos(0.0), object_radius * np.sin(0.0),
    object_radius * np.cos(2 * np.pi / 3), object_radius * np.sin(2 * np.pi / 3),
    object_radius * np.cos(4 * np.pi / 3), object_radius * np.sin(4 * np.pi / 3),
    a_i, a_i, a_i,
    l_r, w_r
]
robot_dims = [l_r, w_r]

def create_rectangle(x_min, y_min, x_max, y_max):
    # Tạo 4 đỉnh của hình chữ nhật từ x_min, y_min, x_max, y_max
    points = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ]).T
    return points

def compute_formation_vertices(z, ru, robot_dims):
    # Hàm tính toán các đỉnh của đội hình robot từ plot_local_path.py
    t_x, t_y, theta = z[0], z[1], z[2]
    l_r, w_r = robot_dims
    three_angles = [theta, 2 * np.pi / 3 + theta, 4 * np.pi / 3 + theta]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    vertices = []

    # Tính toán 3 đỉnh của tam giác đội hình
    for i in range(3):
        x_local = ru[2 * i]
        y_local = ru[2 * i + 1]
        x_global = t_x + cos_theta * x_local - sin_theta * y_local
        y_global = t_y + sin_theta * x_local + cos_theta * y_local
        vertices.append((x_global, y_global))

    # Tính toán các đỉnh của 3 robot
    for i in range(3):
        theta_i = z[3 + i] + three_angles[i]
        cos_theta_i = np.cos(theta_i)
        sin_theta_i = np.sin(theta_i)
        x_g, y_g = vertices[i]
        a_i = ru[6 + i]
        x_center = x_g + (a_i + l_r / 2) * cos_theta_i
        y_center = y_g + (a_i + l_r / 2) * sin_theta_i
        local_corners = [(l_r / 2, w_r / 2), (-l_r / 2, w_r / 2), (-l_r / 2, -w_r / 2), (l_r / 2, -w_r / 2)]
        for x_local, y_local in local_corners:
            x_rotated = x_center + cos_theta_i * x_local - sin_theta_i * y_local
            y_rotated = y_center + sin_theta_i * x_local + cos_theta_i * y_local
            vertices.append((x_rotated, y_rotated))
    return vertices

def test_random_obstacles_2d(show=False):
    bounds = irispy.Polyhedron.from_bounds([0, 0], [3, 3])
    
    # Tạo hai vật cản hình chữ nhật với tọa độ xác định
    obstacles = []
    
    # Vật cản 1: (0.5,0.5,1,1)
    obstacles.append(create_rectangle(0.5, 0.5, 1, 1))
    
    # Vật cản 2: (2.5,2,3,2.5)
    obstacles.append(create_rectangle(2.5, 2, 3, 2.5))

    start = np.array([2.25, 0.75])

    # Tính toán vùng IRIS
    region, debug = irispy.inflate_region(obstacles, start, bounds=bounds, return_debug_data=True)

    # Lấy ma trận A và b từ đa diện
    try:
        polyhedron = region.getPolyhedron()
        A = polyhedron.getA()
        b = polyhedron.getB()
        rospy.loginfo("Region inequalities: A=\n%s\nb=%s", A, b)
    except AttributeError:
        rospy.loginfo("Region: %s (could not access A and b)", region)
        A, b = None, None

    # Tạo figure với một subplot
    fig, ax = plt.subplots(figsize=(6, 5))

    # Vẽ đa diện và vật cản
    polyhedron.draw2d(ax=ax)
    
    for obstacle in obstacles:
        points = list(zip(obstacle[0, :], obstacle[1, :]))
        polygon = Polygon(points, facecolor='gray', edgecolor='black', zorder=1)
        ax.add_patch(polygon)
    
    # Vẽ điểm tại start
    ax.scatter(start[0], start[1], color='green', s=50, zorder=10)

    # Vẽ đội hình robot tại điểm start với góc quay = 0
    z_start = [start[0], start[1], 0.0, 0.0, 0.0, 0.0]  # t_x, t_y, theta, theta1, theta2, theta3
    verts = compute_formation_vertices(z_start, ru, robot_dims)
    # Vẽ tam giác đội hình
    # ax.add_patch(Polygon(verts[:3], closed=True, edgecolor='blue', facecolor='blue', alpha=0.5))
    # Vẽ 3 robot
    # for i in range(3):
    #     start_idx = 3 + 4 * i
    #     robot_vertices = verts[start_idx:start_idx + 4]
    #     ax.add_patch(Polygon(robot_vertices, closed=True, edgecolor='blue', facecolor='blue', alpha=0.5))
    #     # Thêm nhãn số cho robot
    #     x_center = sum(v[0] for v in robot_vertices) / 4
    #     y_center = sum(v[1] for v in robot_vertices) / 4
    #     ax.text(x_center, y_center, str(i + 1), color='white', fontsize=8, ha='center', va='center', weight='bold')

    # Vẽ đội hình robot thứ hai tại tọa độ (1.75, 2.25) với góc quay = 0
    z_goal = [2.0, 2.0, 0.0, -0.65, 0.0, 0.0]  # t_x, t_y, theta, theta1, theta2, theta3
    verts_goal = compute_formation_vertices(z_goal, ru, robot_dims)
    # Vẽ tam giác đội hình
    # ax.add_patch(Polygon(verts_goal[:3], closed=True, edgecolor='orange', facecolor='orange', alpha=0.5))
    # Vẽ 3 robot
    # for i in range(3):
    #     start_idx = 3 + 4 * i
    #     robot_vertices = verts_goal[start_idx:start_idx + 4]
    #     ax.add_patch(Polygon(robot_vertices, closed=True, edgecolor='orange', facecolor='orange', alpha=0.5))
    #     # Thêm nhãn số cho robot
    #     x_center = sum(v[0] for v in robot_vertices) / 4
    #     y_center = sum(v[1] for v in robot_vertices) / 4
    #     ax.text(x_center, y_center, str(i + 1), color='white', fontsize=8, ha='center', va='center', weight='bold')

    # Vẽ điểm tại trung tâm đội hình thứ hai
    # ax.scatter(z_goal[0], z_goal[1], color='green', s=50, zorder=10)

    ax.set_xlim([0.0, 3.0])
    ax.set_ylim([0.0, 3.0])
    ax.grid(True)
    ax.set_title('IRIS Region, Obstacles, and Robot Formations')

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