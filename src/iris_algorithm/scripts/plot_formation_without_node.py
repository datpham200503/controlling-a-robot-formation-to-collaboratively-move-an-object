import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def compute_formation_vertices(z, ru, robot_dims):
    t_x, t_y, theta = z[0], z[1], z[2]
    theta_1, theta_2, theta_3 = z[3], z[4], z[5]
    l_r, w_r = robot_dims[0], robot_dims[1]

    three_angles = [theta, 2 * np.pi / 3 + theta, 4 * np.pi / 3 + theta]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    vertices = []

    # Tính 3 đỉnh của tam giác vật thể
    for i in range(3):
        x_local = ru[2 * i]
        y_local = ru[2 * i + 1]
        x_global = t_x + cos_theta * x_local - sin_theta * y_local
        y_global = t_y + sin_theta * x_local + cos_theta * y_local
        vertices.append((x_global, y_global))

    # Tính các đỉnh robot (3 robot, mỗi robot 4 đỉnh)
    for i in range(3):
        theta_i = z[3 + i] + three_angles[i]
        cos_theta_i = np.cos(theta_i)
        sin_theta_i = np.sin(theta_i)
        x_g, y_g = vertices[i]
        a_i = ru[6 + i]

        # Tính tâm robot
        x_center = x_g + (a_i + l_r / 2) * cos_theta_i
        y_center = y_g + (a_i + l_r / 2) * sin_theta_i

        # 4 đỉnh robot (hình chữ nhật)
        local_corners = [(l_r/2, w_r/2), (-l_r/2, w_r/2), (-l_r/2, -w_r/2), (l_r/2, -w_r/2)]
        for x_local, y_local in local_corners:
            x_rotated = x_center + cos_theta_i * x_local - sin_theta_i * y_local
            y_rotated = y_center + sin_theta_i * x_local + cos_theta_i * y_local
            vertices.append((x_rotated, y_rotated))
    
    return vertices

def plot_formation(vertices, A, b, obstacles=None):
    fig, ax = plt.subplots()
    
    # Vẽ tam giác vật thể
    triangle = Polygon(vertices[:3], closed=True, edgecolor='blue', facecolor='blue', label='Object', zorder=3)
    ax.add_patch(triangle)

    # Vẽ các robot
    for i in range(3):
        start = 3 + 4 * i
        robot_vertices = vertices[start:start + 4]
        rect = Polygon(robot_vertices, closed=True, edgecolor='green', facecolor='green', label=f'Robot' if i == 0 else "", zorder=3)
        ax.add_patch(rect)

    # Vẽ các ràng buộc tuyến tính (Ax <= b)
    x = np.linspace(-1, 6, 400)
    y = np.linspace(-1, 6, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.ones_like(X)

    for i in range(len(A)):
        Z *= (A[i, 0] * X + A[i, 1] * Y <= b[i])

    ax.contourf(X, Y, Z, levels=[0.5, 1], colors=['#ccc'], alpha=0.5)

    if obstacles is not None:
        for idx, obs in enumerate(obstacles):
            obs_coords = list(zip(obs[0], obs[1]))
            label = 'Obstacle' if idx == 0 else None  # Chỉ label cho obstacle đầu tiên
            polygon = Polygon(obs_coords, closed=True, edgecolor='black', facecolor='gray',
                            label=label, zorder=2)
            ax.add_patch(polygon)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("Robot Formation with Obstacles and Feasible Region")
    plt.grid(True)
    plt.show()


# Dữ liệu đầu vào
x = [3.0, 1.0, 0.0, 0.0, 0.0, 0.0]  # t_x, t_y, theta, theta_1, theta_2, theta_3
ru = [
    0.15 * np.cos(0.0), 0.15 * np.sin(0.0),
    0.15 * np.cos(2 * np.pi / 3), 0.15 * np.sin(2 * np.pi / 3),
    0.15 * np.cos(4 * np.pi / 3), 0.15 * np.sin(4 * np.pi / 3),
    0.2, 0.2, 0.2,  # a1, a2, a3
    0.3, 0.3       # l_r, w_r
]
robot_dims = ru[9:11]  # [l_r, w_r]

# Ma trận ràng buộc
A = np.array([
    [-0.63227049, -0.77474772],
    [0.38237612, 0.92400677],
    [1.0, 0.0],
    [0.0, 1.0],
    [-1.0, 0.0],
    [0.0, -1.0]
])
b = np.array([-1.38252855, 3.92711138, 5.0, 5.0, 0.0, 0.0])

# Khai báo vật cản
obstacles = [
    np.array([
        [0.5, 0.9, 0.9, 0.5],  # x
        [0.75, 0.75, 1.05, 1.05]  # y
    ]),
    np.array([
        [2.9, 3.3, 3.3, 2.9],
        [3.05, 3.05, 3.35, 3.35]
    ])
]

# Vẽ hình
vertices = compute_formation_vertices(x, ru, robot_dims)
plot_formation(vertices, A, b, obstacles)
