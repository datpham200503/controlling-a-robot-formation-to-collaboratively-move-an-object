import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def compute_formation_vertices(z, ru, robot_dims):
    t_x, t_y, theta = z[0], z[1], z[2]
    l_r, w_r = robot_dims[0], robot_dims[1]

    three_angles = [theta, 2 * np.pi / 3 + theta, 4 * np.pi / 3 + theta]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    vertices = []

    for i in range(3):
        x_local = ru[2 * i]
        y_local = ru[2 * i + 1]
        x_global = t_x + cos_theta * x_local - sin_theta * y_local
        y_global = t_y + sin_theta * x_local + cos_theta * y_local
        vertices.append((x_global, y_global))

    for i in range(3):
        theta_i = z[3 + i] + three_angles[i]
        cos_theta_i = np.cos(theta_i)
        sin_theta_i = np.sin(theta_i)
        x_g, y_g = vertices[i]
        a_i = ru[6 + i]

        x_center = x_g + (a_i + l_r / 2) * cos_theta_i
        y_center = y_g + (a_i + l_r / 2) * sin_theta_i

        local_corners = [(l_r/2, w_r/2), (-l_r/2, w_r/2), (-l_r/2, -w_r/2), (l_r/2, -w_r/2)]
        for x_local, y_local in local_corners:
            x_rotated = x_center + cos_theta_i * x_local - sin_theta_i * y_local
            y_rotated = y_center + sin_theta_i * x_local + cos_theta_i * y_local
            vertices.append((x_rotated, y_rotated))
    
    return vertices

def plot_formation(vertices_list, A_list, b_list, obstacles=None, vertices_optimized=None):
    fig, ax = plt.subplots()

    # Vẽ các vùng khả thi
    x = np.linspace(-1, 6, 400)
    y = np.linspace(-1, 6, 400)
    X, Y = np.meshgrid(x, y)

    for A, b in zip(A_list, b_list):
        Z = np.ones_like(X)
        for i in range(len(A)):
            Z *= (A[i, 0] * X + A[i, 1] * Y <= b[i])
        ax.contourf(X, Y, Z, levels=[0.5, 1], colors=['lightgray'], alpha=0.3)

    # Vẽ đội hình ban đầu (và goal)
    for vertices in vertices_list:
        triangle = Polygon(vertices[:3], closed=True, edgecolor='green', facecolor='green', zorder=3)
        ax.add_patch(triangle)
        for i in range(3):
            start = 3 + 4 * i
            robot_vertices = vertices[start:start + 4]
            rect = Polygon(robot_vertices, closed=True, edgecolor='green', facecolor='green', zorder=3)
            ax.add_patch(rect)

    # Vẽ đội hình tối ưu nếu có
    if vertices_optimized is not None:
        triangle = Polygon(vertices_optimized[:3], closed=True, edgecolor='green', facecolor='green', zorder=4)
        ax.add_patch(triangle)
        for i in range(3):
            start = 3 + 4 * i
            robot_vertices = vertices_optimized[start:start + 4]
            rect = Polygon(robot_vertices, closed=True, edgecolor='green', facecolor='green', zorder=4)
            ax.add_patch(rect)

    # Vẽ vật cản
    if obstacles:
        for obs in obstacles:
            obs_coords = list(zip(obs[0], obs[1]))
            polygon = Polygon(obs_coords, closed=True, edgecolor='black', facecolor='gray', zorder=2)
            ax.add_patch(polygon)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.set_title("Robot Formations and Feasible Regions")
    plt.grid(True)
    plt.show()

# Vật cản
obstacles = [
    np.array([[0.5, 0.9, 0.9, 0.5], [0.75, 0.75, 1.05, 1.05]]),
    np.array([[2.9, 3.3, 3.3, 2.9], [3.05, 3.05, 3.35, 3.35]]),
]


x_start = [4.0, 3.0, 0.0, 0.0, 0.0, 0.0]  # Trong P_g
x_goal = [3.0, 4.2, 0.0, 0.0, 0.0, 0.0]   # Mục tiêu (trong P_s)
x_opt = [3.60013403, 4.12713839, -0.12081759, 0.0, -0.78539816, 0.78539816] # Giải pháp hợp lệ trong P_s ∩ P_g


ru = [
    0.15 * np.cos(0.0), 0.15 * np.sin(0.0),
    0.15 * np.cos(2 * np.pi / 3), 0.15 * np.sin(2 * np.pi / 3),
    0.15 * np.cos(4 * np.pi / 3), 0.15 * np.sin(4 * np.pi / 3),
    0.2, 0.2, 0.2,
    0.3, 0.3
]
robot_dims = ru[9:11]
vertices_start = compute_formation_vertices(x_start, ru, robot_dims)
vertices_goal = compute_formation_vertices(x_goal, ru, robot_dims)
vertices_opt = compute_formation_vertices(x_opt, ru, robot_dims)

A_s = np.array([
    [-0.99271043,  0.12052388],
    [ 1.0,  0.0],
    [ 0.0,  1.0],
    [-1.0,  0.0],
    [ 0.0, -1.0]
])
b_s = np.array([
    -2.90834659, 5.0, 5.0, 0.0, 0.0
])



A_g = np.array([
    [ 0.05272655, -0.99860899],
    [ 1.0,  0.0],
    [ 0.0,  1.0],
    [-1.0,  0.0],
    [ 0.0, -1.0]
])
b_g = np.array([
    -3.19243311, 5.0, 5.0, 0.0, 0.0
])



A_inter = np.array([
    [-0.6276468, -0.77849823],
    [0.39009207, 0.92077586],
    [1.0, 0.0],
    [0.0, 1.0],
    [-1.0, 0.0],
    [0.0, -1.0],
    [0.05272655, -0.99860899],
    [1.0, 0.0],
    [0.0, 1.0],
    [-1.0, 0.0],
    [0.0, -1.0]
])
b_inter = np.array([
    -1.38230526, 3.93963336, 5.0, 5.0, 0.0, 0.0,
    -3.19243311, 5.0, 5.0, 0.0, 0.0
])


A_inter = np.array([
    [-0.99271043,  0.12052388],
    [ 1.0,  0.0],
    [ 0.0,  1.0],
    [-1.0,  0.0],
    [ 0.0, -1.0],
    [ 0.05272655, -0.99860899],
    [ 1.0,  0.0],
    [ 0.0,  1.0],
    [-1.0,  0.0],
    [ 0.0, -1.0]
])
b_inter = np.array([
    -2.90834659, 5.0, 5.0, 0.0, 0.0,
    -3.19243311, 5.0, 5.0, 0.0, 0.0
])


# Vẽ
plot_formation(
    [vertices_start, vertices_goal],
    [A_s, A_g, A_inter],
    [b_s, b_g, b_inter],
    obstacles=obstacles,
    vertices_optimized=vertices_opt
)
