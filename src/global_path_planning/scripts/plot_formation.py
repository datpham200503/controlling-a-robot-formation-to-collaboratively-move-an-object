import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def compute_formation_vertices(z, ru, robot_dims):
    """
    Tính toán các đỉnh của đội hình robot từ cấu hình z.
    """
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

def plot_path_planning(map_size, z_init, zg, P, G, obstacles):
    """
    Vẽ các vùng lồi khả thi, đội hình robot tại các cấu hình trong G['V'], 
    điểm trung tâm đội hình, và vật cản.
    
    Args:
        map_size: [[x_min, x_max], [y_min, y_max]]
        z_init: Cấu hình khởi tạo cho formation
        zg: Cấu hình đích
        P: {'A': [A_1, A_2, ...], 'b': [b_1, b_2, ...]}
        G: {'V': [z_1, z_2, ...], 'E': [(z_i, z_j, (A, b)), ...]}
        obstacles: Danh sách vật cản [[x_coords], [y_coords]]
    """
    # Khởi tạo ru và robot_dims
    ru = [
        0.15 * np.cos(0.0), 0.15 * np.sin(0.0),
        0.15 * np.cos(2 * np.pi / 3), 0.15 * np.sin(2 * np.pi / 3),
        0.15 * np.cos(4 * np.pi / 3), 0.15 * np.sin(4 * np.pi / 3),
        0.2, 0.2, 0.2,
        0.3, 0.3
    ]
    robot_dims = ru[9:11]  # [0.3, 0.3]

    # Tạo figure
    fig, ax = plt.subplots()

    # Lấy giới hạn từ map_size
    x_min, y_min = map_size[0]
    x_max, y_max = map_size[1]

    # Vẽ các vùng lồi khả thi
    x = np.linspace(x_min - 1, x_max + 1, 400)
    y = np.linspace(y_min - 1, y_max + 1, 400)
    X, Y = np.meshgrid(x, y)

    for A, b in zip(P['A'], P['b']):
        if A is None or b is None:
            continue
        Z = np.ones_like(X)
        for i in range(len(A)):
            Z *= (A[i, 0] * X + A[i, 1] * Y <= b[i])
        ax.contourf(X, Y, Z, levels=[0.5, 1], colors=['lightgray'], alpha=0.3)

    # Vẽ đội hình robot và điểm trung tâm
    for z in G['V']:
        vertices = compute_formation_vertices(z, ru, robot_dims)
        # Vẽ tam giác (3 robot)
        triangle = Polygon(vertices[:3], closed=True, edgecolor='green', facecolor='green', alpha=0.5, zorder=3)
        ax.add_patch(triangle)
        # Vẽ hình chữ nhật cho mỗi robot
        for i in range(3):
            start = 3 + 4 * i
            robot_vertices = vertices[start:start + 4]
            rect = Polygon(robot_vertices, closed=True, edgecolor='green', facecolor='green', alpha=0.5, zorder=3)
            ax.add_patch(rect)
        # Vẽ điểm trung tâm đội hình
        ax.plot(z[0], z[1], 'ro', markersize=5, zorder=4)

    # Vẽ vật cản
    if obstacles:
        for obs in obstacles:
            obs_coords = list(zip(obs[0], obs[1]))
            polygon = Polygon(obs_coords, closed=True, edgecolor='black', facecolor='gray', zorder=2)
            ax.add_patch(polygon)

    # Cài đặt giới hạn và giao diện
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_title("Robot Formations and Feasible Regions")
    plt.grid(True)
    plt.show()