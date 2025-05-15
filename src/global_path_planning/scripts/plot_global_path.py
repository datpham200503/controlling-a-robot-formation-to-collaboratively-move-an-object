import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import json

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

def plot_path_planning(map_size, z_init, zg, P, G, obstacles , T):
    ru = [
        object_radius * np.cos(0.0), object_radius * np.sin(0.0),
        object_radius * np.cos(2 * np.pi / 3), object_radius * np.sin(2 * np.pi / 3),
        object_radius * np.cos(4 * np.pi / 3), object_radius * np.sin(4 * np.pi / 3),
        a_i, a_i, a_i,
        l_r, w_r
    ]
    robot_dims = [l_r, w_r]

    fig, ax = plt.subplots()
    x_min, y_min = map_size[0]
    x_max, y_max = map_size[1]

    x = np.linspace(x_min - 1, x_max + 1, 400)
    y = np.linspace(y_min - 1, y_max + 1, 400)
    X, Y = np.meshgrid(x, y)

    for polytope in P['A']:
        A = np.array(polytope)
        b = P['b'][P['A'].index(polytope)]
        Z = np.ones_like(X)
        for i in range(len(A)):
            Z *= (A[i, 0] * X + A[i, 1] * Y <= b[i])
        ax.contourf(X, Y, Z, levels=[0.5, 1], colors=['lightgray'], alpha=0.3)

    # Vẽ đội hình robot với màu sắc khác nhau
    for idx, z in enumerate(G['V']):
        # Xác định màu dựa trên vị trí
        if idx == 0 or idx == len(G['V']) - 1:  # Đầu hoặc cuối
            color = 'orange'
        else:  # Ở giữa
            color = 'green'

        vertices = compute_formation_vertices(z, ru, robot_dims)
        triangle = Polygon(vertices[:3], closed=True, edgecolor=color, facecolor=color, alpha=0.5, zorder=3)
        ax.add_patch(triangle)
        for i in range(3):
            start = 3 + 4 * i
            robot_vertices = vertices[start:start + 4]
            rect = Polygon(robot_vertices, closed=True, edgecolor=color, facecolor=color, alpha=0.5, zorder=3)
            ax.add_patch(rect)
        ax.plot(z[0], z[1], 'ro', markersize=5, zorder=4)
        ax.text(z[0], z[1], str(T[idx]), fontsize=20, color='black', ha='right', va='center', zorder=5)

    path_x = [z[0] for z in G['V']]
    path_y = [z[1] for z in G['V']]
    ax.plot(path_x, path_y, 'b-', linewidth=2, zorder=2, label='Path')

    if obstacles:
        for obs in obstacles:
            obs_coords = list(zip(obs['coordinates'][0], obs['coordinates'][1]))
            polygon = Polygon(obs_coords, closed=True, edgecolor='black', facecolor='gray', zorder=2)
            ax.add_patch(polygon)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_title("Robot Formations and Feasible Regions with Path")
    plt.grid(True)
    plt.legend()
    plt.show()

# Đọc dữ liệu từ file global.json
global_json_path = '/home/plg/catkin_ws/src/global_path_planning/config/global.json'
try:
    with open(global_json_path, 'r') as file:
        global_data = json.load(file)
    map_size = global_data['map']
    z_init = global_data['initial_configuration']
    obstacles = global_data['obstacles']
    object_radius = global_data['object_radius']
    a_i = global_data['robot_shape']['a_i']
    l_r = global_data['robot_shape']['l_r']
    w_r = global_data['robot_shape']['w_r']
except FileNotFoundError:
    print(f"Error: File {global_json_path} not found. Please check the path.")
except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in {global_json_path}. Please check the file content.")
except KeyError as e:
    print(f"Error: Missing key {e} in {global_json_path}. Please ensure the file has 'map', 'initial_configuration', 'obstacles', 'object_radius', and 'robot_shape'.")
    exit(1)

# Đọc dữ liệu từ file global_path.json
global_path_json_path = '/home/plg/catkin_ws/src/global_path_planning/config/global_path.json'
try:
    with open(global_path_json_path, 'r') as file:
        path_data = json.load(file)
    P = {
        'A': [polytope['A'] for polytope in path_data['polytopes']],
        'b': [polytope['b'] for polytope in path_data['polytopes']]
    }
    G = {'V': path_data['z_values'], 'E': []}
    zg = path_data['z_values'][-1]
    T = path_data['T']
except FileNotFoundError:
    print(f"Error: File {global_path_json_path} not found. Please check the path.")
except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in {global_path_json_path}. Please check the file content.")
except KeyError as e:
    print(f"Error: Missing key {e} in {global_path_json_path}. Please ensure the file has 'polytopes' and 'z_values'.")
    exit(1)

# Gọi hàm để vẽ
plot_path_planning(map_size, z_init, zg, P, G, obstacles, T)