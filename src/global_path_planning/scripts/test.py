import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch

def plot_polytope_region(ax, A, b, color, name):
    x = np.linspace(-1, 6, 600)
    y = np.linspace(-1, 6, 600)
    X, Y = np.meshgrid(x, y)
    Z = np.ones_like(X, dtype=bool)

    for i in range(A.shape[0]):
        Z &= (A[i, 0] * X + A[i, 1] * Y <= b[i])

    contour = ax.contourf(X, Y, Z, levels=[0.5, 1], colors=[color], alpha=0.5)
    return Patch(facecolor=color, edgecolor='black', label=name, alpha=0.5)

def plot_obstacles(ax, obstacles):
    for obs in obstacles:
        x_coords, y_coords = obs["coordinates"]
        polygon = Polygon(list(zip(x_coords, y_coords)), closed=True,
                          facecolor='gray', edgecolor='black', alpha=0.8)
        ax.add_patch(polygon)

# Dữ liệu từ ROS log
A_s = np.array([
    [-0.6276468,  -0.77849823],
    [ 0.39009207,  0.92077586],
    [ 1.0, 0.0],
    [ 0.0, 1.0],
    [-1.0, -0.0],
    [-0.0, -1.0]
])
b_s = np.array([
    -1.38230526,
     3.93963336,
     5.0,
     5.0,
     0.0,
     0.0
])


A_g = np.array([
    [ 0.05272655, -0.99860899],
    [ 1.0, 0.0],
    [ 0.0, 1.0],
    [-1.0, -0.0],
    [-0.0, -1.0]
])
b_g = np.array([
    -3.19243311,
     5.0,
     5.0,
     0.0,
     0.0
])



# Danh sách vật cản
obstacles = [
    {
        "coordinates": [
            [0.5, 0.9, 0.9, 0.5],
            [0.75, 0.75, 1.05, 1.05]
        ]
    },
    {
        "coordinates": [
            [2.9, 3.3, 3.3, 2.9],
            [3.05, 3.05, 3.35, 3.35]
        ]
    }
]

# Vẽ
fig, ax = plt.subplots(figsize=(8, 8))
legend_elements = []
legend_elements.append(plot_polytope_region(ax, A_s, b_s, color='lightblue', name='P_s'))
legend_elements.append(plot_polytope_region(ax, A_g, b_g, color='salmon', name='P_g'))

plot_obstacles(ax, obstacles)
legend_elements.append(Patch(facecolor='gray', edgecolor='black', label='Obstacle'))

ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Hai vùng khả thi và vật cản")
ax.legend(handles=legend_elements)
plt.show()
