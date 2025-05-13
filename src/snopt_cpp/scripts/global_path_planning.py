#!/home/dat/env/bin/python
import rospy
import irispy
import numpy as np
import cvxpy as cp
import json

def load_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def convert_obstacles_to_numpy(obstacles):
    """
    Chuyển đổi obstacles từ file config sang danh sách các mảng numpy shape (2, 4).
    """
    converted_obstacles = []
    for obs in obstacles:
        # Chuyển coordinates thành mảng numpy shape (2, 4)
        coords = np.array(obs['coordinates'])
        converted_obstacles.append(coords)
    return converted_obstacles

def compute_polytope(bounds, start):
    region, debug = irispy.inflate_region(obstacles, start, bounds=bounds, return_debug_data=True)

    A, b = None, None
    try:
        polyhedron = region.getPolyhedron()
        A = polyhedron.getA()
        b = polyhedron.getB()
        rospy.loginfo("Region inequalities: A=\n%s\nb=%s", A, b)
    except AttributeError:
        rospy.loginfo("Region: %s (could not access A and b)", region)

    return A, b

def formation(A, b, start, goal):
    pass

def intersect_polytopes(A_s, b_s, A_g, b_g):
    """
    Kiểm tra xem giao của hai đa diện P_s (A_s x <= b_s) và P_g (A_g x <= b_g) có rỗng hay không.
    
    Args:
        A_s (np.ndarray): Ma trận A của P_s (kích thước m_s x n)
        b_s (np.ndarray): Vector b của P_s (kích thước m_s)
        A_g (np.ndarray): Ma trận A của P_g (kích thước m_g x n)
        b_g (np.ndarray): Vector b của P_g (kích thước m_g)
    
    Returns:
        bool: True nếu giao không rỗng, False nếu rỗng
        A (np.ndarray): Ma trận A của đa diện giao (kích thước (m_s + m_g) x n)
        b (np.ndarray): Vector b của đa diện giao (kích thước m_s + m_g)
    """
    # Kết hợp ma trận A và vector b
    A = np.vstack((A_s, A_g))  # Nối A_s và A_g theo chiều dọc
    b = np.hstack((b_s, b_g))  # Nối b_s và b_g

    # Kiểm tra tính khả thi
    n = A_s.shape[1]  # Số chiều của x
    x = cp.Variable(n)
    constraints = [A @ x <= b]
    prob = cp.Problem(cp.Minimize(0), constraints)  # Bài toán giả để kiểm tra khả thi
    prob.solve(solver=cp.ECOS)  # Dùng solver ECOS cho nhanh

    if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
        return True, A, b
    else:
        return False, A, b

def shortest_path():
    pass

def global_path_planning():
    rospy.init_node('global_path_planning_node', anonymous=True)

    rospy.loginfo("Global path planning started")
    try:
        # Đọc file config
        config = load_config('../config/config.json')

        # Truy cập các thông tin
        obstacles = convert_obstacles_to_numpy(config['obstacles'])
        initial_config = config['initial_configuration']
        start_centroid = np.array(config['start_centroid'])
        goal_centroid = np.array(config['goal_centroid'])
        object_radius = config['object_radius']
        robot_shape = config['robot_shape']
        map_size = config['map']

        # Khởi tạo đồ thị rỗng G = {V, E}
        G = {'V': [], 'E': []}

        # Khởi tạo danh sách vùng lồi rỗng
        P = {'A': [], "b": []}

        # Thêm cấu hình ban đầu zs vào V
        zs = np.array(initial_config)
        G['V'].append(zs)

        # Tạo hai đa diện lồi từ s và g
        bounds = irispy.Polyhedron.from_bounds(map_size)

        A_s, b_s = compute_polytope(bounds, start_centroid)
        P['A'].append(A_s)
        P['b'].append(b_s)

        A_g, b_g = compute_polytope(bounds, goal_centroid)
        P['A'].append(A_g)
        P['b'].append(b_g)

        zg, status = formation(P['A'][1], P['b'], goal_centroid, goal_centroid)
        if status != 1:
            rospy.logerr("OptimaOptimization failed!")
            return None

        G['V'].append(zg)

        # Tạo danh sách các cấu hình phù hợp cho hai đa diện
        L_Ps = [zs]
        L_Pg = [zg]

        # Kiểm tra giao của P_s và P_g
        is_non_empty, A_inter, b_inter = intersect_polytopes(A_s, b_s, A_g, b_g)
        if is_non_empty:
            # Thử tính formation trong đa diện giao P_s ∩ P_g
            z = formation(A_inter, b_inter, start_centroid, goal_centroid)
            if z is not None:
                # Bước 12: Thêm z vào V
                G['V'].append(z)
                # Bước 13: Thêm cạnh (zs, z, Ps) vào E
                G['E'].append((zs, z, (A_s, b_s)))
                # Bước 14: Thêm cạnh (zg, z, Pg) vào E
                G['E'].append((zg, z, (A_g, b_g)))
                print(f"Found valid formation z in P_s ∩ P_g: {z}")
                print("Global path planning completed successfully.")
                return G, P, zs, zg
            else:
                print("Không tìm được formation hợp lệ trong P_s ∩ P_g.")

    except rospy.ROSInterruptException:
        rospy.logerr("Node interrupted")
    except Exception as e:
        rospy.logerr("Error: %s", str(e))
    
    rospy.spin()

if __name__ == '__main__':
    global_path_planning()