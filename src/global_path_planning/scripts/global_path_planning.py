#!/home/dat/env/bin/python
import rospy
import irispy
import numpy as np
import cvxpy as cp
import json
import time
from snopt import formation
from plot_formation import plot_path_planning

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

def compute_polytope(obstacles, start, bounds):
    region, debug = irispy.inflate_region(obstacles, start, bounds=bounds, return_debug_data=True)

    A, b = None, None
    try:
        polyhedron = region.getPolyhedron()
        A = polyhedron.getA()
        b = polyhedron.getB()
    except AttributeError:
        rospy.loginfo("Region: %s (could not access A and b)", region)

    return A, b

def intersect_polytopes(A_s, b_s, A_g, b_g):
    # Kết hợp ma trận A và vector b
    A = np.vstack((A_s, A_g))  # Nối A_s và A_g theo chiều dọc
    b = np.hstack((b_s, b_g))  # Nối b_s và b_g

    # Kiểm tra tính khả thi
    n = A_s.shape[1]  # Số chiều của x
    x = cp.Variable(n)
    constraints = [A @ x <= b]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.CLARABEL)
    if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
        return True, A, b
    else:
        return False, A, b

def sample_random_point(map_size, obstacles, P, max_attempts=1000):
    """
    Lấy mẫu ngẫu nhiên điểm p trong không gian tự do F¯, không thuộc các vật cản hoặc đa diện trong P.
    """
    np.random.seed(int(time.time() * 1000) % (2**32))
    map_size = np.array(map_size)
    
    # Chuẩn hóa map_size
    if map_size.shape == (2, 2):
        x_min, y_min = map_size[0]
        x_max, y_max = map_size[1]
        map_size = [x_min, y_min, x_max, y_max]
    elif map_size.shape == (4,):
        x_min, y_min, x_max, y_max = map_size
    else:
        rospy.logerr("Invalid map_size format: expected [[x_min, x_max], [y_min, y_max]] or [x_min, y_min, x_max, y_max], got %s", map_size)
        return None
    
    if x_max <= x_min or y_max <= y_min:
        rospy.logerr("Invalid map_size: x_max (%f) <= x_min (%f) or y_max (%f) <= y_min (%f)",
                     x_max, x_min, y_max, y_min)
        return None

    for attempt in range(max_attempts):
        p = np.random.uniform([x_min, y_min], [x_max, y_max])
        is_valid = True
        for obs in obstacles:
            x_coords, y_coords = obs
            vertices = np.array(list(zip(x_coords, y_coords)))
            try:
                A = np.array([np.cross(vertices[(i + 1) % len(vertices)] - vertices[i],
                                      vertices[i] - p) <= 0 for i in range(len(vertices))])
                if np.all(A):
                    is_valid = False
                    # rospy.loginfo("Point %s lies in obstacle", p)
                    break
            except Exception as e:
                rospy.logerr("Error checking obstacle: %s", str(e))
                is_valid = False
                break
        if not is_valid:
            continue
        for A_i, b_i in zip(P['A'], P['b']):
            if A_i is None or b_i is None:
                continue
            try:
                if np.all(A_i @ p <= b_i):
                    is_valid = False
                    # rospy.loginfo("Point %s lies in polytope", p)
                    break
            except Exception as e:
                rospy.logerr("Error checking polytope: %s", str(e))
                is_valid = False
                break
        if is_valid:
            return p

    rospy.logerr("Could not find valid random point after %d attempts.", max_attempts)
    return None

def process_new_polytope(A_p, b_p, p, G, P, L_P, zinit, iteration):
    """
    Xử lý đa diện mới P_p theo bước 19-34 của Thuật toán 2.
    """

    rospy.loginfo("Bắt đầu process_new_polytope, vòng lặp %d", iteration)
    rospy.loginfo("Đầu vào: p=%s, A_p=\n%s\nb_p=%s", p, A_p, b_p)
    rospy.loginfo("Trạng thái ban đầu: G['V']=%s, G['E']=%s, L_P=%s", len(G['V']), len(G['E']), L_P)

    # Bước 19: Kiểm tra formation cho P_p
    status_p, z = formation(zinit, p, A_p, b_p)
    rospy.loginfo("Kết quả formation: status_p=%d, z=%s", status_p, z)
    if status_p != 1 or z is None:
        rospy.loginfo("Không tìm thấy formation hợp lệ cho P_p trong vòng lặp %d", iteration)
        return False

    # Bước 20: Khởi tạo L_Pp
    L_Pp = []
    
    # Bước 21: Duyệt qua các đa diện trong P
    for i, (A_i, b_i) in enumerate(zip(P['A'], P['b'])):
        rospy.loginfo("Đang xử lý đa diện P[%d]: A_i=\n%s\nb_i=%s", i, A_i, b_i)
        # Bước 22: Kiểm tra giao P_i ∩ P_p
        is_non_empty, A_inter, b_inter = intersect_polytopes(A_i, b_i, A_p, b_p)
        rospy.loginfo("Kết quả giao: is_non_empty=%s, A_inter=\n%s\nb_inter=%s", is_non_empty, A_inter, b_inter)
        if is_non_empty:
            status_inter, z_1 = formation(zinit, p, A_inter, b_inter)
            rospy.loginfo("Kết quả formation giao: status_inter=%d, z_1=%s", status_inter, z_1)
            if status_inter == 1 and z_1 is not None:
                # Bước 23-25: Thêm cạnh với các cấu hình trong L_P[i]
                for z_i in L_P.get(i, []):
                    G['E'].append((z_1, z_i, (A_i, b_i)))
                
                # Bước 26-28: Thêm cạnh với các cấu hình trong L_Pp
                for z_i in L_Pp:
                    G['E'].append((z_1, z_i, (A_p, b_p)))
                
                # Bước 29: Thêm z_1 vào V
                G['V'].append(z_1)
                
                # Bước 30: Cập nhật L_Pp và L_P[i]
                L_Pp.append(z_1)
                if i not in L_P:
                    L_P[i] = []
                L_P[i].append(z_1)
                rospy.loginfo("Cập nhật L_Pp=%s, L_P[%d]=%s", L_Pp, i, L_P[i])
    
    # Bước 33: Thêm P_p vào P
    P['A'].append(A_p)
    P['b'].append(b_p)
    L_P[len(P['A']) - 1] = L_Pp  # Thêm L_Pp cho đa diện mới
    
    return True

def shortest_path():
    pass

def global_path_planning():
    rospy.init_node('global_path_planning_node', anonymous=True)

    rospy.loginfo("Global path planning started")
    try:
        # Đọc file config
        config = load_config('/home/dat/catkin_ws/src/global_path_planning/config/global.json')

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
        bounds = irispy.Polyhedron.from_bounds(map_size[0], map_size[1])

        A_s, b_s = compute_polytope(obstacles, start_centroid, bounds)
        P['A'].append(A_s)
        P['b'].append(b_s)

        A_g, b_g = compute_polytope(obstacles, goal_centroid, bounds)
        P['A'].append(A_g)
        P['b'].append(b_g)

        # rospy.loginfo("Đã tính P_s: A_s=\n%s\nb_s=%s", A_s, b_s)
        # rospy.loginfo("Đã tính P_g: A_g=\n%s\nb_g=%s", A_g, b_g)

        zinit = np.array([goal_centroid[0], goal_centroid[1], 0.0, 0.0, 0.0, 0.0])  # t_x, t_y, theta, theta_1, theta_2, theta_3
        status_g, zg = formation(zinit, goal_centroid, P['A'][1], P['b'][1])
        # rospy.loginfo("Cấu hình zg: %s", zg)

        if status_g != 1:
            rospy.logerr("OptimaOptimization failed!")
            return None


        G['V'].append(zg)

        # Tạo danh sách các cấu hình phù hợp cho hai đa diện
        L_Ps = [zs]
        L_Pg = [zg]
        L_P = {0: L_Ps, 1: L_Pg}

        # Kiểm tra giao của P_s và P_g
        # is_non_empty, A_inter, b_inter = intersect_polytopes(A_s, b_s, A_g, b_g)
        # rospy.loginfo("Đã tính giao giữa hai đa diện Ps và Pg: A_inter=\n%s\nb_inter=%s", A_inter, b_inter)
        
        # if is_non_empty:
        #     # Thử tính formation trong đa diện giao P_s ∩ P_g
        #     status_sg, z_sg = formation(zinit, start_centroid, A_inter, b_inter)
        #     if z_sg is not None and status_sg == 1:
        #         G['V'].append(z_sg)
        #         G['E'].append((zs, z_sg, (A_s, b_s)))
        #         G['E'].append((zg, z_sg, (A_g, b_g)))
        #         # rospy.loginfo("Found valid formation z in P_s ∩ P_g: %s", z_sg)
        #         rospy.loginfo("Global path planning completed successfully.")
        #         return G, P, zs, zg
        #     else:
        #         rospy.loginfo("Không tìm được formation hợp lệ trong P_s ∩ P_g.")

        max_iterations = 1
        for iteration in range(max_iterations):
            p = sample_random_point(map_size, obstacles, P, max_attempts=1000)
            # rospy.loginfo("Random point sampled successfully: %s", p)
            if p is None:
                rospy.loginfo("Không tìm thấy điểm ngẫu nhiên hợp lệ trong không gian tự do.")
                break

            A_p, b_p = compute_polytope(obstacles, p, bounds)

            success = process_new_polytope(A_p, b_p, p, G, P, L_P, zinit, iteration + 1)
            if not success:
                continue

        plot_path_planning(map_size, initial_config, zg, P, G, obstacles)

        rospy.loginfo("Global path planning completed successfully.")
        return G, P, zs, zg

    except rospy.ROSInterruptException:
        rospy.logerr("Node interrupted")
    except Exception as e:
        rospy.logerr("Error: %s", str(e))
    
    rospy.spin()

if __name__ == '__main__':
    global_path_planning()