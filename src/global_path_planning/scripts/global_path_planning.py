#!/home/plg/env/bin/python
import rospy
import numpy as np
import irispy
import json
from snopt import formation
from plot_formation import plot_path_planning
from planning_functions import (compute_polytope, intersect_polytopes, sample_random_point,
                              compute_centroid, euclidean_distance, process_new_polytope,
                              shortest_path_wrapper)

def load_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def convert_obstacles_to_numpy(obstacles):
    converted_obstacles = []
    for obs in obstacles:
        coords = np.array(obs['coordinates'])
        converted_obstacles.append(coords)
    return converted_obstacles

def save_path_to_json(T, polytopes, G, file_path):
    """
    Save T, polytopes, and corresponding z values to a JSON file.
    """
    try:
        # Ensure T is a list of Python integers
        if not isinstance(T, list):
            T = list(T)
        T = [int(x) for x in T]  # Convert uint64 or other types to Python int
        # Convert polytopes to JSON-serializable format
        polytopes_serializable = []
        for i, (A, b) in enumerate(polytopes):
            # Handle A
            if A is None:
                A_list = []
            elif isinstance(A, np.ndarray) and A.size > 0:
                A_list = A.tolist()
            else:
                A_list = []
            # Handle b
            if b is None:
                b_list = []
            elif isinstance(b, np.ndarray) and b.size > 0:
                b_list = b.tolist()
            else:
                b_list = []
            polytopes_serializable.append({
                "A": A_list,
                "b": b_list
            })
        # Convert z values corresponding to T
        z_values = []
        for idx in T:
            z = G['V'][idx]
            if isinstance(z, np.ndarray) and z.size > 0:
                z_list = z.tolist()
            else:
                z_list = []
            z_values.append(z_list)
        # Create data dictionary
        data = {
            "T": T,
            "polytopes": polytopes_serializable,
            "z_values": z_values
        }
        # Write to JSON file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        rospy.loginfo("Saved path to %s", file_path)
    except Exception as e:
        rospy.logerr("Failed to save path to %s: %s", file_path, str(e))
        raise  # Re-raise for debugging

def global_path_planning():
    rospy.init_node('global_path_planning_node', anonymous=True)
    rospy.loginfo("Global path planning started")
    try:
        config = load_config('/home/plg/catkin_ws/src/global_path_planning/config/global.json')
        obstacles = convert_obstacles_to_numpy(config['obstacles'])
        initial_config = np.array(config['initial_configuration'])
        start_centroid = np.array(config['start_centroid'])
        goal_centroid = np.array(config['goal_centroid'])
        map_size = config['map']
        max_iterations = config.get('max_iterations', 10)

        G = {'V': [], 'E': [], 'zs': None, 'zg': None}
        P = {'A': [], 'b': []}
        L_P = {}

        zs = initial_config
        G['V'].append(zs)
        G['zs'] = 0

        bounds = irispy.Polyhedron.from_bounds(map_size[0], map_size[1])
        A_s, b_s = compute_polytope(obstacles, start_centroid, bounds)
        P['A'].append(A_s)
        P['b'].append(b_s)

        zinit = np.array([goal_centroid[0], goal_centroid[1], 0.0, 0.0, 0.0, 0.0])
        A_g, b_g = compute_polytope(obstacles, goal_centroid, bounds)
        P['A'].append(A_g)
        P['b'].append(b_g)

        status_g, zg = formation(zinit, goal_centroid, A_g, b_g)
        if status_g != 1 or zg is None:
            rospy.logerr("Failed to compute formation for zg")
            return None

        G['V'].append(zg)
        G['zg'] = len(G['V']) - 1

        L_Ps = [G['zs']]
        L_Pg = [G['zg']]
        L_P = {0: L_Ps, 1: L_Pg}

        is_non_empty, A_inter, b_inter = intersect_polytopes(A_s, b_s, A_g, b_g)
        rospy.loginfo("Giao P_s ∩ P_g: is_non_empty=%s, A_inter shape=%s", 
                      is_non_empty, A_inter.shape if A_inter is not None else None)
        # if is_non_empty:
        #     status_sg, z_sg = formation(zinit, start_centroid, A_inter, b_inter)
        #     if status_sg == 1 and z_sg is not None:
        #         z_sg_idx = len(G['V'])
        #         G['V'].append(z_sg)
        #         weight_s = euclidean_distance(zs, z_sg)
        #         weight_g = euclidean_distance(zg, z_sg)
        #         G['E'].append({
        #             'z1': G['zs'],
        #             'z2': z_sg_idx,
        #             'weight': weight_s,
        #             'polytope': (A_s, b_s)
        #         })
        #         G['E'].append({
        #             'z1': z_sg_idx,
        #             'z2': G['zs'],
        #             'weight': weight_s,
        #             'polytope': (A_s, b_s)
        #         })
        #         G['E'].append({
        #             'z1': z_sg_idx,
        #             'z2': G['zg'],
        #             'weight': weight_g,
        #             'polytope': (A_g, b_g)
        #         })
        #         G['E'].append({
        #             'z1': G['zg'],
        #             'z2': z_sg_idx,
        #             'weight': weight_g,
        #             'polytope': (A_g, b_g)
        #         })
        #         L_P[0].append(z_sg_idx)
        #         L_P[1].append(z_sg_idx)
        #         rospy.loginfo("Thêm cạnh từ P_s ∩ P_g: (%d, %d), (%d, %d)", G['zs'], z_sg_idx, z_sg_idx, G['zg'])
        #         T, polytopes = shortest_path_wrapper(G)
        #         if len(T) > 0:
        #             rospy.loginfo("Tìm thấy đường khả thi qua P_s ∩ P_g: T=%s", T)
        #             plot_path_planning(map_size, initial_config, zg, P, G, obstacles)
        #             save_path_to_json(T, polytopes, '/home/plg/catkin_ws/src/global_path_planning/config/global_path.json')
        #             return T, polytopes

        for iteration in range(max_iterations):
            rospy.loginfo("Vòng lặp %d/%d", iteration + 1, max_iterations)
            p = sample_random_point(map_size, obstacles, P, max_attempts=1000)
            rospy.loginfo("Điểm ngẫu nhiên được chọn: p=%s", p)
            if p is None:
                rospy.loginfo("Không tìm thấy điểm ngẫu nhiên hợp lệ")
                continue

            A_p, b_p = compute_polytope(obstacles, p, bounds)
            success = process_new_polytope(A_p, b_p, p, G, P, L_P, zinit, iteration + 1)
            if not success:
                continue

            # rospy.loginfo("Trước shortest_path_wrapper: G['E']=%s", G['E'])
            T, polytopes = shortest_path_wrapper(G)
            if len(T) > 0:
                rospy.loginfo("Tìm thấy đường khả thi: T=%s, polytopes=%d", T, len(polytopes))
                plot_path_planning(map_size, initial_config, zg, P, G, obstacles)
                save_path_to_json(T, polytopes, G, '/home/plg/catkin_ws/src/global_path_planning/config/global_path.json')
                return T, polytopes

        rospy.loginfo("No feasible path found after %d iterations", max_iterations)
        return None

    except rospy.ROSInterruptException:
        rospy.logerr("Node interrupted")
    except Exception as e:
        rospy.logerr("Error: %s", str(e))
        return None

if __name__ == '__main__':
    global_path_planning()