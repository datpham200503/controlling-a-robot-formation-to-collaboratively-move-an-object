#!/home/dat/env/bin/python
import rospy
import irispy
import numpy as np
import cvxpy as cp
import time
from snopt import formation
from shortest_path import shortest_path

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
    A = np.vstack((A_s, A_g))
    b = np.hstack((b_s, b_g))
    n = A_s.shape[1]
    x = cp.Variable(n)
    constraints = [A @ x <= b]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.CLARABEL)
    if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
        return True, A, b
    return False, A, b

def sample_random_point(map_size, obstacles, P, max_attempts=1000):
    np.random.seed(int(time.time() * 1000) % (2**32))
    map_size = np.array(map_size)
    if map_size.shape == (2, 2):
        x_min, y_min = map_size[0]
        x_max, y_max = map_size[1]
    elif map_size.shape == (4,):
        x_min, y_min, x_max, y_max = map_size
    else:
        rospy.logerr("Invalid map_size format: %s", map_size)
        return None
    if x_max <= x_min or y_max <= y_min:
        rospy.logerr("Invalid map_size: x_max=%f, x_min=%f, y_max=%f, y_min=%f",
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
                    break
            except Exception as e:
                rospy.logerr("Error checking polytope: %s", str(e))
                is_valid = False
                break
        if is_valid:
            return p
    rospy.logerr("Could not find valid random point after %d attempts.", max_attempts)
    return None

def compute_centroid(z):
    return z[:2]

def euclidean_distance(z1, z2):
    c1 = compute_centroid(z1)
    c2 = compute_centroid(z2)
    return np.linalg.norm(c1 - c2)

def process_new_polytope(A_p, b_p, p, G, P, L_P, zinit, iteration):
    # rospy.loginfo("Bắt đầu process_new_polytope, vòng lặp %d", iteration)
    status_p, z = formation(zinit, p, A_p, b_p)
    # rospy.loginfo("Kết quả formation: status_p=%d, z=%s", status_p, z)
    if status_p != 1 or z is None:
        rospy.loginfo("Không tìm thấy formation hợp lệ cho P_p trong vòng lặp %d", iteration)
        return False

    L_Pp = []
    vertex_idx = len(G['V'])
    G['V'].append(z)

    for i, (A_i, b_i) in enumerate(zip(P['A'], P['b'])):
        is_non_empty, A_inter, b_inter = intersect_polytopes(A_i, b_i, A_p, b_p)
        # rospy.loginfo("Giao P_%d ∩ P_p: is_non_empty=%s", i, is_non_empty)
        if is_non_empty:
            status_inter, z_1 = formation(zinit, p, A_inter, b_inter)
            # rospy.loginfo("Formation giao: status_inter=%d, z_1=%s", status_inter, z_1)
            if status_inter == 1 and z_1 is not None:
                z1_idx = len(G['V'])
                G['V'].append(z_1)
                for z_i_idx in L_P.get(i, []):
                    weight = euclidean_distance(G['V'][z1_idx], G['V'][z_i_idx])
                    G['E'].append({
                        'z1': z1_idx,
                        'z2': z_i_idx,
                        'weight': weight,
                        'polytope': (A_i, b_i)
                    })
                    G['E'].append({
                        'z1': z_i_idx,
                        'z2': z1_idx,
                        'weight': weight,
                        'polytope': (A_i, b_i)
                    })
                    # rospy.loginfo("Thêm cạnh: (%d, %d) và (%d, %d), polytope=A_%d", z1_idx, z_i_idx, z_i_idx, z1_idx, i)
                for z_i_idx in L_Pp:
                    weight = euclidean_distance(G['V'][z1_idx], G['V'][z_i_idx])
                    G['E'].append({
                        'z1': z1_idx,
                        'z2': z_i_idx,
                        'weight': weight,
                        'polytope': (A_p, b_p)
                    })
                    G['E'].append({
                        'z1': z_i_idx,
                        'z2': z1_idx,
                        'weight': weight,
                        'polytope': (A_p, b_p)
                    })
                    # rospy.loginfo("Thêm cạnh: (%d, %d) và (%d, %d), polytope=A_p", z1_idx, z_i_idx, z_i_idx, z1_idx)
                L_Pp.append(z1_idx)
                if i not in L_P:
                    L_P[i] = []
                L_P[i].append(z1_idx)
        if i in L_P:
            for idx1 in L_P[i]:
                for idx2 in L_P[i]:
                    if idx1 != idx2:
                        weight = euclidean_distance(G['V'][idx1], G['V'][idx2])
                        G['E'].append({
                            'z1': idx1,
                            'z2': idx2,
                            'weight': weight,
                            'polytope': (A_i, b_i)
                        })
                        # rospy.loginfo("Thêm cạnh trong L_P[%d]: (%d, %d)", i, idx1, idx2)

    P['A'].append(A_p)
    P['b'].append(b_p)
    L_P[len(P['A']) - 1] = L_Pp
    # rospy.loginfo("Kết thúc process_new_polytope, G['V']=%d, G['E']=%d", len(G['V']), len(G['E']))
    # rospy.loginfo("L_P: %s", L_P)
    return True

def shortest_path_wrapper(G):
    n = len(G['V'])
    z_s = G['zs']
    z_g = G['zg']
    m = len(G['E'])
    rospy.loginfo("Số lượng đỉnh: n=%d, Số lượng cạnh: m=%d", n, m)
    # rospy.loginfo("z_s=%d, z_g=%d", z_s, z_g)
    # rospy.loginfo("G['E']: %s", G['E'])
    edges = np.zeros(4 * m, dtype=np.uint64)
    weights = np.zeros(2 * m, dtype=np.double)
    polytope_ids = np.zeros(2 * m, dtype=np.uint64)
    for i, edge in enumerate(G['E']):
        edges[4*i] = edge['z1']
        edges[4*i+1] = edge['z2']
        edges[4*i+2] = edge['z2']
        edges[4*i+3] = edge['z1']
        weights[2*i] = edge['weight']
        weights[2*i+1] = edge['weight']
        polytope_ids[2*i] = i
        polytope_ids[2*i+1] = i
    rospy.loginfo("n = %d, edges = %s", n, edges)
    T, P_ids = shortest_path(n, edges, weights, polytope_ids, z_s, z_g)
    rospy.loginfo("T=%s, P_ids=%s", T, P_ids)
    polytopes = []
    for pid in P_ids:
        if pid < len(G['E']):
            polytopes.append(G['E'][pid]['polytope'])
        else:
            polytopes.append((np.array([]), np.array([])))
    return T, polytopes