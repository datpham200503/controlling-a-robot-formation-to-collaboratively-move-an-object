import ctypes
import numpy as np
from ctypes import c_uint64, c_double, POINTER

# Load thư viện
lib = ctypes.cdll.LoadLibrary('/home/plg/catkin_ws/src/global_path_planning/lib/libbidijkstra.so')

# Định nghĩa kiểu trả về và đối số
lib.shortest_path.argtypes = [
    c_uint64,                      # n
    c_uint64,                      # m
    POINTER(c_uint64),             # edge_data
    POINTER(c_double),             # weights
    POINTER(c_uint64),             # polytope_ids
    c_uint64,                      # z_s
    c_uint64,                      # z_g
    POINTER(c_uint64),             # T_out
    POINTER(c_uint64),             # P_out
    POINTER(c_uint64),             # T_len
    POINTER(c_uint64)              # P_len
]

def shortest_path(n, edges, weights, polytope_ids, z_s, z_g):
    """
    Gọi hàm shortest_path từ C++ để tìm đường ngắn nhất.

    Args:
        n: Số đỉnh (uint64)
        edges: Mảng numpy [u1, v1, u2, v2, ...] (kích thước 2*m, dtype=np.uint64)
        weights: Mảng numpy trọng số (kích thước m, dtype=np.double)
        polytope_ids: Mảng numpy chỉ số đa diện (kích thước m, dtype=np.uint64)
        z_s: Đỉnh bắt đầu (uint64)
        z_g: Đỉnh kết thúc (uint64)

    Returns:
        tuple: (T, P)
            - T: Mảng numpy chứa đỉnh trên đường đi (dtype=np.uint64)
            - P: Mảng numpy chứa ID đa diện (dtype=np.uint64)
            - Nếu không có đường, trả về ([], [])
    """
    m = len(weights)
    if edges.size != 2 * m or polytope_ids.size != m:
        raise ValueError("Invalid input dimensions")

    # Chuẩn bị mảng đầu ra
    max_len = n + 1
    T_out = np.zeros(max_len, dtype=np.uint64)
    P_out = np.zeros(max_len, dtype=np.uint64)
    T_len = c_uint64(0)
    P_len = c_uint64(0)

    # Chuyển sang ctypes
    edge_data_c = (c_uint64 * (2 * m))(*edges)
    weights_c = (c_double * m)(*weights)
    polytope_ids_c = (c_uint64 * m)(*polytope_ids)
    T_out_c = (c_uint64 * max_len)(*T_out)
    P_out_c = (c_uint64 * max_len)(*P_out)

    # Gọi hàm C++
    lib.shortest_path(n, m, edge_data_c, weights_c, polytope_ids_c,
                      z_s, z_g, T_out_c, P_out_c,
                      ctypes.byref(T_len), ctypes.byref(P_len))

    # Trích xuất kết quả
    T = np.array([T_out_c[i] for i in range(T_len.value)], dtype=np.uint64)
    P = np.array([P_out_c[i] for i in range(P_len.value)], dtype=np.uint64)
    return T, P

def test_shortest_path():
    """Chạy các test cho hàm shortest_path."""
    print("Running shortest_path tests...")

    # Test 1: Đồ thị 4 đỉnh
    print("\nTest 1: 4 vertices, z_s=0, z_g=2")
    n = 4
    edges = np.array([0, 1, 3, 0, 1, 2, 0, 2], dtype=np.uint64)
    weights = np.array([1.0, 2.0, 2.0, 5.0], dtype=np.double)
    polytope_ids = np.array([0, 1, 2, 3], dtype=np.uint64)
    z_s, z_g = 0, 2
    T, P = shortest_path(n, edges, weights, polytope_ids, z_s, z_g)
    expected_T = np.array([0, 1, 2], dtype=np.uint64)
    expected_P = np.array([0, 2], dtype=np.uint64)
    print(f"T: {T}, P: {P}")
    print(f"Expected T: {expected_T}, P: {expected_P}")
    assert np.array_equal(T, expected_T) and np.array_equal(P, expected_P)
    print("Test 1 passed!")

    # Test 2: Đồ thị 5 đỉnh
    print("\nTest 2: 5 vertices, z_s=0, z_g=4")
    n = 5
    edges = np.array([0, 1, 0, 2, 1, 2, 2, 1, 1, 3, 2, 4, 4, 3, 1, 4, 2, 3], dtype=np.uint64)
    weights = np.array([4.0, 2.0, 2.0, 1.0, 2.0, 4.0, 1.0, 3.0, 4.0], dtype=np.double)
    polytope_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint64)
    z_s, z_g = 0, 4
    T, P = shortest_path(n, edges, weights, polytope_ids, z_s, z_g)
    expected_T = np.array([0, 2, 4], dtype=np.uint64)
    expected_P = np.array([1, 5], dtype=np.uint64)
    print(f"T: {T}, P: {P}")
    print(f"Expected T: {expected_T}, P: {expected_P}")
    assert np.array_equal(T, expected_T) and np.array_equal(P, expected_P)
    print("Test 2 passed!")

    # Test 3: Không có đường
    print("\nTest 3: No path, z_s=0, z_g=1")
    n = 2
    edges = np.array([], dtype=np.uint64)
    weights = np.array([], dtype=np.double)
    polytope_ids = np.array([], dtype=np.uint64)
    z_s, z_g = 0, 1
    T, P = shortest_path(n, edges, weights, polytope_ids, z_s, z_g)
    expected_T = np.array([], dtype=np.uint64)
    expected_P = np.array([], dtype=np.uint64)
    print(f"T: {T}, P: {P}")
    print(f"Expected T: {expected_T}, P: {expected_P}")
    assert np.array_equal(T, expected_T) and np.array_equal(P, expected_P)
    print("Test 3 passed!")

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_shortest_path()