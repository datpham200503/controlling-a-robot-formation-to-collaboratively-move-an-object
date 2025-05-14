import ctypes
import numpy as np

# Load thư viện gián tiếp
try:
    lib = ctypes.CDLL('C:/controlling-a-robot-formation-to-collaboratively-move-an-object/src/global_path_planning/scripts/libbidijkstra.so')
except OSError as e:
    raise OSError("Failed to load libbidijkstra.so. Ensure it is in LD_LIBRARY_PATH or a system library path (e.g., /usr/lib). Error: " + str(e))

# Định nghĩa kiểu trả về và đối số của hàm bidirectional_dijkstra
lib.bidirectional_dijkstra.restype = ctypes.c_double
lib.bidirectional_dijkstra.argtypes = [
    ctypes.c_uint64,                    # n
    ctypes.c_uint64,                    # m
    ctypes.POINTER(ctypes.c_uint64),    # edges
    ctypes.POINTER(ctypes.c_double),    # weights
    ctypes.c_uint64,                    # s
    ctypes.c_uint64,                    # t
    ctypes.POINTER(ctypes.c_uint64),    # path
    ctypes.POINTER(ctypes.c_uint64),    # path_edges
    ctypes.POINTER(ctypes.c_uint64)     # path_len
]

def bidirectional_dijkstra(n, m, edges, weights, s, t):
    """
    Gọi hàm C bidirectional_dijkstra để tìm đường đi ngắn nhất.
    
    Args:
        n: Số đỉnh (int)
        m: Số cạnh (int)
        edges: Mảng numpy uint64 chứa cặp (u, v) cho mỗi cạnh, shape (2*m,)
        weights: Mảng numpy double chứa trọng số, shape (m,)
        s: Chỉ số đỉnh nguồn (int)
        t: Chỉ số đỉnh đích (int)
    
    Returns:
        tuple: (dist, path, path_edges)
            - dist (float): Khoảng cách ngắn nhất, hoặc -1.0 nếu không có đường.
            - path (np.ndarray): Mảng các chỉ số đỉnh trên đường đi.
            - path_edges (np.ndarray): Mảng các chỉ số cạnh tương ứng.
    """
    # Kiểm tra kích thước đầu vào
    if edges.shape != (2 * m,) or weights.shape != (m,) or n <= 0 or m < 0:
        raise ValueError("Invalid input dimensions")

    # Chuyển đổi đầu vào thành mảng ctypes
    edges_arr = (ctypes.c_uint64 * (2 * m))(*edges)
    weights_arr = (ctypes.c_double * m)(*weights)
    # Dự phòng kích thước tối đa cho path (tối đa n đỉnh)
    path_arr = (ctypes.c_uint64 * n)()
    path_edges_arr = (ctypes.c_uint64 * (n - 1))()
    path_len = ctypes.c_uint64(0)

    # Gọi hàm C
    dist = lib.bidirectional_dijkstra(
        n,
        m,
        edges_arr,
        weights_arr,
        s,
        t,
        path_arr,
        path_edges_arr,
        ctypes.byref(path_len)
    )

    # Chuyển path và path_edges thành numpy array
    if path_len.value > 0:
        path = np.array([path_arr[i] for i in range(path_len.value)], dtype=np.uint64)
        path_edges = np.array([path_edges_arr[i] for i in range(path_len.value - 1)], dtype=np.uint64)
    else:
        path = np.array([], dtype=np.uint64)
        path_edges = np.array([], dtype=np.uint64)

    return dist, path, path_edges