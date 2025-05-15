import ctypes
import numpy as np

# Load thư viện
lib = ctypes.cdll.LoadLibrary('/home/plg/catkin_ws/src/global_path_planning/lib/libformation.so')

# Định nghĩa kiểu trả về và đối số của hàm formation
lib.formation.restype = ctypes.c_int
lib.formation.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # zinit
    ctypes.POINTER(ctypes.c_double),  # g
    ctypes.POINTER(ctypes.c_double),  # A
    ctypes.POINTER(ctypes.c_double),  # b
    ctypes.c_int,                    # m
    ctypes.POINTER(ctypes.c_double)  # zout
]

def formation(zinit, g, A, b):
    # Kiểm tra kích thước đầu vào
    if len(zinit) != 6 or len(g) != 2 or A.shape[1] != 2 or A.shape[0] != len(b):
        raise ValueError("Invalid input dimensions")

    m = A.shape[0]  # Số ràng buộc

    # Chuyển đổi đầu vào thành mảng ctypes
    zinit_arr = (ctypes.c_double * 6)(*zinit)
    g_arr = (ctypes.c_double * 2)(*g)
    A_flat = A.flatten()
    A_arr = (ctypes.c_double * (m * 2))(*A_flat)
    b_arr = (ctypes.c_double * m)(*b)
    zout_arr = (ctypes.c_double * 6)()

    # Gọi hàm C
    status = lib.formation(
        zinit_arr,
        g_arr,
        A_arr,
        b_arr,
        m,
        zout_arr
    )

    # Chuyển zout thành numpy array
    zout = np.array([zout_arr[i] for i in range(6)])

    return status, zout

# # Ví dụ sử dụng
# if __name__ == "__main__":
#     # Đầu vào mẫu
#     zinit = np.array([3.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # t_x, t_y, theta, theta_1, theta_2, theta_3
#     g = np.array([2.0, 4.0])  # g_x, g_y
#     A = np.array([
#         [-0.63227049, -0.77474772],
#         [0.38237612, 0.92400677],
#         [1.0, 0.0],
#         [0.0, 1.0],
#         [-1.0, 0.0],
#         [0.0, -1.0]
#     ])
#     b = np.array([-1.38252855, 3.92711138, 5.0, 5.0, 0.0, 0.0])

#     # Gọi hàm
#     status, zout = formation(zinit, g, A, b)

#     # In kết quả
#     print("Status:", status)
#     print("Optimized configuration:")
#     print(f"t_x = {zout[0]:.4f}, t_y = {zout[1]:.4f}")
#     print(f"theta = {zout[2]:.4f}")
#     print(f"theta_1 = {zout[3]:.4f}, theta_2 = {zout[4]:.4f}, theta_3 = {zout[5]:.4f}")
#     if status == 1:
#         print("Optimization successful")
#     else:
#         print("Optimization failed")