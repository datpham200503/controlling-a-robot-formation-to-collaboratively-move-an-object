#include <stdio.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "snoptProblem.hpp"

using namespace std;

// Hàm tính vị trí các đỉnh của đội hình
void computeFormationVertices(double* z, double* ru, double* robot_dims, vector<double>& vertices) {
    double t_x = z[0], t_y = z[1], theta = z[2];
    double theta_1 = z[3], theta_2 = z[4], theta_3 = z[5];
    double l_r = robot_dims[0], w_r = robot_dims[1];
    double three_angles[3] = {0.0 + theta, 2 * M_PI / 3 + theta, 4 * M_PI / 3 + theta};
    double cos_theta = cos(theta), sin_theta = sin(theta);

    vertices.resize(30);

    for (int i = 0; i < 3; ++i) {
        double x_local = ru[2 * i];
        double y_local = ru[2 * i + 1];
        vertices[2 * i] = t_x + cos_theta * x_local - sin_theta * y_local;
        vertices[2 * i + 1] = t_y + sin_theta * x_local + cos_theta * y_local;
    }

    for (int i = 0; i < 3; ++i) {
        double theta_i = z[3 + i] + three_angles[i];
        double cos_theta_i = cos(theta_i), sin_theta_i = sin(theta_i);
        double x_g = vertices[2 * i], y_g = vertices[2 * i + 1];
        double a_i = ru[6 + i];
        double x_center = x_g + (a_i + l_r / 2) * cos_theta_i;
        double y_center = y_g + (a_i + l_r / 2) * sin_theta_i;

        double local_corners[8] = { l_r/2, w_r/2, -l_r/2, w_r/2, -l_r/2, -w_r/2, l_r/2, -w_r/2 };
        for (int j = 0; j < 4; ++j) {
            double x_local = local_corners[2 * j];
            double y_local = local_corners[2 * j + 1];
            vertices[6 + 8 * i + 2 * j] = x_center + cos_theta_i * x_local - sin_theta_i * y_local;
            vertices[6 + 8 * i + 2 * j + 1] = y_center + sin_theta_i * x_local + cos_theta_i * y_local;
        }
    }
}

// Hàm mục tiêu và gradient
void objectiveFunction(int* Status, int* n, double x[],
                      int* needF, int* neF, double F[],
                      int* needG, int* neG, double G[],
                      char* cu, int* lencu,
                      int iu[], int* leniu,
                      double ru[], int* lenru) {
    double g_x = ru[11], g_y = ru[12]; // Lấy g_x, g_y từ ru
    const double* A = reinterpret_cast<const double*>(static_cast<std::uintptr_t>(ru[13]));
    const double* b = reinterpret_cast<const double*>(static_cast<std::uintptr_t>(ru[14]));
    int num_constraints = static_cast<int>(ru[15]);

    if (!A || !b || num_constraints <= 0) {
        *Status = -1;
        return;
    }

    if (*needF > 0) {
        double t_x = x[0], t_y = x[1];
        F[0] = (t_x - g_x) * (t_x - g_x) + (t_y - g_y) * (t_y - g_y);

        vector<double> vertices(30);
        computeFormationVertices(x, &ru[0], &ru[9], vertices);

        for (int i = 0; i < 15; ++i) {
            double vx = vertices[2 * i];
            double vy = vertices[2 * i + 1];
            for (int j = 0; j < num_constraints; ++j) {
                F[1 + i * num_constraints + j] = A[j * 2] * vx + A[j * 2 + 1] * vy - b[j];
            }
        }
    }

    if (*needG > 0) {
        int g_idx = 0;
        G[g_idx++] = 2 * (x[0] - g_x); // dF0/dx0
        G[g_idx++] = 2 * (x[1] - g_y); // dF0/dx1

        for (int i = 0; i < 3; ++i) {
            double x_local = ru[2 * i];
            double y_local = ru[2 * i + 1];
            double cos_theta = cos(x[2]), sin_theta = sin(x[2]);

            for (int j = 0; j < num_constraints; ++j) {
                G[g_idx++] = A[j * 2]; // dFi/dx0
                G[g_idx++] = A[j * 2 + 1]; // dFi/dx1
                double dvx_dtheta = -sin_theta * x_local - cos_theta * y_local;
                double dvy_dtheta = cos_theta * x_local - sin_theta * y_local;
                G[g_idx++] = A[j * 2] * dvx_dtheta + A[j * 2 + 1] * dvy_dtheta; // dFi/dx2
            }
        }

        for (int i = 0; i < 3; ++i) {
            double x_local = ru[2 * i];
            double y_local = ru[2 * i + 1];
            double cos_theta = cos(x[2]), sin_theta = sin(x[2]);
            double theta_i = x[3 + i] + (2 * M_PI / 3 * i) + x[2];
            double cos_theta_i = cos(theta_i), sin_theta_i = sin(theta_i);
            double a_i = ru[6 + i];
            double l_r = ru[9], w_r = ru[10];
            double x_g = x[0] + cos_theta * x_local - sin_theta * y_local;
            double y_g = x[1] + sin_theta * x_local + cos_theta * y_local;
            double x_center = x_g + (a_i + l_r / 2) * cos_theta_i;
            double y_center = y_g + (a_i + l_r / 2) * sin_theta_i;

            double local_corners[8] = { l_r/2, w_r/2, -l_r/2, w_r/2, -l_r/2, -w_r/2, l_r/2, -w_r/2 };
            for (int k = 0; k < 4; ++k) {
                double x_local_corner = local_corners[2 * k];
                double y_local_corner = local_corners[2 * k + 1];

                for (int j = 0; j < num_constraints; ++j) {
                    G[g_idx++] = A[j * 2]; // dFi/dx0
                    G[g_idx++] = A[j * 2 + 1]; // dFi/dx1
                    double dxg_dtheta = -sin_theta * x_local - cos_theta * y_local;
                    double dyg_dtheta = cos_theta * x_local - sin_theta * y_local;
                    double dxcenter_dtheta = dxg_dtheta + (a_i + l_r / 2) * (-sin_theta_i) * 1.0;
                    double dycenter_dtheta = dyg_dtheta + (a_i + l_r / 2) * (cos_theta_i) * 1.0;
                    double dvx_dtheta = dxcenter_dtheta + (-sin_theta_i * x_local_corner - cos_theta_i * y_local_corner) * 1.0;
                    double dvy_dtheta = dycenter_dtheta + (cos_theta_i * x_local_corner - sin_theta_i * y_local_corner) * 1.0;
                    G[g_idx++] = A[j * 2] * dvx_dtheta + A[j * 2 + 1] * dvy_dtheta; // dFi/dx2
                    double dxcenter_dtheta_i = (a_i + l_r / 2) * (-sin_theta_i);
                    double dycenter_dtheta_i = (a_i + l_r / 2) * (cos_theta_i);
                    double dvx_dtheta_i = dxcenter_dtheta_i + (-sin_theta_i * x_local_corner - cos_theta_i * y_local_corner);
                    double dvy_dtheta_i = dycenter_dtheta_i + (cos_theta_i * x_local_corner - sin_theta_i * y_local_corner);
                    G[g_idx++] = A[j * 2] * dvx_dtheta_i + A[j * 2 + 1] * dvy_dtheta_i; // dFi/dx(3+i)
                }
            }
        }
    }
}

// Hàm formation để gọi từ Python
extern "C" int formation(double* zinit, double* g, double* A, double* b, int m, double* zout) {
    // Kiểm tra đầu vào
    if (m <= 0) return -1;

    // Khởi tạo SNOPT
    snoptProblemA ToyProb;
    ToyProb.initialize("", 0);
    ToyProb.setProbName("CollaborativeTransport");
    ToyProb.setIntParameter("Derivative option", 1);
    ToyProb.setIntParameter("Major Iteration limit", 250);
    ToyProb.setIntParameter("Verify level", 3);

    // Thiết lập tham số
    int n = 6; // t_x, t_y, theta, theta_1, theta_2, theta_3
    int neF = 1 + 15 * m; // 1 mục tiêu + 15 đỉnh * m ràng buộc
    int neG = 2 + 3 * m * 3 + 3 * 4 * m * 4; // Gradient entries
    int lenG = neG;
    int lenA = 0; // Không có ràng buộc tuyến tính
    int nS = 0, nInf;
    double sInf;
    int ObjRow = 0;
    double ObjAdd = 0;
    int Cold = 0;

    // Cấp phát bộ nhớ
    double* x = new double[n];
    double* xlow = new double[n];
    double* xupp = new double[n];
    double* xmul = new double[n];
    int* xstate = new int[n];
    double* F = new double[neF];
    double* Flow = new double[neF];
    double* Fupp = new double[neF];
    double* Fmul = new double[neF];
    int* Fstate = new int[neF];
    int* iAfun = new int[lenA];
    int* jAvar = new int[lenA];
    double* linearA = new double[lenA];
    int* iGfun = new int[lenG];
    int* jGvar = new int[lenG];

    // Thiết lập ru
    int lenru = 16; // 11 original + g_x, g_y, A, b, num_constraints
    double* ru = new double[lenru];
    ru[0] = 0.15 * cos(0.0); ru[1] = 0.15 * sin(0.0);
    ru[2] = 0.15 * cos(2 * M_PI / 3); ru[3] = 0.15 * sin(2 * M_PI / 3);
    ru[4] = 0.15 * cos(4 * M_PI / 3); ru[5] = 0.15 * sin(4 * M_PI / 3);
    ru[6] = 0.2; ru[7] = 0.2; ru[8] = 0.2; // a_1, a_2, a_3
    ru[9] = 0.3; ru[10] = 0.3; // l_r, w_r
    ru[11] = g[0]; ru[12] = g[1]; // g_x, g_y
    ru[13] = static_cast<double>(reinterpret_cast<std::uintptr_t>(A));
    ru[14] = static_cast<double>(reinterpret_cast<std::uintptr_t>(b));
    ru[15] = static_cast<double>(m); // num_constraints
    ToyProb.setUserR(ru, lenru);

    // Thiết lập giới hạn
    xlow[0] = -1e20; xupp[0] = 1e20;
    xlow[1] = -1e20; xupp[1] = 1e20;
    xlow[2] = -1e20; xupp[2] = 1e20;
    xlow[3] = -M_PI / 2; xupp[3] = M_PI / 2;
    xlow[4] = -M_PI / 2; xupp[4] = M_PI / 2;
    xlow[5] = -M_PI / 2; xupp[5] = M_PI / 2;
    for (int i = 0; i < n; ++i) xstate[i] = 0;

    Flow[0] = -1e20; Fupp[0] = 1e20;
    for (int i = 1; i < neF; ++i) {
        Flow[i] = -1e20; Fupp[i] = 0.0;
    }
    for (int i = 0; i < neF; ++i) {
        Fmul[i] = 0;
        Fstate[i] = 0;
    }

    // Thiết lập điểm khởi tạo
    for (int i = 0; i < n; ++i) x[i] = zinit[i];

    // Thiết lập Jacobian
    int g_idx = 0;
    iGfun[g_idx] = 0; jGvar[g_idx] = 0; g_idx++; // dF0/dx0
    iGfun[g_idx] = 0; jGvar[g_idx] = 1; g_idx++; // dF0/dx1
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < m; ++j) {
            int F_idx = 1 + i * m + j;
            iGfun[g_idx] = F_idx; jGvar[g_idx] = 0; g_idx++;
            iGfun[g_idx] = F_idx; jGvar[g_idx] = 1; g_idx++;
            iGfun[g_idx] = F_idx; jGvar[g_idx] = 2; g_idx++;
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < 4; ++k) {
            for (int j = 0; j < m; ++j) {
                int F_idx = 1 + (3 + i * 4 + k) * m + j;
                iGfun[g_idx] = F_idx; jGvar[g_idx] = 0; g_idx++;
                iGfun[g_idx] = F_idx; jGvar[g_idx] = 1; g_idx++;
                iGfun[g_idx] = F_idx; jGvar[g_idx] = 2; g_idx++;
                iGfun[g_idx] = F_idx; jGvar[g_idx] = 3 + i; g_idx++;
            }
        }
    }

    // Giải bài toán
    int status = ToyProb.solve(Cold, neF, n, ObjAdd, ObjRow, objectiveFunction,
                               iAfun, jAvar, linearA, lenA,
                               iGfun, jGvar, neG,
                               xlow, xupp, Flow, Fupp,
                               x, xstate, xmul,
                               F, Fstate, Fmul,
                               nS, nInf, sInf);

    // Sao chép kết quả vào zout
    for (int i = 0; i < n; ++i) zout[i] = x[i];

    // Dọn dẹp
    delete[] x;
    delete[] xlow;
    delete[] xupp;
    delete[] xmul;
    delete[] xstate;
    delete[] F;
    delete[] Flow;
    delete[] Fupp;
    delete[] Fmul;
    delete[] Fstate;
    delete[] iAfun;
    delete[] jAvar;
    delete[] linearA;
    delete[] iGfun;
    delete[] jGvar;
    delete[] ru;

    return status; // Trả về trạng thái tối ưu hóa
}