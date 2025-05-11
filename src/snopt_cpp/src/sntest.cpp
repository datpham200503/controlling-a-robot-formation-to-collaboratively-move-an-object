#include <ros/ros.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "snoptProblem.hpp"

using namespace std;

// Ma trận A và vector b từ IRIS
const double A[6][2] = {
    {-0.63227049, -0.77474772},
    {0.38237612, 0.92400677},
    {1.0, 0.0},
    {0.0, 1.0},
    {-1.0, 0.0},
    {0.0, -1.0}
};
const double b[6] = {-1.38252855, 3.92711138, 5.0, 5.0, 0.0, 0.0};

// Hàm tính vị trí các đỉnh của đội hình dựa trên cấu hình z
void computeFormationVertices(double* z, double* ru, double* robot_dims, vector<double>& vertices) {
    double t_x = z[0], t_y = z[1], theta = z[2];
    double theta_1 = z[3], theta_2 = z[4], theta_3 = z[5];
    double l_r = robot_dims[0], w_r = robot_dims[1];
    double three_angles[3] = {0.0 + theta, 2 * M_PI / 3 + theta, 4 * M_PI / 3 + theta};
    double cos_theta = cos(theta), sin_theta = sin(theta);

    // Tính toán 3 đỉnh của vật thể (tam giác)
    for (int i = 0; i < 3; ++i) {
        double x_local = ru[2 * i];
        double y_local = ru[2 * i + 1];
        vertices[2 * i] = t_x + cos_theta * x_local - sin_theta * y_local;
        vertices[2 * i + 1] = t_y + sin_theta * x_local + cos_theta * y_local;
    }

    // Tính toán 4 đỉnh của mỗi robot (hình chữ nhật)
    for (int i = 0; i < 3; ++i) {
        double theta_i = z[3 + i] + three_angles[i];
        double cos_theta_i = cos(theta_i), sin_theta_i = sin(theta_i);
        double x_g = vertices[2 * i], y_g = vertices[2 * i + 1];
        double a_i = ru[6 + i]; // Khoảng cách cánh tay a_i
        // Tính tâm robot
        double x_center = x_g + (a_i + l_r / 2) * cos_theta_i;
        double y_center = y_g + (a_i + l_r / 2) * sin_theta_i;
        // ROS_INFO("Center: (%f, %f)", x_center, y_center);

        // 4 đỉnh của robot: (+l_r/2, +w_r/2), (-l_r/2, +w_r/2), (-l_r/2, -w_r/2), (l_r/2, -w_r/2)
        double local_corners[8] = { l_r/2, w_r/2, -l_r/2, w_r/2, -l_r/2, -w_r/2, l_r/2, -w_r/2 };
        for (int j = 0; j < 4; ++j) {
            double x_local = local_corners[2 * j];
            double y_local = local_corners[2 * j + 1];
            vertices[6 + 8 * i + 2 * j] = x_center + cos_theta_i * x_local - sin_theta_i * y_local;
            vertices[6 + 8 * i + 2 * j + 1] = y_center + sin_theta_i * x_local + cos_theta_i * y_local;
        }
    }
}

// // Hàm mục tiêu và gradient
void objectiveFunction(int* Status, int* n, double x[],
                      int* needF, int* neF, double F[],
                      int* needG, int* neG, double G[],
                      char* cu, int* lencu,
                      int iu[], int* leniu, 
                      double ru[], int* lenru) {
    if (*needF > 0) {
        // Mục tiêu: giảm thiểu khoảng cách đến g(t_1) = (5, 5)
        double g_x = 1.4, g_y = 3.2; // Vị trí mục tiêu
        double t_x = x[0], t_y = x[1];
        F[0] = (t_x - g_x) * (t_x - g_x) + (t_y - g_y) * (t_y - g_y); // ||t - g||^2

        vector<double> vertices(30, 0.0);
        computeFormationVertices(x, &ru[0], &ru[9], vertices);

        ROS_INFO("Computed vertices:");
        for (int i = 0; i < 15; ++i) {
            ROS_INFO("Vertex %d: (x = %f, y = %f)", i, vertices[2 * i], vertices[2 * i + 1]);
        }

        // Ràng buộc vùng lồi: A * [vx, vy] <= b cho mỗi đỉnh
        for (int i = 0; i < 15; ++i) { // 15 đỉnh (3 vật thể + 12 robot)
            double vx = vertices[2 * i];
            double vy = vertices[2 * i + 1];
            for (int j = 0; j < 6; ++j) { // 6 bất đẳng thức từ A, b
                F[1 + i * 6 + j] = A[j][0] * vx + A[j][1] * vy - b[j];
            }
        }
    }

    if (*needG > 0) {
        // Gradient của hàm mục tiêu
        double g_x = 1.4, g_y = 3.2;
        G[0] = 2 * (x[0] - g_x); // dF[0]/dt_x
        G[1] = 2 * (x[1] - g_y); // dF[0]/dt_y
        // G[2] = 0.0; // dF[0]/dtheta
        // G[3] = 0.0; // dF[0]/dtheta_1
        // G[4] = 0.0; // dF[0]/dtheta_2
        // G[5] = 0.0; // dF[0]/dtheta_3

        // Gradient cho 3 đỉnh vật thể (chỉ tính 3 đỉnh, mỗi đỉnh 6 ràng buộc, mỗi ràng buộc 3 gradient)
        for (int i = 0; i < 3; ++i) {
            double x_local = ru[2 * i];
            double y_local = ru[2 * i + 1];
            double cos_theta = cos(x[2]), sin_theta = sin(x[2]);

            for (int j = 0; j < 6; ++j) {
                int F_idx = 1 + i * 6 + j; // Chỉ số hàm F bắt đầu từ F[1]
                int G_idx = (F_idx - 1) * 3; // Mỗi ràng buộc có 3 gradient (t_x, t_y, theta)
                // dF/dt_x = A[j][0]
                G[G_idx + 0] = A[j][0];
                // dF/dt_y = A[j][1]
                G[G_idx + 1] = A[j][1];
                // dF/dtheta
                double dvx_dtheta = -sin_theta * x_local - cos_theta * y_local;
                double dvy_dtheta = cos_theta * x_local - sin_theta * y_local;
                G[G_idx + 2] = A[j][0] * dvx_dtheta + A[j][1] * dvy_dtheta;
            }
        }

        // Gradient cho 12 đỉnh robot
        for (int i = 0; i < 3; ++i) { // 3 robot
            double x_local = ru[2 * i];
            double y_local = ru[2 * i + 1];
            double cos_theta = cos(x[2]), sin_theta = sin(x[2]);
            double theta_i = x[3 + i] + (2 * M_PI / 3 * i + x[2]);
            double cos_theta_i = cos(theta_i), sin_theta_i = sin(theta_i);
            double a_i = ru[6 + i];
            double l_r = ru[9], w_r = ru[10];
            double x_g = x[0] + cos_theta * x_local - sin_theta * y_local;
            double y_g = x[1] + sin_theta * x_local + cos_theta * y_local;
            double x_center = x_g + (a_i + l_r / 2) * cos_theta_i;
            double y_center = y_g + (a_i + l_r / 2) * sin_theta_i;
            // ROS_INFO("Center: (%f, %f)", x_center, y_center);
            for (int k = 0; k < 4; ++k) { // 4 đỉnh của robot
                double local_corners[8] = { l_r/2, w_r/2, -l_r/2, w_r/2, -l_r/2, -w_r/2, l_r/2, -w_r/2 };
                double x_local_corner = local_corners[2 * k];
                double y_local_corner = local_corners[2 * k + 1];

                for (int j = 0; j < 6; ++j) {
                    int F_idx = 1 + (3 + i * 4 + k) * 6 + j;
                    G[F_idx * 6 + 0] = A[j][0];
                    G[F_idx * 6 + 1] = A[j][1];

                    // Gradient theo theta
                    double dvx_dtheta_g = -sin_theta * x_local - cos_theta * y_local;
                    double dvy_dtheta_g = cos_theta * x_local - sin_theta * y_local;
                    double dxcenter_dtheta = dvx_dtheta_g + (a_i + l_r / 2) * (-sin_theta_i) * 1.0;
                    double dycenter_dtheta = dvy_dtheta_g + (a_i + l_r / 2) * (cos_theta_i) * 1.0;
                    double dvx_dtheta = dxcenter_dtheta + (-sin_theta_i * x_local_corner - cos_theta_i * y_local_corner) * 1.0;
                    double dvy_dtheta = dycenter_dtheta + (cos_theta_i * x_local_corner - (-sin_theta_i) * y_local_corner) * 1.0;
                    G[F_idx * 6 + 2] = A[j][0] * dvx_dtheta + A[j][1] * dvy_dtheta;

                    // Gradient theo theta_i
                    double dxcenter_dtheta_i = (a_i + l_r / 2) * (-sin_theta_i);
                    double dycenter_dtheta_i = (a_i + l_r / 2) * (cos_theta_i);
                    double dvx_dtheta_i = dxcenter_dtheta_i + (-sin_theta_i * x_local_corner - cos_theta_i * y_local_corner);
                    double dvy_dtheta_i = dycenter_dtheta_i + (cos_theta_i * x_local_corner - (-sin_theta_i) * y_local_corner);
                    G[F_idx * 6 + (3 + i)] = A[j][0] * dvx_dtheta_i + A[j][1] * dvy_dtheta_i;
                }
            }  
        }
    }
}

int main(int argc, char** argv) {
    // Khởi tạo node ROS
    ros::init(argc, argv, "sntest_node");
    ros::NodeHandle nh;
    ROS_INFO("SNOPT Collaborative Transport Node started");



    // Test hàm computeFormationVertices
    // double z[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // t_x, t_y, theta, theta_1, theta_2, theta_3
    // double robot_dims[2] = {0.3, 0.3}; // Chiều dài và chiều rộng robot
    // double ru[9] = {
    //     0.15 * cos(0), 0.15 * sin(0),
    //     0.15 * cos(2 * M_PI / 3), 0.15 * sin(2 * M_PI / 3),
    //     0.15 * cos(4 * M_PI / 3), 0.15 * sin(4 * M_PI / 3),
    //     
    //     0.2, 0.2, 0.2 // a_1, a_2, a_3
    // };
    // std::vector<double> vertices(30, 0.0); // 15 đỉnh x 2 (x, y)
    // computeFormationVertices(z, ru, robot_dims, vertices);
    // ROS_INFO("Computed vertices:");
    // for (int i = 0; i < 15; ++i) {
    //     ROS_INFO("Vertex %d: (%f, %f)", i, vertices[2 * i], vertices[2 * i + 1]);
    // }



    // Thiết lập SNOPT
    snoptProblemA ToyProb;

    int n = 6; // Biến: t_x, t_y, theta, theta_1, theta_2, theta_3
    int neF = 91; // 1 mục tiêu + 15 đỉnh * 6 ràng buộc vùng lồi
    int nS = 0, nInf;
    double sInf;

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

    int ObjRow = 0;
    double ObjAdd = 0;

    int Cold = 0;

    // Thiết lập giới hạn
    xlow[0] = -1e20; xupp[0] = 1e20; // t_x
    xlow[1] = -1e20; xupp[1] = 1e20; // t_y
    xlow[2] = -1e20; xupp[2] = 1e20; // theta
    xlow[3] = -1e20; xupp[3] = 1e20; // theta_1
    xlow[4] = -1e20; xupp[4] = 1e20; // theta_2
    xlow[5] = -1e20; xupp[5] = 1e20; // theta_3
    for (int i = 0; i < n; ++i) xstate[i] = 0;

    Flow[0] = -1e20; Fupp[0] = 1e20; // Mục tiêu
    for (int i = 1; i < neF; ++i) {
        Flow[i] = -1e20; Fupp[i] = 0.0; // Ràng buộc vùng lồi
    }
    for (int i = 0; i < neF; ++i) Fmul[i] = 0;

    // Điểm khởi tạo
    x[0] = 1.9; x[1] = 0.9; // t_x, t_y
    x[2] = 0.0; // theta
    x[3] = 0.0; x[4] = 0.0; x[5] = 0.0; // theta_1, theta_2, theta_3

    // Thiết lập Jacobian
    int lenA = 0; // Không có ràng buộc tuyến tính
    int* iAfun = new int[lenA];
    int* jAvar = new int[lenA];
    double* A = new double[lenA];

    int lenG = 91 * 6; // 91 hàm, mỗi hàm có tối đa 6 gradient (t_x, t_y, theta, theta_1, theta_2, theta_3)
    int* iGfun = new int[lenG];
    int* jGvar = new int[lenG];

    int neA = 0, neG = 344;
    
    // Gradient cho hàm mục tiêu
    iGfun[0] = 0; jGvar[0] = 0; // dF[0]/dt_x
    iGfun[1] = 0; jGvar[1] = 1; // dF[0]/dt_y
    // Gradient cho ràng buộc vùng lồi
    // int g_idx = 2;
    // // 3 đỉnh vật thể
    // for (int i = 0; i < 3; ++i) {
    //     for (int j = 0; j < 6; ++j) {
    //         int F_idx = 1 + i * 6 + j;
    //         // dF/dt_x
    //         iGfun[g_idx] = F_idx; jGvar[g_idx] = 0; g_idx++;
    //         // dF/dt_y
    //         iGfun[g_idx] = F_idx; jGvar[g_idx] = 1; g_idx++;
    //         // dF/dtheta
    //         iGfun[g_idx] = F_idx; jGvar[g_idx] = 2; g_idx++;
    //     }
    // }
    // // 12 đỉnh robot
    // for (int i = 0; i < 3; ++i) {
    //     for (int k = 0; k < 4; ++k) {
    //         for (int j = 0; j < 6; ++j) {
    //             int F_idx = 1 + (3 + i * 4 + k) * 6 + j;
    //             iGfun[g_idx] = F_idx; jGvar[g_idx] = 0; g_idx++; // dF/dt_x
    //             iGfun[g_idx] = F_idx; jGvar[g_idx] = 1; g_idx++; // dF/dt_y
    //             iGfun[g_idx] = F_idx; jGvar[g_idx] = 2; g_idx++; // dF/dtheta
    //             iGfun[g_idx] = F_idx; jGvar[g_idx] = 3 + i; g_idx++; // dF/dtheta_i
    //         }
    //     }
    // }

    double* ru = new double[11];
    ru[0] = 0.15 * cos(0.0); ru[1] = 0.15 * sin(0.0);
    ru[2] = 0.15 * cos(2 * M_PI / 3); ru[3] = 0.15 * sin(2 * M_PI / 3);
    ru[4] = 0.15 * cos(4 * M_PI / 3); ru[5] = 0.15 * sin(4 * M_PI / 3);
    ru[6] = 0.2; ru[7] = 0.2; ru[8] = 0.2; // a_1, a_2, a_3
    ru[9] = 0.3; ru[10] = 0.3; // Chiều dài và chiều rộng robot
    int lenru = 11;

    // Khởi tạo SNOPT
    ToyProb.initialize("", 1); // Không xuất summary
    ToyProb.setProbName("CollaborativeTransport");
    ToyProb.setIntParameter("Derivative option", 1);
    ToyProb.setIntParameter("Major Iteration limit", 250);
    ToyProb.setIntParameter("Verify level", 3);
    ToyProb.setUserR(ru, lenru);

    // Giải bài toán
    int status = ToyProb.solve(Cold, neF, n, ObjAdd, ObjRow, objectiveFunction,
                               iAfun, jAvar, A, neA,
                               iGfun, jGvar, neG,
                               xlow, xupp, Flow, Fupp,
                               x, xstate, xmul,
                               F, Fstate, Fmul,
                               nS, nInf, sInf);

    // Xuất kết quả
    ROS_INFO("Optimized configuration:");
    ROS_INFO("t_x = %f, t_y = %f", x[0], x[1]);
    ROS_INFO("theta = %f", x[2]);
    ROS_INFO("theta_1 = %f, theta_2 = %f, theta_3 = %f", x[3], x[4], x[5]);
    if (nInf == 0 && sInf == 0) {
        ROS_INFO("Optimization successful");
    } else {
        ROS_INFO("Optimization failed: nInf = %d, sInf = %f", nInf, sInf);
    }

    // Dọn dẹp
    delete[] iAfun; delete[] jAvar; delete[] A;
    delete[] iGfun; delete[] jGvar;
    delete[] x; delete[] xlow; delete[] xupp;
    delete[] xmul; delete[] xstate;
    delete[] F; delete[] Flow; delete[] Fupp;
    delete[] Fmul; delete[] Fstate;
    delete[] ru;

    return 0;
}