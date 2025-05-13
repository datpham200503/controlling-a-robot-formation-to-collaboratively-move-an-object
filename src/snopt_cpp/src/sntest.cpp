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

// Hàm kiểm tra điểm nằm trong vùng lồi
bool isPointInPolytope(double px, double py) {
    for (int j = 0; j < 6; ++j) {
        if (A[j][0] * px + A[j][1] * py > b[j]) return false;
    }
    return true;
}

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
        // Mục tiêu: giảm thiểu khoảng cách đến g(t_1) = (g_x, g_y)
        double g_x = 2.0, g_y = 4.0; // Vị trí mục tiêu
        double t_x = x[0], t_y = x[1];
        F[0] = (t_x - g_x) * (t_x - g_x) + (t_y - g_y) * (t_y - g_y); // ||t - g||^2

        vector<double> vertices(30, 0.0);
        computeFormationVertices(x, &ru[0], &ru[9], vertices);

        // ROS_INFO("Computed vertices:");
        // for (int i = 0; i < 15; ++i) {
        //     ROS_INFO("Vertex %d: (x = %f, y = %f)", i, vertices[2 * i], vertices[2 * i + 1]);
        // }

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
        int g_idx = 0;

        // Gradient của hàm mục tiêu
        double g_x = 2.0, g_y =4.0;
        G[g_idx++] = 2 * (x[0] - g_x); // dF[0]/dt_x
        G[g_idx++] = 2 * (x[1] - g_y); // dF[0]/dt_y
        // G[2] = 0.0; // dF[0]/dtheta
        // G[3] = 0.0; // dF[0]/dtheta_1
        // G[4] = 0.0; // dF[0]/dtheta_2
        // G[5] = 0.0; // dF[0]/dtheta_3

        // Gradient cho 3 đỉnh vật thể (chỉ tính 3 đỉnh, mỗi đỉnh 6 ràng buộc, mỗi ràng buộc 3 gradient)
        for (int i = 0; i < 3; ++i) {
            // x_local, y_local là tọa độ vật thể trong hệ tọa độ cục bộ (hệ trục nằm ở tâm vật thể)
            double x_local = ru[2 * i];
            double y_local = ru[2 * i + 1];
            double cos_theta = cos(x[2]), sin_theta = sin(x[2]);

            for (int j = 0; j < 6; ++j) {
                G[g_idx++] = A[j][0]; // dF/dt_x
                G[g_idx++] = A[j][1]; // dF/dt_y
                
                /*
                    vx = t_x + cos_theta * x_local - sin_theta * y_local
                    vy = t_y + sin_theta * x_local + cos_theta * y_local

                    F = A[j][0] * vx + A[j][1] * vy - b[j]

                    Đạo hàm của F theo theta được tính như sau:
                        ==> dF/dtheta = A[j][0] * dvx/dtheta + A[j][1] * dvy/dtheta
                        ==> dvx/dtheta = -sin_theta * x_local - cos_theta * y_local
                        ==> dvy/dtheta = cos_theta * x_local - sin_theta * y_local
                    
                */
                double dvx_dtheta = -sin_theta * x_local - cos_theta * y_local;
                double dvy_dtheta = cos_theta * x_local - sin_theta * y_local;
                
                G[g_idx++] = A[j][0] * dvx_dtheta + A[j][1] * dvy_dtheta; // dF/dtheta
            }
        }

        // Gradient cho 12 đỉnh robot
        for (int i = 0; i < 3; ++i) { // 3 robot
            // Toạ độ 3 đỉnh của vật thể trong hệ cục bộ
            double x_local = ru[2 * i];
            double y_local = ru[2 * i + 1];
            double cos_theta = cos(x[2]), sin_theta = sin(x[2]);
            // Góc quay của robot trong hệ toàn cục
            //     theta_i = theta_i (cục bộ) + góc quay ban đầu (0, 120, 270) + góc theta
            double theta_i = x[3 + i] + (2 * M_PI / 3 * i) + x[2];
            double cos_theta_i = cos(theta_i), sin_theta_i = sin(theta_i);
            // Các thông số của robot
            double a_i = ru[6 + i];
            double l_r = ru[9], w_r = ru[10];
            // Tọa độ 3 đỉnh của vật thể trong hệ tọa độ toàn cục
            double x_g = x[0] + cos_theta * x_local - sin_theta * y_local;
            double y_g = x[1] + sin_theta * x_local + cos_theta * y_local;
            // Tọa độ trọng tâm của robot trong hệ tọa độ toàn cục
            double x_center = x_g + (a_i + l_r / 2) * cos_theta_i;
            double y_center = y_g + (a_i + l_r / 2) * sin_theta_i;
            
            // ROS_INFO("Center: (%f, %f)", x_center, y_center);

            for (int k = 0; k < 4; ++k) { // 4 đỉnh của robot
                // Tọa độ bốn đỉnh của robot trong hệ trục cục bộ
                double local_corners[8] = { l_r/2, w_r/2, -l_r/2, w_r/2, -l_r/2, -w_r/2, l_r/2, -w_r/2 };
                double x_local_corner = local_corners[2 * k];
                double y_local_corner = local_corners[2 * k + 1];

                for (int j = 0; j < 6; ++j) { // 6 ràng buộc
                    G[g_idx++] = A[j][0]; // dF/dt_x
                    G[g_idx++] = A[j][1]; // dF/dt_y

                    /*
                        x[0] = t_x
                        x[1] = t_y
                        x[2] = theta
                        x[3..5] = theta_i
                        
                        theta_i = x[3..5] + (2 * M_PI / 3 * i) + x[2]
                        
                        x_g = x[0] + cos_theta * x_local - sin_theta * y_local;
                        y_g = x[1] + sin_theta * x_local + cos_theta * y_local;

                        x_center = x_g + (a_i + l_r / 2) * cos_theta_i;
                        y_center = y_g + (a_i + l_r / 2) * sin_theta_i;

                        vx = x_center + cos_theta_i * x_local_corner - sin_theta_i * y_local_corner
                        vy = y_center + sin_theta_i * x_local_corner + cos_theta_i * y_local_corner

                        F = A[j][0] * vx + A[j][1] * vy - b[j]

                        Đạo hàm của F tính theo t_x:
                            ==> dF/dt_x = A[j][0] * dvx/dt_x + A[j][1] * dvy/dt_x
                            ==> dvx/dt_x = dx_center/dt_x 
                            ==> dvy/dt_x = dy_center/dt_x 
                            ==> dx_center/dt_x = dx_g/dt_x 
                            ==> dy_center/dt_x = dy_g/dt_x 
                            ==> dx_g/dt_x = 1
                            ==> dy_g/dt_x = 0

                        Đạo hàm của F tính theo t_y:
                            ==> dF/dt_y = A[j][0] * dvx/dt_y + A[j][1] * dvy/dt_y
                            ==> dvx/dt_y = dx_center/dt_y 
                            ==> dvy/dt_y = dy_center/dt_y 
                            ==> dx_center/dt_y = dx_g/dt_y 
                            ==> dy_center/dt_y = dy_g/dt_y 
                            ==> dx_g/dt_y = 0
                            ==> dy_g/dt_y = 1

                        Đạo hàm của F tính theo theta:
                            ==> dF/dtheta = A[j][0] * dvx/dtheta + A[j][1] * dvy/dtheta
                            ==> dvx/dtheta = dx_center/dtheta - sin_theta_i * x_local_corner - cos_theta_i * y _local_corner
                            ==> dvy/dtheta = dy_center/dtheta + cos_theta_i * x_local_corner - sin_theta_i * y _local_corner
                            ==> dx_center/dtheta = dx_g/dtheta - (a_i + l_r / 2) * sin_theta_i
                            ==> dy_center/dtheta = dy_g/dtheta + (a_i + l_r / 2) * cos_theta_i
                            ==> dx_g/dtheta = -sin_theta * x_local - cos_theta * y_local
                            ==> dy_g/dtheta = cos_theta * x_local - sin_theta * y_local

                        Đạo hàm của F tính theo theta_i_local:
                            ==> dF/dtheta_i = A[j][0] * dvx/dtheta_i + A[j][1] * dvy/dtheta_i
                            ==> dvx/dtheta_i = dx_center/dtheta_i - sin_theta_i * x_local_corner - cos_theta_i * y _local_corner
                            ==> dvy/dtheta_i = dy_center/dtheta_i + cos_theta_i * x_local_corner - sin_theta_i * y _local_corner
                            ==> dx_center/dtheta_i = dx_g/dtheta_i - (a_i + l_r / 2) * sin_theta_i
                            ==> dy_center/dtheta_i = dy_g/dtheta_i + (a_i + l_r / 2) * cos_theta_i
                            ==> dx_g/dtheta_i = 0
                            ==> dy_g/dtheta_i = 0
                    */

                    // Gradient theo theta
                    double dxg_dtheta = -sin_theta * x_local - cos_theta * y_local;
                    double dyg_dtheta = cos_theta * x_local - sin_theta * y_local;
                    double dxcenter_dtheta = dxg_dtheta + (a_i + l_r / 2) * (-sin_theta_i) * 1.0;
                    double dycenter_dtheta = dyg_dtheta + (a_i + l_r / 2) * (cos_theta_i) * 1.0;
                    double dvx_dtheta = dxcenter_dtheta + (-sin_theta_i * x_local_corner - cos_theta_i * y_local_corner) * 1.0;
                    double dvy_dtheta = dycenter_dtheta + (cos_theta_i * x_local_corner - sin_theta_i * y_local_corner) * 1.0;
                    G[g_idx++] = A[j][0] * dvx_dtheta + A[j][1] * dvy_dtheta;

                    // Gradient theo theta_i
                    double dxcenter_dtheta_i = (a_i + l_r / 2) * (-sin_theta_i);
                    double dycenter_dtheta_i = (a_i + l_r / 2) * (cos_theta_i);
                    double dvx_dtheta_i = dxcenter_dtheta_i + (-sin_theta_i * x_local_corner - cos_theta_i * y_local_corner);
                    double dvy_dtheta_i = dycenter_dtheta_i + (cos_theta_i * x_local_corner - sin_theta_i * y_local_corner);
                    G[g_idx++] = A[j][0] * dvx_dtheta_i + A[j][1] * dvy_dtheta_i;
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
    xlow[0] = -1e20; xupp[0] = 1e20;            // -inf < t_x < inf
    xlow[1] = -1e20; xupp[1] = 1e20;            // -inf < t_y < inf
    xlow[2] = -1e20; xupp[2] = 1e20;            // -inf < theta < inf
    xlow[3] = -M_PI / 2; xupp[3] = M_PI / 2;    // -pi/2 < theta_1 < pi/2
    xlow[4] = -M_PI / 2; xupp[4] = M_PI / 2;    // -pi/2 < theta_2 < pi/2
    xlow[5] = -M_PI / 2; xupp[5] = M_PI / 2;    // -pi/2 < theta_3 < pi/2
    for (int i = 0; i < n; ++i) xstate[i] = 0;

    Flow[0] = -1e20; Fupp[0] = 1e20; // Mục tiêu
    for (int i = 1; i < neF; ++i) {
        Flow[i] = -1e20; Fupp[i] = 0.0; // Ràng buộc vùng lồi
    }
    for (int i = 0; i < neF; ++i) {
        Fmul[i] = 0;
        Fstate[i] = 0;
    }

    // Điểm khởi tạo
    x[0] = 3.0; x[1] = 1.0; // t_x, t_y
    x[2] = 0.0; // theta
    x[3] = 0.0; x[4] = 0.0; x[5] = 0.0; // theta_1, theta_2, theta_3

    double* ru = new double[11];
    ru[0] = 0.15 * cos(0.0); ru[1] = 0.15 * sin(0.0);
    ru[2] = 0.15 * cos(2 * M_PI / 3); ru[3] = 0.15 * sin(2 * M_PI / 3);
    ru[4] = 0.15 * cos(4 * M_PI / 3); ru[5] = 0.15 * sin(4 * M_PI / 3);
    ru[6] = 0.2; ru[7] = 0.2; ru[8] = 0.2; // a_1, a_2, a_3
    ru[9] = 0.3; ru[10] = 0.3; // Chiều dài và chiều rộng robot
    int lenru = 11;

    // Kiểm tra tính khả thi của điểm khởi tạo
    // vector<double> vertices(30);
    // computeFormationVertices(x, &ru[0], &ru[9], vertices);
    // bool feasible = true;
    // for (int i = 0; i < 15; ++i) {
    //     if (!isPointInPolytope(vertices[2*i], vertices[2*i+1])) {
    //         ROS_WARN("Initial vertex %d is outside polytope!", i);
    //         feasible = false;
    //     }
    // }

    // if (feasible) {
    //     ROS_INFO("Initial configuration is feasible");
    // } else {
    //     ROS_WARN("Initial configuration is not feasible");
    // }

    // Thiết lập Jacobian
    int lenA = 0; // Không có ràng buộc tuyến tính
    int* iAfun = new int[lenA];
    int* jAvar = new int[lenA];
    double* linearA = new double[lenA];

    int lenG = 546; // 91 hàm, mỗi hàm có tối đa 6 gradient (t_x, t_y, theta, theta_1, theta_2, theta_3)
    int* iGfun = new int[lenG];
    int* jGvar = new int[lenG];

    int neA = 0, neG = 344; // Có 344 hàm G khác không
    
    /*
        Hàm Gradient có dạng (trường hợp ma trận Ax=b có 6 ràng buộc):

        |    G[0]   G[1]    G[2]    G[3]    G[4]    G[5]    |   --> Gradient của hàm mục tiêu
        |    G[6]   G[7]    G[8]    G[9]    G[10]   G[11]   |   --> Gradient của hàm F[1]
        |    G[12]  G[13]   G[14]   G[15]   G[16]   G[17]   |   --> Gradient của hàm F[2]
        |      :      :       :       :       :       :     |
        |      :      :       :       :       :       :     |
        |    G[108] G[109]  G[110]  G[111]  G[112]  G[113]  |   --> Gradient của hàm F[18]
        |    G[114] G[115]  G[116]  G[117]  G[118]  G[119]  |   --> Gradient của hàm F[19]
        |      :      :       :       :       :       :     |
        |      :      :       :       :       :       :     |
        |    G[540] G[541]  G[542]  G[543]  G[544]  G[545]  |   --> Gradient của hàm F[90]
    */

    // Gradient cho hàm mục tiêu
    int g_idx = 0;
    // Gradient cho hàm mục tiêu
    iGfun[g_idx] = 0; jGvar[g_idx] = 0; g_idx++; // dF[0]/dt_x
    iGfun[g_idx] = 0; jGvar[g_idx] = 1; g_idx++; // dF[0]/dt_y

    // Gradient cho ràng buộc vùng lồi
    // 3 đỉnh vật thể
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 6; ++j) {
            int F_idx = 1 + i * 6 + j;
            // dF/dt_x
            iGfun[g_idx] = F_idx; jGvar[g_idx] = 0; g_idx++;
            // dF/dt_y
            iGfun[g_idx] = F_idx; jGvar[g_idx] = 1; g_idx++;
            // dF/dtheta
            iGfun[g_idx] = F_idx; jGvar[g_idx] = 2; g_idx++;
        }
    }
    // 12 đỉnh robot
    for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < 4; ++k) {
            for (int j = 0; j < 6; ++j) {
                int F_idx = 1 + (3 + i * 4 + k) * 6 + j;
                iGfun[g_idx] = F_idx; jGvar[g_idx] = 0; g_idx++; // dF/dt_x
                iGfun[g_idx] = F_idx; jGvar[g_idx] = 1; g_idx++; // dF/dt_y
                iGfun[g_idx] = F_idx; jGvar[g_idx] = 2; g_idx++; // dF/dtheta
                iGfun[g_idx] = F_idx; jGvar[g_idx] = 3 + i; g_idx++; // dF/dtheta_i
            }
        }
    }

    // Khởi tạo SNOPT
    ToyProb.initialize("", 1); // Không xuất summary
    ToyProb.setProbName("CollaborativeTransport");
    ToyProb.setIntParameter("Derivative option", 1);
    ToyProb.setIntParameter("Major Iteration limit", 250);
    ToyProb.setIntParameter("Verify level", 3);
    ToyProb.setUserR(ru, lenru);

    // Giải bài toán
    int status = ToyProb.solve(Cold, neF, n, ObjAdd, ObjRow, objectiveFunction,
                               iAfun, jAvar, linearA, neA,
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
    if (status == 1) {
        ROS_INFO("Optimization successful");
    } else {
        ROS_INFO("Optimization failed: status = %d", status);
    }

    // Dọn dẹp
    delete[] iAfun; delete[] jAvar; delete[] linearA;
    delete[] iGfun; delete[] jGvar;
    delete[] x; delete[] xlow; delete[] xupp;
    delete[] xmul; delete[] xstate;
    delete[] F; delete[] Flow; delete[] Fupp;
    delete[] Fmul; delete[] Fstate;
    delete[] ru;

    return 0;
}