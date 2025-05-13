#include <ros/ros.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <std_msgs/Float64MultiArray.h>
#include <Eigen/Dense>
#include <cstdint>
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
    // Lấy A và b từ ru
    Eigen::MatrixXd* A = reinterpret_cast<Eigen::MatrixXd*>(static_cast<std::uintptr_t>(ru[11]));
    Eigen::VectorXd* b = reinterpret_cast<Eigen::VectorXd*>(static_cast<std::uintptr_t>(ru[12]));
    if (!A || !b || A->rows() != b->size()) {
        ROS_ERROR("Invalid A or b in objectiveFunction");
        *Status = -1;
        return;
    }
    int num_constraints = A->rows();

    if (*needF > 0) {
        double g_x = 1.0, g_y = 4.0;
        double t_x = x[0], t_y = x[1];
        F[0] = (t_x - g_x) * (t_x - g_x) + (t_y - g_y) * (t_y - g_y);

        vector<double> vertices(30);
        computeFormationVertices(x, &ru[0], &ru[9], vertices);

        for (int i = 0; i < 15; ++i) {
            double vx = vertices[2 * i];
            double vy = vertices[2 * i + 1];
            for (int j = 0; j < num_constraints; ++j) {
                F[1 + i * num_constraints + j] = (*A)(j, 0) * vx + (*A)(j, 1) * vy - (*b)(j);
            }
        }
    }

    if (*needG > 0) {
        int g_idx = 0;
        double g_x = 1.0, g_y = 4.0;
        G[g_idx++] = 2 * (x[0] - g_x); // dF0/dx0
        G[g_idx++] = 2 * (x[1] - g_y); // dF0/dx1

        // 3 robots, mỗi robot có 3 gradient entries cho mỗi ràng buộc
        for (int i = 0; i < 3; ++i) {
            double x_local = ru[2 * i];
            double y_local = ru[2 * i + 1];
            double cos_theta = cos(x[2]), sin_theta = sin(x[2]);

            for (int j = 0; j < num_constraints; ++j) {
                G[g_idx++] = (*A)(j, 0); // dFi/dx0
                G[g_idx++] = (*A)(j, 1); // dFi/dx1
                double dvx_dtheta = -sin_theta * x_local - cos_theta * y_local;
                double dvy_dtheta = cos_theta * x_local - sin_theta * y_local;
                G[g_idx++] = (*A)(j, 0) * dvx_dtheta + (*A)(j, 1) * dvy_dtheta; // dFi/dx2
            }
        }

        // 3 robots, mỗi robot có 4 góc, mỗi góc có 4 gradient entries
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
                    G[g_idx++] = (*A)(j, 0); // dFi/dx0
                    G[g_idx++] = (*A)(j, 1); // dFi/dx1
                    double dxg_dtheta = -sin_theta * x_local - cos_theta * y_local;
                    double dyg_dtheta = cos_theta * x_local - sin_theta * y_local;
                    double dxcenter_dtheta = dxg_dtheta + (a_i + l_r / 2) * (-sin_theta_i) * 1.0;
                    double dycenter_dtheta = dyg_dtheta + (a_i + l_r / 2) * (cos_theta_i) * 1.0;
                    double dvx_dtheta = dxcenter_dtheta + (-sin_theta_i * x_local_corner - cos_theta_i * y_local_corner) * 1.0;
                    double dvy_dtheta = dycenter_dtheta + (cos_theta_i * x_local_corner - sin_theta_i * y_local_corner) * 1.0;
                    G[g_idx++] = (*A)(j, 0) * dvx_dtheta + (*A)(j, 1) * dvy_dtheta; // dFi/dx2
                    double dxcenter_dtheta_i = (a_i + l_r / 2) * (-sin_theta_i);
                    double dycenter_dtheta_i = (a_i + l_r / 2) * (cos_theta_i);
                    double dvx_dtheta_i = dxcenter_dtheta_i + (-sin_theta_i * x_local_corner - cos_theta_i * y_local_corner);
                    double dvy_dtheta_i = dycenter_dtheta_i + (cos_theta_i * x_local_corner - sin_theta_i * y_local_corner);
                    G[g_idx++] = (*A)(j, 0) * dvx_dtheta_i + (*A)(j, 1) * dvy_dtheta_i; // dFi/dx(3+i)
                }
            }
        }
    }
}

class CollaborativeTransport {
public:
    CollaborativeTransport() : nh_(), A_(nullptr), b_(nullptr), current_neF_(0), current_neG_(0) {
        // Initialize ROS subscriber
        sub_ = nh_.subscribe("/iris_polytope", 10, &CollaborativeTransport::polytopeCallback, this);

        // Initialize SNOPT parameters
        n_ = 6;
        lenA_ = 0;
        nS_ = 0;
        ObjRow_ = 0;
        ObjAdd_ = 0.0;
        Cold_ = 0;

        // Allocate memory
        x_ = new double[n_];
        xlow_ = new double[n_];
        xupp_ = new double[n_];
        xmul_ = new double[n_];
        xstate_ = new int[n_];

        // Khởi tạo các mảng với kích thước ban đầu (0 hoặc mặc định)
        neF_ = 0;
        neG_ = 0;
        lenG_ = 0;
        F_ = nullptr;
        Flow_ = nullptr;
        Fupp_ = nullptr;
        Fmul_ = nullptr;
        Fstate_ = nullptr;
        iGfun_ = nullptr;
        jGvar_ = nullptr;

        iAfun_ = new int[lenA_];
        jAvar_ = new int[lenA_];
        linearA_ = new double[lenA_];

        // Initialize ru with space for A and b pointers
        lenru_ = 13; // 11 original + 2 for pointers
        ru_ = new double[lenru_];
        ru_[0] = 0.15 * cos(0.0); ru_[1] = 0.15 * sin(0.0);
        ru_[2] = 0.15 * cos(2 * M_PI / 3); ru_[3] = 0.15 * sin(2 * M_PI / 3);
        ru_[4] = 0.15 * cos(4 * M_PI / 3); ru_[5] = 0.15 * sin(4 * M_PI / 3);
        ru_[6] = 0.2; ru_[7] = 0.2; ru_[8] = 0.2;
        ru_[9] = 0.3; ru_[10] = 0.3;
        ru_[11] = 0.0; ru_[12] = 0.0;

        // Set bounds for x
        xlow_[0] = -1e20; xupp_[0] = 1e20;
        xlow_[1] = -1e20; xupp_[1] = 1e20;
        xlow_[2] = -1e20; xupp_[2] = 1e20;
        xlow_[3] = -M_PI / 2; xupp_[3] = M_PI / 2;
        xlow_[4] = -M_PI / 2; xupp_[4] = M_PI / 2;
        xlow_[5] = -M_PI / 2; xupp_[5] = M_PI / 2;
        for (int i = 0; i < n_; ++i) xstate_[i] = 0;

        // Set initial guess
        x_[0] = 3.0; x_[1] = 1.0;
        x_[2] = 0.0;
        x_[3] = 0.0; x_[4] = 0.0; x_[5] = 0.0;

        // Initialize SNOPT
        ToyProb_.initialize("", 0);
        ToyProb_.setProbName("CollaborativeTransport");
        ToyProb_.setIntParameter("Derivative option", 1);
        ToyProb_.setIntParameter("Major Iteration limit", 250);
        ToyProb_.setIntParameter("Verify level", 3);
        ToyProb_.setUserR(ru_, lenru_);

        ROS_INFO("Collaborative Transport Node started");
    }

    ~CollaborativeTransport() {
        delete[] x_;
        delete[] xlow_;
        delete[] xupp_;
        delete[] xmul_;
        delete[] xstate_;
        delete[] F_;
        delete[] Flow_;
        delete[] Fupp_;
        delete[] Fmul_;
        delete[] Fstate_;
        delete[] iAfun_;
        delete[] jAvar_;
        delete[] linearA_;
        delete[] iGfun_;
        delete[] jGvar_;
        delete[] ru_;
    }

private:
    void setupSNOPT(int num_constraints) {
        // Tính neF_ và neG_
        neF_ = 1 + 15 * num_constraints; // 1 objective + 15 vertices * num_constraints
        neG_ = 2 + // dF0/dx0, dF0/dx1
               3 * num_constraints * 3 + // 3 robots * num_constraints * (dx0, dx1, dx2)
               3 * 4 * num_constraints * 4; // 3 robots * 4 corners * num_constraints * (dx0, dx1, dx2, dx3+i)
        lenG_ = neG_;

        ROS_INFO("Setting up SNOPT: neF_ = %d, neG_ = %d", neF_, neG_);

        // Cấp phát lại bộ nhớ nếu cần
        if (current_neF_ != neF_) {
            // Chỉ delete nếu con trỏ không null
            if (F_) delete[] F_;
            if (Flow_) delete[] Flow_;
            if (Fupp_) delete[] Fupp_;
            if (Fmul_) delete[] Fmul_;
            if (Fstate_) delete[] Fstate_;

            F_ = new double[neF_];
            Flow_ = new double[neF_];
            Fupp_ = new double[neF_];
            Fmul_ = new double[neF_];
            Fstate_ = new int[neF_];
            current_neF_ = neF_;
        }
        if (current_neG_ != neG_) {
            if (iGfun_) delete[] iGfun_;
            if (jGvar_) delete[] jGvar_;
            iGfun_ = new int[lenG_];
            jGvar_ = new int[lenG_];
            current_neG_ = neG_;
        }

        // Set bounds for F
        Flow_[0] = -1e20; Fupp_[0] = 1e20; // Objective function
        for (int i = 1; i < neF_; ++i) {
            Flow_[i] = -1e20; Fupp_[i] = 0.0; // Constraints Ax <= b
        }
        for (int i = 0; i < neF_; ++i) {
            Fmul_[i] = 0;
            Fstate_[i] = 0;
        }

        // Setup Jacobian
        int g_idx = 0;
        iGfun_[g_idx] = 0; jGvar_[g_idx] = 0; g_idx++; // dF0/dx0
        iGfun_[g_idx] = 0; jGvar_[g_idx] = 1; g_idx++; // dF0/dx1
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < num_constraints; ++j) {
                int F_idx = 1 + i * num_constraints + j;
                iGfun_[g_idx] = F_idx; jGvar_[g_idx] = 0; g_idx++; // dFi/dx0
                iGfun_[g_idx] = F_idx; jGvar_[g_idx] = 1; g_idx++; // dFi/dx1
                iGfun_[g_idx] = F_idx; jGvar_[g_idx] = 2; g_idx++; // dFi/dx2
            }
        }
        for (int i = 0; i < 3; ++i) {
            for (int k = 0; k < 4; ++k) {
                for (int j = 0; j < num_constraints; ++j) {
                    int F_idx = 1 + (3 + i * 4 + k) * num_constraints + j;
                    iGfun_[g_idx] = F_idx; jGvar_[g_idx] = 0; g_idx++; // dFi/dx0
                    iGfun_[g_idx] = F_idx; jGvar_[g_idx] = 1; g_idx++; // dFi/dx1
                    iGfun_[g_idx] = F_idx; jGvar_[g_idx] = 2; g_idx++; // dFi/dx2
                    iGfun_[g_idx] = F_idx; jGvar_[g_idx] = 3 + i; g_idx++; // dFi/dx(3+i)
                }
            }
        }
    }

    void polytopeCallback(const std_msgs::Float64MultiArray::ConstPtr& msg) {
        if (msg->layout.dim.size() == 2) {
            int rows = msg->layout.dim[0].size;
            int cols = msg->layout.dim[1].size;
            if (cols != 2) {
                ROS_WARN("Received A with invalid columns: %d, expected 2", cols);
                return;
            }
            if (rows <= 0) {
                ROS_WARN("Received A with invalid rows: %d", rows);
                return;
            }
            A_ = std::make_shared<Eigen::MatrixXd>(rows, cols);
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    (*A_)(i, j) = msg->data[i * cols + j];
                }
            }
        } else if (msg->layout.dim.size() == 1) {
            int rows = msg->layout.dim[0].size;
            if (rows <= 0) {
                ROS_WARN("Received b with invalid size: %d", rows);
                return;
            }
            b_ = std::make_shared<Eigen::VectorXd>(rows);
            for (int i = 0; i < rows; ++i) {
                (*b_)(i) = msg->data[i];
            }
        }

        if (A_ && b_) {
            if (A_->rows() != b_->size()) {
                ROS_WARN("A and b dimensions mismatch: A rows = %ld, b size = %ld", A_->rows(), b_->size());
                A_ = nullptr;
                b_ = nullptr;
                return;
            }
            ROS_INFO("Received A (%ldx2) and b (%ld), starting optimization", A_->rows(), b_->size());
            optimize();
        }
    }

    void optimize() {
        if (!A_ || !b_ || A_->rows() != b_->size() || A_->cols() != 2) {
            ROS_WARN("Cannot optimize: Invalid A or b");
            return;
        }

        int num_constraints = A_->rows();
        ROS_INFO("Optimizing with %d constraints", num_constraints);
        setupSNOPT(num_constraints);

        // Update ru_ with pointers to A_ and b_
        ru_[11] = static_cast<double>(reinterpret_cast<std::uintptr_t>(A_.get()));
        ru_[12] = static_cast<double>(reinterpret_cast<std::uintptr_t>(b_.get()));
        ROS_INFO("ru_[11] = %f, ru_[12] = %f", ru_[11], ru_[12]);

        // Reset initial guess
        x_[0] = 2.5; x_[1] = 1.5;
        x_[2] = 0.0;
        x_[3] = 0.0; x_[4] = 0.0; x_[5] = 0.0;

        // Check initial feasibility
        vector<double> vertices(30);
        computeFormationVertices(x_, &ru_[0], &ru_[9], vertices);

        // Run SNOPT
        ROS_INFO("Calling SNOPT solve");
        int nInf;
        double sInf;
        int status = ToyProb_.solve(Cold_, neF_, n_, ObjAdd_, ObjRow_, objectiveFunction,
                                    iAfun_, jAvar_, linearA_, lenA_,
                                    iGfun_, jGvar_, neG_,
                                    xlow_, xupp_, Flow_, Fupp_,
                                    x_, xstate_, xmul_,
                                    F_, Fstate_, Fmul_,
                                    nS_, nInf, sInf);
 
        // Output results
        ROS_INFO("Optimized configuration:");
        ROS_INFO("t_x = %f, t_y = %f", x_[0], x_[1]);
        ROS_INFO("theta = %f", x_[2]);
        ROS_INFO("theta_1 = %f, theta_2 = %f, theta_3 = %f", x_[3], x_[4], x_[5]);
        if (status == 1) {
            ROS_INFO("Optimization successful");
        } else {
            ROS_INFO("Optimization failed: status = %d", status);
        }
    }

    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    std::shared_ptr<Eigen::MatrixXd> A_;
    std::shared_ptr<Eigen::VectorXd> b_;
    snoptProblemA ToyProb_;
    int n_, neF_, neG_, lenG_, lenA_, nS_;
    int current_neF_, current_neG_; // Lưu kích thước hiện tại để tái cấp phát
    double ObjAdd_;
    int ObjRow_, Cold_;
    double *x_, *xlow_, *xupp_, *xmul_;
    int *xstate_;
    double *F_, *Flow_, *Fupp_, *Fmul_;
    int *Fstate_;
    int *iAfun_, *jAvar_;
    double *linearA_;
    int *iGfun_, *jGvar_;
    double *ru_;
    int lenru_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "sntest_node");
    try {
        CollaborativeTransport transport;
        ros::spin();
    } catch (const std::exception& e) {
        ROS_ERROR("Exception occurred: %s", e.what());
        return 1;
    }
    return 0;
}