#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <Eigen/Dense>
#include <sstream>

class PolytopeListener {
public:
    PolytopeListener() : nh_(), A_(nullptr), b_(nullptr) {
        // Subscribe to /iris_polytope topic
        sub_ = nh_.subscribe("/iris_polytope", 10, &PolytopeListener::callback, this);
        ROS_INFO("Polytope listener started");
    }

private:
    void callback(const std_msgs::Float64MultiArray::ConstPtr& msg) {
        // Check if message is for A (2D) or b (1D)
        if (msg->layout.dim.size() == 2) {
            // Matrix A
            int rows = msg->layout.dim[0].size;
            int cols = msg->layout.dim[1].size;
            A_ = std::make_shared<Eigen::MatrixXd>(rows, cols);
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    (*A_)(i, j) = msg->data[i * cols + j];
                }
            }
        } else if (msg->layout.dim.size() == 1) {
            // Vector b
            int rows = msg->layout.dim[0].size;
            b_ = std::make_shared<Eigen::VectorXd>(rows);
            for (int i = 0; i < rows; ++i) {
                (*b_)(i) = msg->data[i];
            }
        }

        // Log A and b when both are received
        if (A_ && b_) {
            std::stringstream ss;
            ss << "Region inequalities: A=\n" << *A_ << "\nb=" << b_->transpose();
            ROS_INFO("%s", ss.str().c_str());
            // Optionally reset A and b to wait for next pair
            // A_ = nullptr;
            // b_ = nullptr;
        }
    }

    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    std::shared_ptr<Eigen::MatrixXd> A_;
    std::shared_ptr<Eigen::VectorXd> b_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "polytope_listener");
    try {
        PolytopeListener listener;
        ros::spin();
    } catch (const std::exception& e) {
        ROS_ERROR("Exception occurred: %s", e.what());
        return 1;
    }
    return 0;
}