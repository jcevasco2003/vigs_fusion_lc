#include "rclcpp/rclcpp.hpp"
#include "f_vigs_slam/GSSlamNode.hpp"

using namespace f_vigs_slam;

// Implementamos la funcion que inicia, mantiene y apaga el nodo

int main(int argc, char** argv){

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<f_vigs_slam::GSSlamNode>());
    rclcpp::shutdown();

    return 0;
}

