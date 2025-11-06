#include <rclcpp/rclcpp.hpp>
#include "eye_in_hand_calibration/hand_eye_calibrator.hpp"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<eye_in_hand_calibration::HandEyeCalibrator>();
    
    RCLCPP_INFO(rclcpp::get_logger("main"), "Starting Hand-Eye Calibrator Node");
    
    try {
        // Usa executor multi-thread per gestire callback in parallelo
        rclcpp::executors::MultiThreadedExecutor executor;
        executor.add_node(node);
        executor.spin();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception caught: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}
