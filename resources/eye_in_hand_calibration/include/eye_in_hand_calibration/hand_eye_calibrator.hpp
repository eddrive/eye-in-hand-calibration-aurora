#ifndef EYE_IN_HAND_CALIBRATION_HAND_EYE_CALIBRATOR_HPP
#define EYE_IN_HAND_CALIBRATION_HAND_EYE_CALIBRATOR_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <aurora_ndi_ros2_driver/msg/aurora_data.hpp>

#include "eye_in_hand_calibration/calibration_config.hpp"
#include "eye_in_hand_calibration/chessboard_detector.hpp"
#include "eye_in_hand_calibration/pose_estimator.hpp"
#include "eye_in_hand_calibration/aurora_synchronizer.hpp"
#include "eye_in_hand_calibration/sample_manager.hpp"
#include "eye_in_hand_calibration/calibration_solver.hpp"

#include <queue>
#include <thread>
#include <atomic>
#include <condition_variable>

namespace eye_in_hand_calibration {

/**
 * @brief Image processing task for worker threads
 */
struct ImageProcessingTask {
    sensor_msgs::msg::CompressedImage::SharedPtr image_msg;
    rclcpp::Time image_timestamp;
};

/**
 * @brief Main hand-eye calibration node
 * 
 * Orchestrates the entire calibration pipeline:
 * 1. Collects synchronized image + Aurora data
 * 2. Detects chessboard patterns
 * 3. Estimates camera poses
 * 4. Manages sample diversity
 * 5. Solves hand-eye calibration
 */
class HandEyeCalibrator : public rclcpp::Node {
public:
    HandEyeCalibrator();
    ~HandEyeCalibrator();

private:
    // ========== Callbacks ==========
    void imageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
    void auroraCallback(const aurora_ndi_ros2_driver::msg::AuroraData::SharedPtr msg);
    
    // ========== Processing ==========
    void processingThreadFunction();
    void processImage(const ImageProcessingTask& task);
    
    // ========== Calibration Pipeline ==========
    void selectBestSamplesAndCalibrate();
    
    // ========== Components ==========
    CalibrationConfig config_;
    
    std::unique_ptr<ChessboardDetector> detector_;
    std::unique_ptr<PoseEstimator> pose_estimator_;
    std::unique_ptr<AuroraSynchronizer> aurora_sync_;
    std::unique_ptr<SampleManager> sample_manager_;
    std::unique_ptr<CalibrationSolver> solver_;
    
    // ========== ROS2 Communication ==========
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
    rclcpp::Subscription<aurora_ndi_ros2_driver::msg::AuroraData>::SharedPtr aurora_sub_;
    
    // ========== Threading ==========
    std::vector<std::thread> processing_threads_;
    std::queue<ImageProcessingTask> processing_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> should_stop_;
    int num_processing_threads_;
    
    // ========== State ==========
    std::atomic<bool> collection_complete_;
    std::atomic<bool> calibration_started_;
    std::atomic<int> total_images_processed_;
    std::atomic<int> successful_detections_;
    std::atomic<int> samples_saved_;
    std::atomic<int> images_dropped_;

    Eigen::Matrix4d final_transformation_;
};

} // namespace eye_in_hand_calibration

#endif // EYE_IN_HAND_CALIBRATION_HAND_EYE_CALIBRATOR_HPP