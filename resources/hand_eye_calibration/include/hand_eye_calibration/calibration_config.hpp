#ifndef HAND_EYE_CALIBRATION_CONFIG_HPP
#define HAND_EYE_CALIBRATION_CONFIG_HPP

#include <string>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

namespace hand_eye_calibration {

struct CalibrationConfig {
    // ========== Topics ==========
    std::string image_topic;
    std::string aurora_topic;
    
    // ========== Camera Calibration ==========
    std::string camera_calibration_file;
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    bool is_fisheye;
    
    // ========== Chessboard Pattern ==========
    cv::Size chessboard_size;  // (cols, rows) - internal corners
    double square_size;        // meters
    bool use_measured_object_points;      // true = use Aurora measurements, false = ideal grid
    std::string measured_points_file;     // Path to YAML with measured 3D points
    
    // ========== Collection Parameters ==========
    int max_samples;           // Total samples to collect
    int final_poses;           // Best samples to use for calibration
    double max_error_threshold;
    
    // ========== Synchronization ==========
    int max_pose_age_ms;       // Max time diff for Aurora sync
    int aurora_buffer_size;
    
    // ========== Diversity Filtering ==========
    double min_movement_threshold;  // meters
    double min_rotation_threshold;  // radians
    
    // ========== Calibration Method ==========
    int calibration_method;    // 0=TSAI, 1=PARK, 2=HORAUD, 3=ANDREFF, 4=DANIILIDIS
    
    // ========== Output ==========
    bool save_result;
    std::string result_frame_id;
    std::string target_frame_id;
    
    // ========== Logging ==========
    bool verbose;
    
    // ========== Factory Method ==========
    static CalibrationConfig loadFromNode(rclcpp::Node* node) {
        CalibrationConfig config;
        
        // Topics
        config.image_topic = node->declare_parameter<std::string>(
            "image_topic", "/endoscope/image_raw/compressed");
        config.aurora_topic = node->declare_parameter<std::string>(
            "aurora_topic", "/aurora_data_sensor0");
        
        // Camera calibration
        config.camera_calibration_file = node->declare_parameter<std::string>(
            "camera_calibration_file", 
            "/workspace/hand_eye_calibration/config/camera_calibration_fisheye_1080p.yaml");
        
        // Chessboard
        int rows = node->declare_parameter<int>("chessboard_rows", 9);
        int cols = node->declare_parameter<int>("chessboard_cols", 6);
        config.chessboard_size = cv::Size(cols, rows);
        config.square_size = node->declare_parameter<double>("chessboard_square_size", 0.005);

        // Object points configuration
        config.use_measured_object_points = node->declare_parameter<bool>(
            "use_measured_object_points", false);
        config.measured_points_file = node->declare_parameter<std::string>(
            "measured_points_file", 
            "/workspace/hand_eye_calibration/config/chessboard_measured_points.yaml");
        
        
        // Collection
        config.max_samples = node->declare_parameter<int>("max_samples", 100);
        config.final_poses = node->declare_parameter<int>("final_poses", 20);
        config.max_error_threshold = node->declare_parameter<double>("max_error_threshold", 0.01);
        
        // Synchronization
        config.max_pose_age_ms = node->declare_parameter<int>("max_pose_age_ms", 50);
        config.aurora_buffer_size = node->declare_parameter<int>("aurora_buffer_size", 3000);
        
        // Diversity
        config.min_movement_threshold = node->declare_parameter<double>("min_movement_threshold", 0.015);
        config.min_rotation_threshold = node->declare_parameter<double>("min_rotation_threshold", 0.15);
        
        // Calibration
        config.calibration_method = node->declare_parameter<int>("calibration_method", 1);
        
        // Output
        config.save_result = node->declare_parameter<bool>("save_result", true);
        config.result_frame_id = node->declare_parameter<std::string>("result_frame_id", "endo_aurora");
        config.target_frame_id = node->declare_parameter<std::string>("target_frame_id", "endo_optical");
        
        // Logging
        config.verbose = node->declare_parameter<bool>("verbose", true);
        
        // Load camera calibration
        if (!config.loadCameraCalibration(node)) {
            throw std::runtime_error("Failed to load camera calibration");
        }
        
        return config;
    }
    
private:
    bool loadCameraCalibration(rclcpp::Node* node) {
        try {
            YAML::Node yaml = YAML::LoadFile(camera_calibration_file);
            
            auto cam_matrix_data = yaml["camera_matrix"]["data"].as<std::vector<double>>();
            camera_matrix = cv::Mat(3, 3, CV_64F, cam_matrix_data.data()).clone();
            
            auto dist_data = yaml["distortion_coefficients"]["data"].as<std::vector<double>>();
            dist_coeffs = cv::Mat(1, dist_data.size(), CV_64F, dist_data.data()).clone();
            
            std::string distortion_model = yaml["distortion_model"].as<std::string>();
            is_fisheye = (distortion_model == "fisheye" || distortion_model == "equidistant");
            
            RCLCPP_INFO(node->get_logger(), "✓ Camera calibration loaded: %s", 
                        distortion_model.c_str());
            return true;
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(node->get_logger(), "❌ Camera calibration error: %s", e.what());
            return false;
        }
    }
};

} // namespace hand_eye_calibration

#endif // HAND_EYE_CALIBRATION_CONFIG_HPP