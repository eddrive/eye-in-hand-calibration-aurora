#include "eye_in_hand_calibration/chessboard_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <filesystem>
#include <numeric>
#include <cmath>
namespace eye_in_hand_calibration {
ChessboardDetector::ChessboardDetector(const cv::Size& pattern_size,
                                       double square_size,
                                       rclcpp::Logger logger)
    : pattern_size_(pattern_size),
      square_size_(square_size),
      max_variance_threshold_(0.20),
      using_measured_points_(false),
      logger_(logger)
{
    flags_ = cv::CALIB_CB_ADAPTIVE_THRESH | 
             cv::CALIB_CB_NORMALIZE_IMAGE | 
             cv::CALIB_CB_FAST_CHECK;
    subpix_window_ = cv::Size(11, 11);
    subpix_zero_zone_ = cv::Size(-1, -1);
    subpix_criteria_ = cv::TermCriteria(
        cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 
        30, 0.1);
    initializeIdealObjectPoints();
    RCLCPP_INFO(logger_, "ChessboardDetector initialized with IDEAL grid: %dx%d, %.3fm squares",
                pattern_size_.width, pattern_size_.height, square_size_);
}
ChessboardDetector::ChessboardDetector(const cv::Size& pattern_size,
                                       const std::string& measured_points_file,
                                       rclcpp::Logger logger)
    : pattern_size_(pattern_size),
      square_size_(0.0),  // Not used with measured points
      max_variance_threshold_(0.20),
      using_measured_points_(true),
      logger_(logger)
{
    flags_ = cv::CALIB_CB_ADAPTIVE_THRESH | 
             cv::CALIB_CB_NORMALIZE_IMAGE | 
             cv::CALIB_CB_FAST_CHECK;
    subpix_window_ = cv::Size(11, 11);
    subpix_zero_zone_ = cv::Size(-1, -1);
    subpix_criteria_ = cv::TermCriteria(
        cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 
        30, 0.1);
    if (!loadMeasuredObjectPoints(measured_points_file)) {
        RCLCPP_ERROR(logger_, "Failed to load measured points, falling back to ideal grid");
        using_measured_points_ = false;
        square_size_ = 0.005;  // Default 5mm
        initializeIdealObjectPoints();
    } else {
        RCLCPP_INFO(logger_, "ChessboardDetector initialized with MEASURED points from: %s",
                    measured_points_file.c_str());
    }
}
void ChessboardDetector::initializeIdealObjectPoints() {
    object_points_.clear();
    object_points_.reserve(pattern_size_.width * pattern_size_.height);
    for (int i = 0; i < pattern_size_.height; ++i) {
        for (int j = 0; j < pattern_size_.width; ++j) {
            object_points_.emplace_back(
                j * square_size_,
                i * square_size_,
                0.0f  // Ideal planar pattern (Z = 0)
            );
        }
    }
    RCLCPP_INFO(logger_, "Generated %zu ideal object points (planar grid)",
                object_points_.size());
}
bool ChessboardDetector::loadMeasuredObjectPoints(const std::string& filepath) {
    try {
        if (!std::filesystem::exists(filepath)) {
            RCLCPP_ERROR(logger_, "Measured points file not found: %s", filepath.c_str());
            return false;
        }
        YAML::Node yaml = YAML::LoadFile(filepath);
        if (!yaml["chessboard_corners"]) {
            RCLCPP_ERROR(logger_, "Missing 'chessboard_corners' in measured points file");
            return false;
        }
        auto corners_node = yaml["chessboard_corners"];
        int file_rows = corners_node["rows"].as<int>();
        int file_cols = corners_node["cols"].as<int>();
        if (file_rows != pattern_size_.height || file_cols != pattern_size_.width) {
            RCLCPP_ERROR(logger_, 
                        "Pattern size mismatch: file has %dx%d, expected %dx%d",
                        file_cols, file_rows, pattern_size_.width, pattern_size_.height);
            return false;
        }
        auto points_node = corners_node["points"];
        if (!points_node.IsSequence()) {
            RCLCPP_ERROR(logger_, "Invalid 'points' format in measured points file");
            return false;
        }
        object_points_.clear();
        object_points_.reserve(pattern_size_.width * pattern_size_.height);
        for (const auto& point : points_node) {
            float x = point["x"].as<float>();
            float y = point["y"].as<float>();
            float z = point["z"].as<float>();
            object_points_.emplace_back(x, y, z);
        }
        size_t expected_count = pattern_size_.width * pattern_size_.height;
        if (object_points_.size() != expected_count) {
            RCLCPP_ERROR(logger_, 
                        "Point count mismatch: loaded %zu, expected %zu",
                        object_points_.size(), expected_count);
            object_points_.clear();
            return false;
        }
        float min_x = object_points_[0].x;
        float max_x = object_points_[0].x;
        for (const auto& pt : object_points_) {
            min_x = std::min(min_x, pt.x);
            max_x = std::max(max_x, pt.x);
        }
        RCLCPP_INFO(logger_, "Loaded %zu measured object points", object_points_.size());
        RCLCPP_INFO(logger_, "X range: [%.6f, %.6f] m (planarity deviation: %.6f m)",
                    min_x, max_x, max_x - min_x);
        return true;
    } catch (const YAML::Exception& e) {
        RCLCPP_ERROR(logger_, "YAML parsing error: %s", e.what());
        return false;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger_, "Error loading measured points: %s", e.what());
        return false;
    }
}
bool ChessboardDetector::detectPattern(const cv::Mat& image, 
                                       std::vector<cv::Point2f>& corners) {
    if (image.empty()) {
        RCLCPP_WARN(logger_, "Empty image provided to detector");
        return false;
    }
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 1) {
        gray = image.clone();
    } else {
        RCLCPP_ERROR(logger_, "Unsupported image format: %d channels", image.channels());
        return false;
    }
    corners.clear();
    bool found = cv::findChessboardCorners(gray, pattern_size_, corners, flags_);
    if (!found) {
        return false;
    }
    size_t expected_corners = pattern_size_.width * pattern_size_.height;
    if (corners.size() != expected_corners) {
        RCLCPP_WARN(logger_, "Corner count mismatch: detected %zu, expected %zu",
                    corners.size(), expected_corners);
        return false;
    }
    if (!validateCornerQuality(corners)) {
        RCLCPP_DEBUG(logger_, "Corner quality validation failed");
        return false;
    }
    if (!refineCorners(gray, corners)) {
        RCLCPP_WARN(logger_, "Corner refinement failed");
        return false;
    }
    return true;
}
bool ChessboardDetector::refineCorners(const cv::Mat& gray, 
                                       std::vector<cv::Point2f>& corners) {
    try {
        cv::cornerSubPix(gray, corners, 
                        subpix_window_, 
                        subpix_zero_zone_, 
                        subpix_criteria_);
        return true;
    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(logger_, "cornerSubPix failed: %s", e.what());
        return false;
    }
}
bool ChessboardDetector::validateCornerQuality(
    const std::vector<cv::Point2f>& corners) const {
    if (corners.size() != static_cast<size_t>(pattern_size_.width * pattern_size_.height)) {
        return false;
    }
    double variance = calculateDistanceVariance(corners);
    if (variance > max_variance_threshold_) {
        RCLCPP_DEBUG(logger_, "Corner distance variance too high: %.3f (threshold: %.3f)",
                    variance, max_variance_threshold_);
        return false;
    }
    return true;
}
double ChessboardDetector::calculateDistanceVariance(
    const std::vector<cv::Point2f>& corners) const {
    std::vector<double> distances;
    distances.reserve(pattern_size_.width * pattern_size_.height);
    for (int row = 0; row < pattern_size_.height; ++row) {
        for (int col = 0; col < pattern_size_.width - 1; ++col) {
            int idx = row * pattern_size_.width + col;
            cv::Point2f diff = corners[idx + 1] - corners[idx];
            distances.push_back(cv::norm(diff));
        }
    }
    for (int row = 0; row < pattern_size_.height - 1; ++row) {
        for (int col = 0; col < pattern_size_.width; ++col) {
            int idx = row * pattern_size_.width + col;
            int next_idx = (row + 1) * pattern_size_.width + col;
            cv::Point2f diff = corners[next_idx] - corners[idx];
            distances.push_back(cv::norm(diff));
        }
    }
    if (distances.empty()) {
        return 1.0;  // Maximum variance
    }
    double mean = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
    if (mean < 1e-6) {
        return 1.0;  // Avoid division by zero
    }
    double variance_sum = 0.0;
    for (double dist : distances) {
        double diff = dist - mean;
        variance_sum += diff * diff;
    }
    double std_dev = std::sqrt(variance_sum / distances.size());
    return std_dev / mean;
}
double ChessboardDetector::calculateCornerQuality(
    const std::vector<cv::Point2f>& corners) const {
    if (corners.size() != static_cast<size_t>(pattern_size_.width * pattern_size_.height)) {
        return 0.0;
    }
    double variance = calculateDistanceVariance(corners);
    double quality = std::max(0.0, 1.0 - variance / max_variance_threshold_);
    return quality;
}
void ChessboardDetector::drawCorners(cv::Mat& image, 
                                     const std::vector<cv::Point2f>& corners) const {
    if (image.empty() || corners.empty()) {
        return;
    }
    cv::drawChessboardCorners(image, pattern_size_, corners, true);
    if (corners.size() == static_cast<size_t>(pattern_size_.width * pattern_size_.height)) {
        double quality = calculateCornerQuality(corners);
        std::string text = cv::format("Corners: %zu, Quality: %.2f", corners.size(), quality);
        cv::putText(image, text, 
                   cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX,
                   0.7,
                   cv::Scalar(0, 255, 0),
                   2);
    }
}
} // namespace eye_in_hand_calibration