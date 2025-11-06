#ifndef EYE_IN_HAND_CALIBRATION_CHESSBOARD_DETECTOR_HPP
#define EYE_IN_HAND_CALIBRATION_CHESSBOARD_DETECTOR_HPP

#include <opencv2/core.hpp>
#include <vector>
#include <rclcpp/rclcpp.hpp>

namespace eye_in_hand_calibration {

/**
 * @brief Detector for chessboard calibration patterns
 * 
 * This class handles detection of chessboard corners in images,
 * validates corner quality, and refines corner positions for
 * accurate pose estimation.
 */
class ChessboardDetector {
public:
    /**
     * @brief Constructor with ideal grid object points
     * @param pattern_size Chessboard size (cols, rows) - internal corners
     * @param square_size Physical size of squares in meters
     * @param logger ROS2 logger for diagnostics
     */
    ChessboardDetector(const cv::Size& pattern_size,
                       double square_size,
                       rclcpp::Logger logger);

    /**
     * @brief Constructor with measured object points from file
     * @param pattern_size Chessboard size (cols, rows) - internal corners
     * @param square_size Physical size of squares in meters
     * @param path to the yaml file that contains chessboard coordinates in aurora frame
     * @param logger ROS2 logger for diagnostics
     */
    ChessboardDetector(const cv::Size& pattern_size,
                       const std::string& measured_points_file,
                       rclcpp::Logger logger);
    
    /**
     * @brief Detect chessboard pattern in image
     * @param image Input image (BGR or grayscale)
     * @param corners Output detected corners (subpixel refined)
     * @return true if pattern detected and validated
     */
    bool detectPattern(const cv::Mat& image, 
                       std::vector<cv::Point2f>& corners);
    
    /**
     * @brief Get 3D object points for the chessboard pattern
     * @return Vector of 3D points in pattern coordinate frame (Z=0)
     */
    const std::vector<cv::Point3f>& getObjectPoints() const {
        return object_points_;
    }
    
    /**
     * @brief Get pattern size
     * @return Chessboard dimensions (cols, rows)
     */
    cv::Size getPatternSize() const {
        return pattern_size_;
    }
    
    /**
     * @brief Get square size in meters
     */
    double getSquareSize() const {
        return square_size_;
    }
    
    /**
     * @brief Calculate corner detection quality metric
     * @param corners Detected corners
     * @return Quality score [0-1], where 1 is perfect
     */
    double calculateCornerQuality(const std::vector<cv::Point2f>& corners) const;
    
    /**
     * @brief Draw detected corners on image for visualization
     * @param image Input/output image
     * @param corners Detected corners
     */
    void drawCorners(cv::Mat& image, 
                     const std::vector<cv::Point2f>& corners) const;
    
private:
    /**
     * @brief Initialize ideal grid object points
     */
    void initializeIdealObjectPoints();

    /**
     * @brief Load measured object points from YAML file
     * @param path to the yaml file that contains chessboard coordinates in aurora frame
     */
    bool loadMeasuredObjectPoints(const std::string& filepath);
    
    /**
     * @brief Refine corner positions to subpixel accuracy
     * @param gray Grayscale image
     * @param corners Input/output corners
     * @return true if refinement successful
     */
    bool refineCorners(const cv::Mat& gray, 
                       std::vector<cv::Point2f>& corners);
    
    /**
     * @brief Validate corner distribution quality
     * @param corners Detected corners
     * @return true if corners meet quality criteria
     */
    bool validateCornerQuality(const std::vector<cv::Point2f>& corners) const;
    
    /**
     * @brief Calculate variance of distances between consecutive corners
     * @param corners Detected corners
     * @return Normalized variance (coefficient of variation)
     */
    double calculateDistanceVariance(const std::vector<cv::Point2f>& corners) const;
    
    // Configuration
    cv::Size pattern_size_;              // Chessboard dimensions (cols, rows)
    double square_size_;                 // Physical square size in meters
    std::vector<cv::Point3f> object_points_;  // 3D pattern points
    
    // Detection parameters
    int flags_;                          // OpenCV findChessboardCorners flags
    double max_variance_threshold_;      // Max acceptable corner variance
    
    // Subpixel refinement parameters
    cv::Size subpix_window_;            // Search window size
    cv::Size subpix_zero_zone_;         // Dead zone
    cv::TermCriteria subpix_criteria_;  // Convergence criteria

    bool using_measured_points_;      // Track which mode we're using
    
    // Logger
    rclcpp::Logger logger_;
};

} // namespace eye_in_hand_calibration

#endif // EYE_IN_HAND_CALIBRATION_CHESSBOARD_DETECTOR_HPP