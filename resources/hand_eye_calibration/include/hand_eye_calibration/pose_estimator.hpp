#ifndef HAND_EYE_CALIBRATION_POSE_ESTIMATOR_HPP
#define HAND_EYE_CALIBRATION_POSE_ESTIMATOR_HPP

#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <vector>
#include <rclcpp/rclcpp.hpp>

namespace hand_eye_calibration {

/**
 * @brief Estimates camera pose from 2D-3D correspondences
 * 
 * This class handles pose estimation using PnP algorithms optimized
 * for fisheye and standard camera models. It includes reprojection
 * error calculation and pose validation.
 */
class PoseEstimator {
public:
    /**
     * @brief Constructor
     * @param camera_matrix Camera intrinsic matrix (3x3)
     * @param dist_coeffs Distortion coefficients
     * @param is_fisheye True if using fisheye camera model
     * @param logger ROS2 logger for diagnostics
     */
    PoseEstimator(const cv::Mat& camera_matrix,
                  const cv::Mat& dist_coeffs,
                  bool is_fisheye,
                  rclcpp::Logger logger);
    
    /**
     * @brief Estimate camera pose from 2D-3D point correspondences
     * @param image_points 2D points in image (detected corners)
     * @param object_points 3D points in world frame (pattern coordinates)
     * @param rvec Output rotation vector (Rodrigues representation)
     * @param tvec Output translation vector
     * @return true if pose estimation successful
     */
    bool estimatePose(const std::vector<cv::Point2f>& image_points,
                      const std::vector<cv::Point3f>& object_points,
                      cv::Mat& rvec,
                      cv::Mat& tvec);
    
    /**
     * @brief Compute reprojection error for a given pose
     * @param image_points Original 2D points
     * @param object_points 3D points
     * @param rvec Rotation vector
     * @param tvec Translation vector
     * @return RMS reprojection error in pixels
     */
    double computeReprojectionError(const std::vector<cv::Point2f>& image_points,
                                    const std::vector<cv::Point3f>& object_points,
                                    const cv::Mat& rvec,
                                    const cv::Mat& tvec);
    
    /**
     * @brief Convert pose to 4x4 transformation matrix
     * @param rvec Rotation vector
     * @param tvec Translation vector
     * @return 4x4 homogeneous transformation matrix
     */
    Eigen::Matrix4d poseToMatrix(const cv::Mat& rvec, const cv::Mat& tvec) const;
    
    /**
     * @brief Validate estimated pose
     * @param rvec Rotation vector
     * @param tvec Translation vector
     * @param min_distance Minimum acceptable distance to target (meters)
     * @param max_distance Maximum acceptable distance to target (meters)
     * @return true if pose is valid
     */
    bool validatePose(const cv::Mat& rvec,
                      const cv::Mat& tvec,
                      double min_distance = 0.05,
                      double max_distance = 0.3) const;
    
    /**
     * @brief Get camera matrix
     */
    const cv::Mat& getCameraMatrix() const { return camera_matrix_; }
    
    /**
     * @brief Get distortion coefficients
     */
    const cv::Mat& getDistCoeffs() const { return dist_coeffs_; }
    
    /**
     * @brief Check if using fisheye model
     */
    bool isFisheye() const { return is_fisheye_; }
    
private:
    /**
     * @brief Estimate pose using fisheye camera model
     */
    bool estimatePoseFisheye(const std::vector<cv::Point2f>& image_points,
                             const std::vector<cv::Point3f>& object_points,
                             cv::Mat& rvec,
                             cv::Mat& tvec);
    
    /**
     * @brief Estimate pose using standard pinhole model
     */
    bool estimatePoseStandard(const std::vector<cv::Point2f>& image_points,
                              const std::vector<cv::Point3f>& object_points,
                              cv::Mat& rvec,
                              cv::Mat& tvec);
    
    /**
     * @brief Undistort points for fisheye model
     */
    void undistortPointsFisheye(const std::vector<cv::Point2f>& distorted,
                                std::vector<cv::Point2f>& undistorted) const;
    
    /**
     * @brief Project 3D points to 2D for fisheye model
     */
    void projectPointsFisheye(const std::vector<cv::Point3f>& object_points,
                              std::vector<cv::Point2f>& image_points,
                              const cv::Mat& rvec,
                              const cv::Mat& tvec) const;
    
    /**
     * @brief Refine pose using Levenberg-Marquardt
     */
    bool refinePose(const std::vector<cv::Point2f>& image_points,
                    const std::vector<cv::Point3f>& object_points,
                    cv::Mat& rvec,
                    cv::Mat& tvec);
    
    // Camera parameters
    cv::Mat camera_matrix_;      // 3x3 intrinsic matrix
    cv::Mat dist_coeffs_;        // Distortion coefficients
    bool is_fisheye_;            // Fisheye vs standard model
    
    // Algorithm parameters
    int ransac_iterations_;      // RANSAC iterations
    double ransac_threshold_;    // RANSAC reprojection threshold (pixels)
    double ransac_confidence_;   // RANSAC confidence
    
    // Validation thresholds
    double max_reprojection_error_;  // Maximum acceptable error (pixels)
    
    // Logger
    rclcpp::Logger logger_;
};

} // namespace hand_eye_calibration

#endif // HAND_EYE_CALIBRATION_POSE_ESTIMATOR_HPP