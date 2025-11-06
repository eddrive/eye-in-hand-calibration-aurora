#include "eye_in_hand_calibration/pose_estimator.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <numeric>

namespace eye_in_hand_calibration {

PoseEstimator::PoseEstimator(const cv::Mat& camera_matrix,
                             const cv::Mat& dist_coeffs,
                             bool is_fisheye,
                             rclcpp::Logger logger)
    : camera_matrix_(camera_matrix.clone()),
      dist_coeffs_(dist_coeffs.clone()),
      is_fisheye_(is_fisheye),
      ransac_iterations_(2000),
      ransac_threshold_(2.0),      // pixels
      ransac_confidence_(0.999),
      max_reprojection_error_(3.0), // pixels
      logger_(logger)
{
    RCLCPP_INFO(logger_, "PoseEstimator initialized: %s model",
                is_fisheye_ ? "fisheye" : "pinhole");
}

bool PoseEstimator::estimatePose(const std::vector<cv::Point2f>& image_points,
                                 const std::vector<cv::Point3f>& object_points,
                                 cv::Mat& rvec,
                                 cv::Mat& tvec) {
    // Validate input
    if (image_points.size() != object_points.size() || image_points.size() < 4) {
        RCLCPP_ERROR(logger_,
                     "Invalid point correspondences: image=%zu, object=%zu (need >= 4)",
                     image_points.size(), object_points.size());
        return false;
    }
    
    bool success = false;
    
    if (is_fisheye_) {
        success = estimatePoseFisheye(image_points, object_points, rvec, tvec);
    } else {
        success = estimatePoseStandard(image_points, object_points, rvec, tvec);
    }
    
    if (!success) {
        return false;
    }
    
    // Validate pose
    if (!validatePose(rvec, tvec)) {
        RCLCPP_WARN(logger_, "Pose validation failed");
        return false;
    }
    
    // Compute and log reprojection error
    double error = computeReprojectionError(image_points, object_points, rvec, tvec);
    
    RCLCPP_DEBUG(logger_,
                "Pose estimated: rvec=[%.3f, %.3f, %.3f], tvec=[%.3f, %.3f, %.3f], error=%.3fpx",
                rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2),
                tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2),
                error);
    
    return true;
}

bool PoseEstimator::estimatePoseFisheye(const std::vector<cv::Point2f>& image_points,
                                        const std::vector<cv::Point3f>& object_points,
                                        cv::Mat& rvec,
                                        cv::Mat& tvec) {
    try {
        // Step 1: Undistort points to normalized image coordinates
        std::vector<cv::Point2f> undistorted_points;
        undistortPointsFisheye(image_points, undistorted_points);
        
        // Step 2: Convert to double precision for better accuracy
        std::vector<cv::Point2d> image_pts_d;
        image_pts_d.reserve(undistorted_points.size());
        for (const auto& pt : undistorted_points) {
            image_pts_d.emplace_back(pt.x, pt.y);
        }
        
        std::vector<cv::Point3d> object_pts_d;
        object_pts_d.reserve(object_points.size());
        for (const auto& pt : object_points) {
            object_pts_d.emplace_back(pt.x, pt.y, pt.z);
        }
        
        // Step 3: Zero distortion for undistorted points
        cv::Mat zero_dist = cv::Mat::zeros(4, 1, CV_64F);
        
        // Step 4: Try IPPE first (best for planar patterns)
        bool success = cv::solvePnP(
            object_pts_d, image_pts_d,
            camera_matrix_, zero_dist,
            rvec, tvec,
            false,                    // useExtrinsicGuess
            cv::SOLVEPNP_IPPE         // Infinitesimal Plane-based Pose Estimation
        );
        
        // Step 5: Fallback to RANSAC + EPNP if IPPE fails
        if (!success) {
            RCLCPP_DEBUG(logger_, "IPPE failed, trying RANSAC+EPNP");
            
            cv::Mat inliers;
            success = cv::solvePnPRansac(
                object_pts_d, image_pts_d,
                camera_matrix_, zero_dist,
                rvec, tvec,
                false,                     // useExtrinsicGuess
                ransac_iterations_,
                ransac_threshold_,
                ransac_confidence_,
                inliers,
                cv::SOLVEPNP_EPNP
            );
            
            // Refine with inliers if we have enough
            if (success && inliers.rows >= 4) {
                std::vector<cv::Point3d> inlier_obj;
                std::vector<cv::Point2d> inlier_img;
                
                for (int i = 0; i < inliers.rows; ++i) {
                    int idx = inliers.at<int>(i, 0);
                    inlier_obj.push_back(object_pts_d[idx]);
                    inlier_img.push_back(image_pts_d[idx]);
                }
                
                cv::solvePnPRefineLM(
                    inlier_obj, inlier_img,
                    camera_matrix_, zero_dist,
                    rvec, tvec
                );
                
                RCLCPP_DEBUG(logger_, "RANSAC inliers: %d/%zu", 
                            inliers.rows, image_points.size());
            }
        } else {
            // Step 6: Always refine IPPE solution with LM
            cv::solvePnPRefineLM(
                object_pts_d, image_pts_d,
                camera_matrix_, zero_dist,
                rvec, tvec
            );
        }
        
        if (!success) {
            RCLCPP_WARN(logger_, "All PnP methods failed for fisheye");
            return false;
        }

        return true;
        
    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(logger_, "Fisheye pose estimation failed: %s", e.what());
        return false;
    }
}

bool PoseEstimator::estimatePoseStandard(const std::vector<cv::Point2f>& image_points,
                                         const std::vector<cv::Point3f>& object_points,
                                         cv::Mat& rvec,
                                         cv::Mat& tvec) {
    try {
        // Try IPPE first
        bool success = cv::solvePnP(
            object_points, image_points,
            camera_matrix_, dist_coeffs_,
            rvec, tvec,
            false,
            cv::SOLVEPNP_IPPE
        );
        
        // Fallback to RANSAC
        if (!success) {
            RCLCPP_DEBUG(logger_, "IPPE failed, trying RANSAC");
            
            cv::Mat inliers;
            success = cv::solvePnPRansac(
                object_points, image_points,
                camera_matrix_, dist_coeffs_,
                rvec, tvec,
                false,
                ransac_iterations_,
                ransac_threshold_,
                ransac_confidence_,
                inliers
            );
            
            if (success && inliers.rows >= 4) {
                // Refine with inliers
                std::vector<cv::Point3f> inlier_obj;
                std::vector<cv::Point2f> inlier_img;
                
                for (int i = 0; i < inliers.rows; ++i) {
                    int idx = inliers.at<int>(i, 0);
                    inlier_obj.push_back(object_points[idx]);
                    inlier_img.push_back(image_points[idx]);
                }
                
                cv::solvePnPRefineLM(
                    inlier_obj, inlier_img,
                    camera_matrix_, dist_coeffs_,
                    rvec, tvec
                );
            }
        } else {
            // Refine IPPE solution
            cv::solvePnPRefineLM(
                object_points, image_points,
                camera_matrix_, dist_coeffs_,
                rvec, tvec
            );
        }

        if (!success) {
            return false;
        }

        return true;

    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(logger_, "Standard pose estimation failed: %s", e.what());
        return false;
    }
}

void PoseEstimator::undistortPointsFisheye(const std::vector<cv::Point2f>& distorted,
                                           std::vector<cv::Point2f>& undistorted) const {
    cv::fisheye::undistortPoints(
        distorted,
        undistorted,
        camera_matrix_,      // K
        dist_coeffs_,        // D
        cv::noArray(),       // R (no rectification)
        camera_matrix_       // P = K (output in pixel coordinates)
    );
}

void PoseEstimator::projectPointsFisheye(const std::vector<cv::Point3f>& object_points,
                                         std::vector<cv::Point2f>& image_points,
                                         const cv::Mat& rvec,
                                         const cv::Mat& tvec) const {
    cv::fisheye::projectPoints(
        object_points,
        image_points,
        rvec, tvec,
        camera_matrix_,
        dist_coeffs_
    );
}

double PoseEstimator::computeReprojectionError(const std::vector<cv::Point2f>& image_points,
                                               const std::vector<cv::Point3f>& object_points,
                                               const cv::Mat& rvec,
                                               const cv::Mat& tvec) {
    if (image_points.size() != object_points.size() || image_points.empty()) {
        return std::numeric_limits<double>::max();
    }
    
    std::vector<cv::Point2f> projected_points;
    
    if (is_fisheye_) {
        projectPointsFisheye(object_points, projected_points, rvec, tvec);
    } else {
        cv::projectPoints(object_points, rvec, tvec, 
                         camera_matrix_, dist_coeffs_, 
                         projected_points);
    }
    
    // Calculate RMS error
    double sum_squared_error = 0.0;
    for (size_t i = 0; i < image_points.size(); ++i) {
        cv::Point2f diff = image_points[i] - projected_points[i];
        sum_squared_error += diff.x * diff.x + diff.y * diff.y;
    }
    
    double rms_error = std::sqrt(sum_squared_error / image_points.size());
    return rms_error;
}

Eigen::Matrix4d PoseEstimator::poseToMatrix(const cv::Mat& rvec, 
                                            const cv::Mat& tvec) const {
    // Convert rotation vector to rotation matrix
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    
    // Build 4x4 transformation matrix
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    
    // Copy rotation
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            transform(i, j) = R_cv.at<double>(i, j);
        }
    }
    
    // Copy translation
    transform(0, 3) = tvec.at<double>(0);
    transform(1, 3) = tvec.at<double>(1);
    transform(2, 3) = tvec.at<double>(2);
    
    return transform;
}

bool PoseEstimator::validatePose(const cv::Mat& rvec,
                                 const cv::Mat& tvec,
                                 double min_distance,
                                 double max_distance) const {
    // Check vector sizes
    if (rvec.rows != 3 || rvec.cols != 1 || 
        tvec.rows != 3 || tvec.cols != 1) {
        RCLCPP_ERROR(logger_, "Invalid pose vector dimensions");
        return false;
    }
    
    // Check for NaN or Inf
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(rvec.at<double>(i)) || 
            !std::isfinite(tvec.at<double>(i))) {
            RCLCPP_ERROR(logger_, "Pose contains NaN or Inf values");
            return false;
        }
    }
    
    // Check distance to target
    double tx = tvec.at<double>(0);
    double ty = tvec.at<double>(1);
    double tz = tvec.at<double>(2);
    double distance = std::sqrt(tx*tx + ty*ty + tz*tz);
    
    if (distance < min_distance || distance > max_distance) {
        RCLCPP_WARN(logger_,
                   "Distance out of range: %.3fm (valid: %.3f-%.3fm)",
                   distance, min_distance, max_distance);
        return false;
    }
    
    // Check rotation magnitude (should be reasonable for calibration)
    double rotation_angle = cv::norm(rvec);
    if (rotation_angle > M_PI) {
        RCLCPP_WARN(logger_, 
                   "Large rotation angle: %.2f rad (%.1f deg)",
                   rotation_angle, rotation_angle * 180.0 / M_PI);
    }
    
    return true;
}

} // namespace eye_in_hand_calibration