#ifndef EYE_IN_HAND_CALIBRATION_CALIBRATION_SOLVER_HPP
#define EYE_IN_HAND_CALIBRATION_CALIBRATION_SOLVER_HPP

#include "eye_in_hand_calibration/sample_manager.hpp"
#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <string>
#include <rclcpp/rclcpp.hpp>

namespace eye_in_hand_calibration {

/**
 * @brief Calibration result with quality metrics
 */
struct CalibrationResult {
    Eigen::Matrix4d transformation;     // Hand-eye transformation matrix
    double average_error;               // Average error across all sample pairs
    double max_error;                   // Maximum error in any sample pair
    double min_error;                   // Minimum error in any sample pair
    int method_used;                    // Calibration method (0-4)
    size_t num_samples_used;            // Number of samples used
    bool success;                       // Overall success flag
    
    CalibrationResult()
        : transformation(Eigen::Matrix4d::Identity()),
          average_error(0.0),
          max_error(0.0),
          min_error(0.0),
          method_used(-1),
          num_samples_used(0),
          success(false) {}
};

/**
 * @brief Solves hand-eye calibration problem AX=XB
 * 
 * This class implements various hand-eye calibration algorithms
 * and provides quality evaluation metrics for the computed transformation.
 */
class CalibrationSolver {
public:
    /**
     * @brief Calibration methods available
     */
    enum Method {
        TSAI = 0,         // Tsai-Lenz method
        PARK = 1,         // Park-Martin method
        HORAUD = 2,       // Horaud-Dornaika method
        ANDREFF = 3,      // Andreff method
        DANIILIDIS = 4    // Daniilidis method
    };
    
    /**
     * @brief Constructor
     * @param method Calibration method to use (default: PARK)
     * @param logger ROS2 logger for diagnostics
     */
    CalibrationSolver(Method method, rclcpp::Logger logger);
    
    /**
     * @brief Perform hand-eye calibration
     * @param samples All calibration samples
     * @param selected_indices Indices of samples to use
     * @param verbose Enable detailed logging (default: true)
     * @return Calibration result with transformation and metrics
     */
    CalibrationResult solve(const std::vector<CalibrationSample>& samples,
                           const std::vector<size_t>& selected_indices,
                           bool verbose = true);
    
    /**
     * @brief Evaluate calibration quality
     * @param samples All samples
     * @param selected_indices Indices of samples used
     * @param transformation Computed hand-eye transformation
     * @return Average reprojection error
     */
    double evaluateCalibration(const std::vector<CalibrationSample>& samples,
                               const std::vector<size_t>& selected_indices,
                               const Eigen::Matrix4d& transformation);
    
    /**
     * @brief Save calibration result to YAML file
     * @param filename Output file path
     * @param result Calibration result to save
     * @param result_frame_id Frame ID for result
     * @param target_frame_id Frame ID for target
     * @param total_samples Total number of samples collected
     * @param successful_detections Number of successful detections
     * @return true if successful
     */
    bool saveResult(const std::string& filename,
                   const CalibrationResult& result,
                   const std::string& result_frame_id,
                   const std::string& target_frame_id,
                   int total_samples,
                   int successful_detections);
    
    /**
     * @brief Set calibration method
     */
    void setMethod(Method method) { method_ = method; }
    
    /**
     * @brief Get calibration method
     */
    Method getMethod() const { return method_; }
    
    /**
     * @brief Get method name as string
     */
    static std::string getMethodName(Method method);

    /**
     * @brief Iterative refinement: removes worst samples based on AX≈XB error
     * @param samples All calibration samples
     * @param indices Current sample indices
     * @param target_pairs Target number of pairs (samples - 1)
     * @param max_iterations Maximum refinement iterations
     * @return Refined list of sample indices
     */
    std::vector<size_t> refineByError(const std::vector<CalibrationSample>& samples,
                                       std::vector<size_t> indices,
                                       int target_pairs,
                                       int max_iterations);

    /**
     * @brief Compute and print absolute errors using direct prediction comparison
     *
     * Matches Python script's compute_errors_alternative() method:
     * - For each sample: T_predicted = T_sensor @ X, compare with T_measured
     * - Rotation error: angle between R_pred and R_meas (degrees)
     * - Translation error: ||t_meas - t_pred|| (mm)
     * - Prints statistics: min, median, mean, std, rms, max, IQR, quartiles
     *
     * @param samples All calibration samples
     * @param selected_indices Indices of samples used
     * @param transformation Computed hand-eye transformation X
     */
    void computeAndPrintAbsoluteErrors(const std::vector<CalibrationSample>& samples,
                                        const std::vector<size_t>& selected_indices,
                                        const Eigen::Matrix4d& transformation);

private:
    /**
     * @brief Convert OpenCV method enum
     */
    cv::HandEyeCalibrationMethod toCvMethod(Method method) const;
    
    /**
     * @brief Prepare data for OpenCV calibration
     */
    void prepareCalibrationData(
        const std::vector<CalibrationSample>& samples,
        const std::vector<size_t>& selected_indices,
        std::vector<cv::Mat>& R_gripper2base,
        std::vector<cv::Mat>& t_gripper2base,
        std::vector<cv::Mat>& R_target2cam,
        std::vector<cv::Mat>& t_target2cam
    );
    
    /**
     * @brief Convert Eigen matrix to OpenCV rotation and translation
     */
    void eigenToCv(const Eigen::Matrix4d& transform,
                  cv::Mat& R,
                  cv::Mat& t) const;
    
    /**
     * @brief Convert OpenCV rotation and translation to Eigen matrix
     */
    Eigen::Matrix4d cvToEigen(const cv::Mat& R, const cv::Mat& t) const;
    
    /**
     * @brief Calculate error for a single sample pair
     */
    double calculatePairError(const Eigen::Matrix4d& A,
                             const Eigen::Matrix4d& B,
                             const Eigen::Matrix4d& X) const;
    
    /**
     * @brief Print transformation details
     */
    void printTransformation(const Eigen::Matrix4d& transform) const;
    
    /**
     * @brief Validate calibration result
     */
    bool validateResult(const CalibrationResult& result) const;
    
    // Configuration
    Method method_;
    
    // Logger
    rclcpp::Logger logger_;
};

} // namespace eye_in_hand_calibration

#endif // EYE_IN_HAND_CALIBRATION_CALIBRATION_SOLVER_HPP