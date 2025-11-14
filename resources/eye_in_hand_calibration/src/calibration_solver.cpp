#include "eye_in_hand_calibration/calibration_solver.hpp"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <map>
#include <algorithm>
#include <numeric>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
namespace eye_in_hand_calibration {
CalibrationSolver::CalibrationSolver(Method method, rclcpp::Logger logger)
    : method_(method), logger_(logger)
{
    RCLCPP_INFO(logger_, "CalibrationSolver initialized with method: %s",
                getMethodName(method_).c_str());
}
CalibrationResult CalibrationSolver::solve(
    const std::vector<CalibrationSample>& samples,
    const std::vector<size_t>& selected_indices,
    bool verbose) {
    CalibrationResult result;
    result.method_used = static_cast<int>(method_);
    result.num_samples_used = selected_indices.size();
    if (selected_indices.size() < 3) {
        RCLCPP_ERROR(logger_, "Need at least 3 samples for calibration (got %zu)",
                    selected_indices.size());
        return result;
    }
    try {
        if (verbose) {
            RCLCPP_INFO(logger_, "\n========== PERFORMING CALIBRATION ==========");
            RCLCPP_INFO(logger_, "Method: %s", getMethodName(method_).c_str());
            RCLCPP_INFO(logger_, "Using %zu selected samples", selected_indices.size());
        }
        std::vector<cv::Mat> R_gripper2base, t_gripper2base;
        std::vector<cv::Mat> R_target2cam, t_target2cam;
        prepareCalibrationData(samples, selected_indices,
                              R_gripper2base, t_gripper2base,
                              R_target2cam, t_target2cam);
        cv::Mat R_cam2gripper, t_cam2gripper;
        cv::HandEyeCalibrationMethod cv_method = toCvMethod(method_);
        if (verbose) {
            RCLCPP_INFO(logger_, "Running OpenCV calibrateHandEye...");
        }
        cv::calibrateHandEye(R_gripper2base, t_gripper2base,
                            R_target2cam, t_target2cam,
                            R_cam2gripper, t_cam2gripper,
                            cv_method);
        result.transformation = cvToEigen(R_cam2gripper, t_cam2gripper);
        result.success = validateResult(result);
        if (verbose) {
            RCLCPP_INFO(logger_, "\n========== CALIBRATION RESULT ==========");
            RCLCPP_INFO(logger_, "Success: %s", result.success ? "YES" : "NO");
            RCLCPP_INFO(logger_, "Samples used: %zu", result.num_samples_used);
            printTransformation(result.transformation);
            RCLCPP_INFO(logger_, "========================================\n");
        }
        return result;
    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(logger_, "OpenCV calibration failed: %s", e.what());
        return result;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger_, "Calibration failed: %s", e.what());
        return result;
    }
}
void CalibrationSolver::prepareCalibrationData(
    const std::vector<CalibrationSample>& samples,
    const std::vector<size_t>& selected_indices,
    std::vector<cv::Mat>& R_gripper2base,
    std::vector<cv::Mat>& t_gripper2base,
    std::vector<cv::Mat>& R_target2cam,
    std::vector<cv::Mat>& t_target2cam) {
    R_gripper2base.clear();
    t_gripper2base.clear();
    R_target2cam.clear();
    t_target2cam.clear();
    for (size_t idx : selected_indices) {
        if (idx >= samples.size()) {
            RCLCPP_WARN(logger_, "Invalid sample index: %zu", idx);
            continue;
        }
        const auto& sample = samples[idx];
        cv::Mat R_gb, t_gb;
        eigenToCv(sample.sensor_pose, R_gb, t_gb);
        R_gripper2base.push_back(R_gb);
        t_gripper2base.push_back(t_gb);
        Eigen::Matrix4d T_camera_to_aurora = sample.camera_pose.inverse();
        cv::Mat R_tc, t_tc;
        eigenToCv(T_camera_to_aurora, R_tc, t_tc);
        R_target2cam.push_back(R_tc);
        t_target2cam.push_back(t_tc);
    }
    RCLCPP_DEBUG(logger_, "Prepared %zu pose pairs for calibration",
                R_gripper2base.size());
}
void CalibrationSolver::eigenToCv(const Eigen::Matrix4d& transform,
                                  cv::Mat& R,
                                  cv::Mat& t) const {
    R = cv::Mat(3, 3, CV_64F);
    t = cv::Mat(3, 1, CV_64F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R.at<double>(i, j) = transform(i, j);
        }
        t.at<double>(i, 0) = transform(i, 3);
    }
}
Eigen::Matrix4d CalibrationSolver::cvToEigen(const cv::Mat& R, 
                                             const cv::Mat& t) const {
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            transform(i, j) = R.at<double>(i, j);
        }
        transform(i, 3) = t.at<double>(i, 0);
    }
    return transform;
}
void CalibrationSolver::printTransformation(const Eigen::Matrix4d& transform) const {
    Eigen::Vector3d translation = transform.block<3, 1>(0, 3);
    Eigen::Matrix3d rotation = transform.block<3, 3>(0, 0);
    Eigen::Vector3d euler = rotation.eulerAngles(0, 1, 2) * 180.0 / M_PI;
    Eigen::Quaterniond quat(rotation);
    RCLCPP_INFO(logger_, "Translation (Sensor to Camera):");
    RCLCPP_INFO(logger_, "  [%.6f, %.6f, %.6f] m",
               translation.x(), translation.y(), translation.z());
    RCLCPP_INFO(logger_, "  Sensor-Camera Distance: %.6f m (%.2f mm)",
               translation.norm(), translation.norm() * 1000.0);
    RCLCPP_INFO(logger_, "Rotation (Euler XYZ):");
    RCLCPP_INFO(logger_, "  [%.2f, %.2f, %.2f] degrees",
               euler.x(), euler.y(), euler.z());
    RCLCPP_INFO(logger_, "Rotation (Quaternion):");
    RCLCPP_INFO(logger_, "  [x=%.6f, y=%.6f, z=%.6f, w=%.6f]",
               quat.x(), quat.y(), quat.z(), quat.w());
    RCLCPP_INFO(logger_, "Transformation Matrix:");
    for (int i = 0; i < 4; ++i) {
        RCLCPP_INFO(logger_, "  [%9.6f %9.6f %9.6f %9.6f]",
                   transform(i, 0), transform(i, 1),
                   transform(i, 2), transform(i, 3));
    }
}
bool CalibrationSolver::validateResult(const CalibrationResult& result) const {
    if (!result.success && result.num_samples_used < 3) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (!std::isfinite(result.transformation(i, j))) {
                RCLCPP_ERROR(logger_, "Transformation contains NaN or Inf");
                return false;
            }
        }
    }
    Eigen::Matrix3d R = result.transformation.block<3, 3>(0, 0);
    double det = R.determinant();
    if (std::abs(det - 1.0) > 0.1) {
        RCLCPP_WARN(logger_, "Rotation matrix determinant: %.6f (expected 1.0)", det);
    }
    Eigen::Matrix3d I = R * R.transpose();
    double orthogonality_error = (I - Eigen::Matrix3d::Identity()).norm();
    if (orthogonality_error > 0.1) {
        RCLCPP_WARN(logger_, "Rotation matrix orthogonality error: %.6f", 
                   orthogonality_error);
    }
    double translation_norm = result.transformation.block<3, 1>(0, 3).norm();
    if (translation_norm > 1.0) {  // More than 1 meter seems unusual for hand-eye
        RCLCPP_WARN(logger_, "Large translation: %.3f m", translation_norm);
    }
    return true;
}
bool CalibrationSolver::saveResult(const std::string& filename,
                                   const CalibrationResult& result,
                                   const std::string& result_frame_id,
                                   const std::string& target_frame_id,
                                   int total_samples,
                                   int successful_detections) {
    try {
        YAML::Emitter out;
        out << YAML::BeginMap;
        out << YAML::Key << "eye_in_hand_calibration";
        out << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "timestamp" << YAML::Value << std::time(nullptr);
        out << YAML::Key << "method" << YAML::Value << getMethodName(static_cast<Method>(result.method_used));
        out << YAML::Key << "samples_used" << YAML::Value << result.num_samples_used;
        out << YAML::Key << "result_frame_id" << YAML::Value << result_frame_id;
        out << YAML::Key << "target_frame_id" << YAML::Value << target_frame_id;
        out << YAML::Key << "success" << YAML::Value << result.success;
        out << YAML::Key << "transformation_matrix";
        out << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "rows" << YAML::Value << 4;
        out << YAML::Key << "cols" << YAML::Value << 4;
        out << YAML::Key << "data" << YAML::Value << YAML::Flow;
        out << YAML::BeginSeq;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                out << result.transformation(i, j);
            }
        }
        out << YAML::EndSeq;
        out << YAML::EndMap;
        Eigen::Vector3d translation = result.transformation.block<3, 1>(0, 3);
        out << YAML::Key << "translation";
        out << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "x" << YAML::Value << translation.x();
        out << YAML::Key << "y" << YAML::Value << translation.y();
        out << YAML::Key << "z" << YAML::Value << translation.z();
        out << YAML::EndMap;
        Eigen::Matrix3d rotation = result.transformation.block<3, 3>(0, 0);
        Eigen::Quaterniond quat(rotation);
        out << YAML::Key << "rotation";
        out << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "x" << YAML::Value << quat.x();
        out << YAML::Key << "y" << YAML::Value << quat.y();
        out << YAML::Key << "z" << YAML::Value << quat.z();
        out << YAML::Key << "w" << YAML::Value << quat.w();
        out << YAML::EndMap;
        out << YAML::Key << "quality";
        out << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "total_samples_collected" << YAML::Value << total_samples;
        out << YAML::Key << "successful_detections" << YAML::Value << successful_detections;
        out << YAML::Key << "detection_rate_percent" << YAML::Value <<
                    (total_samples > 0 ? (double)successful_detections / total_samples * 100.0 : 0.0);
        out << YAML::EndMap;
        out << YAML::EndMap;
        out << YAML::EndMap;
        std::ofstream file(filename);
        if (!file.is_open()) {
            RCLCPP_ERROR(logger_, "Failed to open file: %s", filename.c_str());
            return false;
        }
        file << out.c_str();
        file.close();
        RCLCPP_INFO(logger_, "✓ Calibration result saved to: %s", filename.c_str());
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger_, "Failed to save calibration result: %s", e.what());
        return false;
    }
}
cv::HandEyeCalibrationMethod CalibrationSolver::toCvMethod(Method method) const {
    switch (method) {
        case TSAI:       return cv::CALIB_HAND_EYE_TSAI;
        case PARK:       return cv::CALIB_HAND_EYE_PARK;
        case HORAUD:     return cv::CALIB_HAND_EYE_HORAUD;
        case ANDREFF:    return cv::CALIB_HAND_EYE_ANDREFF;
        case DANIILIDIS: return cv::CALIB_HAND_EYE_DANIILIDIS;
        default:         return cv::CALIB_HAND_EYE_PARK;
    }
}
std::string CalibrationSolver::getMethodName(Method method) {
    switch (method) {
        case TSAI:       return "Tsai-Lenz";
        case PARK:       return "Park-Martin";
        case HORAUD:     return "Horaud-Dornaika";
        case ANDREFF:    return "Andreff";
        case DANIILIDIS: return "Daniilidis";
        default:         return "Unknown";
    }
}
std::vector<size_t> CalibrationSolver::refineByError(
    const std::vector<CalibrationSample>& samples,
    std::vector<size_t> indices,
    int target_pairs,
    int max_iterations) {
    if (indices.empty()) {
        RCLCPP_ERROR(logger_, "No indices provided for refinement");
        return {};
    }
    RCLCPP_INFO(logger_, "\n========== ITERATIVE REFINEMENT ==========");
    RCLCPP_INFO(logger_, "Starting with %zu samples (%zu pairs)",
                indices.size(), indices.size() - 1);
    RCLCPP_INFO(logger_, "Target: %d pairs\n", target_pairs);
    std::vector<size_t> current_indices = indices;
    std::vector<std::pair<size_t, size_t>> worst_pair_history;
    const int PATTERN_THRESHOLD = 3;
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        if (static_cast<int>(current_indices.size()) - 1 <= target_pairs) {
            RCLCPP_INFO(logger_, "\n✓ Reached target of %d pairs", target_pairs);
            break;
        }
        CalibrationResult result = solve(samples, current_indices, false);
        if (!result.success) {
            RCLCPP_WARN(logger_, "Calibration failed during refinement iteration %d",
                       iteration + 1);
            break;
        }
        struct SampleError {
            size_t idx;
            double trans_err_mm;
            double rot_err_deg;
            double combined_err;
        };
        std::vector<SampleError> sample_errors;
        for (size_t idx : current_indices) {
            const auto& sample = samples[idx];
            Eigen::Matrix4d T_camera_pred = sample.sensor_pose * result.transformation;
            const Eigen::Matrix4d& T_camera_meas = sample.camera_pose;
            Eigen::Vector3d t_pred = T_camera_pred.block<3, 1>(0, 3);
            Eigen::Vector3d t_meas = T_camera_meas.block<3, 1>(0, 3);
            double trans_err_mm = (t_meas - t_pred).norm() * 1000.0;
            Eigen::Matrix3d R_pred = T_camera_pred.block<3, 3>(0, 0);
            Eigen::Matrix3d R_meas = T_camera_meas.block<3, 3>(0, 0);
            Eigen::Matrix3d R_rel = R_pred.transpose() * R_meas;
            double trace = R_rel.trace();
            double cos_angle = std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0));
            double rot_err_deg = std::acos(cos_angle) * 180.0 / M_PI;
            if (rot_err_deg > 180.0) {
                rot_err_deg = 360.0 - rot_err_deg;
            }
            double combined_err = trans_err_mm + rot_err_deg * 5.0;
            sample_errors.push_back({idx, trans_err_mm, rot_err_deg, combined_err});
        }
        auto worst_it = std::max_element(sample_errors.begin(), sample_errors.end(),
            [](const SampleError& a, const SampleError& b) {
                return a.combined_err < b.combined_err;
            });
        if (worst_it == sample_errors.end()) {
            RCLCPP_WARN(logger_, "No worst sample found, stopping refinement");
            break;
        }
        SampleError worst = *worst_it;
        worst_pair_history.push_back({worst.idx, worst.idx});  // Dummy pair for compatibility
        if (worst_pair_history.size() > static_cast<size_t>(PATTERN_THRESHOLD)) {
            worst_pair_history.erase(worst_pair_history.begin());
        }
        size_t sample_to_remove = worst.idx;  // Remove worst sample
        std::string removal_reason = "worst_prediction_error";
        if (worst_pair_history.size() >= static_cast<size_t>(PATTERN_THRESHOLD)) {
            std::map<size_t, int> sample_counts;
            for (const auto& [si, sj] : worst_pair_history) {
                sample_counts[si]++;  // Both are the same (dummy pairs)
            }
            std::vector<size_t> problematic_samples;
            for (const auto& [sample, count] : sample_counts) {
                if (count >= PATTERN_THRESHOLD) {
                    problematic_samples.push_back(sample);
                }
            }
            if (!problematic_samples.empty()) {
                if (problematic_samples.size() == 1) {
                    sample_to_remove = problematic_samples[0];
                    removal_reason = "outlier (worst " + std::to_string(sample_counts[sample_to_remove]) + " times)";
                } else {
                    double max_err = -1.0;
                    for (const auto& se : sample_errors) {
                        if (std::find(problematic_samples.begin(), problematic_samples.end(), se.idx) != problematic_samples.end()) {
                            if (se.combined_err > max_err) {
                                max_err = se.combined_err;
                                sample_to_remove = se.idx;
                            }
                        }
                    }
                    removal_reason = "outlier (worst " + std::to_string(sample_counts[sample_to_remove]) + " times)";
                }
                worst_pair_history.clear();
            }
        }
        // Remove the selected sample
        auto remove_it = std::find(current_indices.begin(), current_indices.end(),
                                   sample_to_remove);
        if (remove_it != current_indices.end()) {
            current_indices.erase(remove_it);
        }
        RCLCPP_INFO(logger_,
            "[Iter %2d] Removed sample %3zu: "
            "err=%5.1fmm/%5.1f° | Reason: %s | Remaining: %zu samples",
            iteration + 1, sample_to_remove,
            worst.trans_err_mm, worst.rot_err_deg,
            removal_reason.c_str(), current_indices.size());
    }
    RCLCPP_INFO(logger_, "\n✓ Refinement complete: %zu samples selected\n",
               current_indices.size());
    return current_indices;
}
void CalibrationSolver::computeAndPrintAbsoluteErrors(
    const std::vector<CalibrationSample>& samples,
    const std::vector<size_t>& selected_indices,
    const Eigen::Matrix4d& transformation) {
    if (selected_indices.empty()) {
        RCLCPP_WARN(logger_, "No samples provided for error calculation");
        return;
    }
    std::vector<double> rotation_errors_deg;
    std::vector<double> translation_errors_mm;
    for (size_t idx : selected_indices) {
        const auto& sample = samples[idx];
        Eigen::Matrix4d T_aurora_to_camera_pred = sample.sensor_pose * transformation;
        const Eigen::Matrix4d& T_aurora_to_camera_meas = sample.camera_pose;
        Eigen::Matrix3d R_pred = T_aurora_to_camera_pred.block<3, 3>(0, 0);
        Eigen::Matrix3d R_meas = T_aurora_to_camera_meas.block<3, 3>(0, 0);
        Eigen::Vector3d t_pred = T_aurora_to_camera_pred.block<3, 1>(0, 3);
        Eigen::Vector3d t_meas = T_aurora_to_camera_meas.block<3, 1>(0, 3);
        Eigen::Matrix3d R_rel = R_pred.transpose() * R_meas;
        Eigen::AngleAxisd angle_axis(R_rel);
        double rot_err_rad = std::abs(angle_axis.angle());
        // Normalize to [0, pi]
        if (rot_err_rad > M_PI) {
            rot_err_rad = 2.0 * M_PI - rot_err_rad;
        }
        double rot_err_deg = rot_err_rad * 180.0 / M_PI;
        double trans_err_m = (t_meas - t_pred).norm();
        double trans_err_mm = trans_err_m * 1000.0;
        rotation_errors_deg.push_back(rot_err_deg);
        translation_errors_mm.push_back(trans_err_mm);
    }
    auto compute_stats = [](const std::vector<double>& data) -> std::map<std::string, double> {
        std::map<std::string, double> stats;
        if (data.empty()) {
            return stats;
        }
        // Sort for percentile calculations
        std::vector<double> sorted_data = data;
        std::sort(sorted_data.begin(), sorted_data.end());
        size_t n = sorted_data.size();
        // Min, max
        stats["min"] = sorted_data.front();
        stats["max"] = sorted_data.back();
        // Median
        if (n % 2 == 0) {
            stats["median"] = (sorted_data[n/2 - 1] + sorted_data[n/2]) / 2.0;
        } else {
            stats["median"] = sorted_data[n/2];
        }
        double sum = std::accumulate(data.begin(), data.end(), 0.0);
        stats["mean"] = sum / n;
        // Standard deviation (sample std with ddof=1)
        double mean = stats["mean"];
        double sq_sum = 0.0;
        for (double val : data) {
            sq_sum += (val - mean) * (val - mean);
        }
        stats["std"] = std::sqrt(sq_sum / (n - 1));
        // RMS (root mean square)
        double sq_sum_rms = 0.0;
        for (double val : data) {
            sq_sum_rms += val * val;
        }
        stats["rms"] = std::sqrt(sq_sum_rms / n);
        stats["count"] = static_cast<double>(n);
        return stats;
    };
    auto rot_stats = compute_stats(rotation_errors_deg);
    auto trans_stats = compute_stats(translation_errors_mm);
    RCLCPP_INFO(logger_, "\n========== ABSOLUTE ERRORS (Direct Prediction Comparison) ==========");
    RCLCPP_INFO(logger_, "Method: T_predicted = T_sensor @ X, compare with T_measured");
    RCLCPP_INFO(logger_, "Samples: %zu", selected_indices.size());
    RCLCPP_INFO(logger_, "");
    RCLCPP_INFO(logger_, "ROTATION ERRORS (degrees):");
    RCLCPP_INFO(logger_, "  min    = %7.3f°", rot_stats["min"]);
    RCLCPP_INFO(logger_, "  median = %7.3f°", rot_stats["median"]);
    RCLCPP_INFO(logger_, "  max    = %7.3f°", rot_stats["max"]);
    RCLCPP_INFO(logger_, "  mean   = %7.3f°", rot_stats["mean"]);
    RCLCPP_INFO(logger_, "  std    = %7.3f°", rot_stats["std"]);
    RCLCPP_INFO(logger_, "  rms    = %7.3f°", rot_stats["rms"]);
    RCLCPP_INFO(logger_, "");
    RCLCPP_INFO(logger_, "TRANSLATION ERRORS (mm):");
    RCLCPP_INFO(logger_, "  min    = %7.3f mm", trans_stats["min"]);
    RCLCPP_INFO(logger_, "  median = %7.3f mm", trans_stats["median"]);
    RCLCPP_INFO(logger_, "  max    = %7.3f mm", trans_stats["max"]);
    RCLCPP_INFO(logger_, "  mean   = %7.3f mm", trans_stats["mean"]);
    RCLCPP_INFO(logger_, "  std    = %7.3f mm", trans_stats["std"]);
    RCLCPP_INFO(logger_, "  rms    = %7.3f mm", trans_stats["rms"]);
    RCLCPP_INFO(logger_, "====================================================================\n");
}
// NONLINEAR REFINEMENT (Bundle Adjustment)
/**
 * @brief Ceres cost function for hand-eye calibration refinement
 *
 * Residual: T_camera_meas vs T_camera_pred = T_sensor @ X
 * Where X is parametrized as [translation(3), rotation_axis_angle(3)]
 */
struct HandEyeResidual {
    HandEyeResidual(const Eigen::Matrix4d& T_sensor_measured,
                    const Eigen::Matrix4d& T_camera_measured,
                    double rotation_weight)
        : T_sensor_measured_(T_sensor_measured),
          T_camera_measured_(T_camera_measured),
          rotation_weight_(rotation_weight) {}
    template <typename T>
    bool operator()(const T* const translation,
                   const T* const rotation_axis_angle,
                   T* residuals) const {
        // X = [R(rotation_axis_angle) | translation]
        //     [        0              |      1      ]
        T rotation_matrix[9];
        ceres::AngleAxisToRotationMatrix(rotation_axis_angle, rotation_matrix);
        Eigen::Matrix<T, 3, 3> R_sensor = T_sensor_measured_.block<3, 3>(0, 0).cast<T>();
        Eigen::Matrix<T, 3, 1> t_sensor = T_sensor_measured_.block<3, 1>(0, 3).cast<T>();
        Eigen::Matrix<T, 3, 3> R_X;
        R_X << rotation_matrix[0], rotation_matrix[1], rotation_matrix[2],
               rotation_matrix[3], rotation_matrix[4], rotation_matrix[5],
               rotation_matrix[6], rotation_matrix[7], rotation_matrix[8];
        Eigen::Matrix<T, 3, 1> t_X;
        t_X << translation[0], translation[1], translation[2];
        Eigen::Matrix<T, 3, 3> R_camera_pred = R_sensor * R_X;
        Eigen::Matrix<T, 3, 1> t_camera_pred = R_sensor * t_X + t_sensor;
        Eigen::Matrix<T, 3, 3> R_camera_meas = T_camera_measured_.block<3, 3>(0, 0).cast<T>();
        Eigen::Matrix<T, 3, 1> t_camera_meas = T_camera_measured_.block<3, 1>(0, 3).cast<T>();
        // Translation residuals (meters)
        Eigen::Matrix<T, 3, 1> translation_error = t_camera_meas - t_camera_pred;
        residuals[0] = translation_error(0);
        residuals[1] = translation_error(1);
        residuals[2] = translation_error(2);
        // R_error = R_pred^T * R_meas
        Eigen::Matrix<T, 3, 3> R_error = R_camera_pred.transpose() * R_camera_meas;
        T rotation_error[9];
        rotation_error[0] = R_error(0, 0);
        rotation_error[1] = R_error(0, 1);
        rotation_error[2] = R_error(0, 2);
        rotation_error[3] = R_error(1, 0);
        rotation_error[4] = R_error(1, 1);
        rotation_error[5] = R_error(1, 2);
        rotation_error[6] = R_error(2, 0);
        rotation_error[7] = R_error(2, 1);
        rotation_error[8] = R_error(2, 2);
        T axis_angle_error[3];
        ceres::RotationMatrixToAngleAxis(rotation_error, axis_angle_error);
        residuals[3] = axis_angle_error[0] * T(rotation_weight_);
        residuals[4] = axis_angle_error[1] * T(rotation_weight_);
        residuals[5] = axis_angle_error[2] * T(rotation_weight_);
        return true;
    }
    static ceres::CostFunction* Create(const Eigen::Matrix4d& T_sensor_measured,
                                       const Eigen::Matrix4d& T_camera_measured,
                                       double rotation_weight) {
        return new ceres::AutoDiffCostFunction<HandEyeResidual, 6, 3, 3>(
            new HandEyeResidual(T_sensor_measured, T_camera_measured, rotation_weight));
    }
private:
    const Eigen::Matrix4d T_sensor_measured_;
    const Eigen::Matrix4d T_camera_measured_;
    const double rotation_weight_;
};
Eigen::Matrix4d CalibrationSolver::refineHandEyeNonlinear(
    const std::vector<CalibrationSample>& samples,
    const std::vector<size_t>& selected_indices,
    const Eigen::Matrix4d& X_init,
    int max_iterations,
    double rotation_weight) {
    RCLCPP_INFO(logger_, "\n========== NONLINEAR REFINEMENT (Bundle Adjustment) ==========");
    RCLCPP_INFO(logger_, "Using Ceres Solver with Levenberg-Marquardt");
    RCLCPP_INFO(logger_, "Rotation weight: %.1f", rotation_weight);
    RCLCPP_INFO(logger_, "Max iterations: %d", max_iterations);
    double initial_trans_norm = X_init.block<3, 1>(0, 3).norm() * 1000.0;
    RCLCPP_INFO(logger_, "Initial transformation norm: %.3f mm\n", initial_trans_norm);
    // Parametrize X as [translation(3), rotation_axis_angle(3)]
    Eigen::Vector3d translation = X_init.block<3, 1>(0, 3);
    Eigen::Matrix3d rotation_matrix = X_init.block<3, 3>(0, 0);
    double rotation_matrix_array[9] = {
        rotation_matrix(0, 0), rotation_matrix(0, 1), rotation_matrix(0, 2),
        rotation_matrix(1, 0), rotation_matrix(1, 1), rotation_matrix(1, 2),
        rotation_matrix(2, 0), rotation_matrix(2, 1), rotation_matrix(2, 2)
    };
    double rotation_axis_angle[3];
    ceres::RotationMatrixToAngleAxis(rotation_matrix_array, rotation_axis_angle);
    // Parameters to optimize
    double params_translation[3] = {translation(0), translation(1), translation(2)};
    double params_rotation[3] = {rotation_axis_angle[0], rotation_axis_angle[1], rotation_axis_angle[2]};
    ceres::Problem problem;
    for (size_t idx : selected_indices) {
        if (idx >= samples.size()) {
            RCLCPP_WARN(logger_, "Invalid sample index: %zu", idx);
            continue;
        }
        const auto& sample = samples[idx];
        ceres::CostFunction* cost_function = HandEyeResidual::Create(
            sample.sensor_pose,
            sample.camera_pose,
            rotation_weight
        );
        problem.AddResidualBlock(cost_function, nullptr, params_translation, params_rotation);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = max_iterations;
    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-10;
    options.parameter_tolerance = 1e-8;
    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // Print summary
    RCLCPP_INFO(logger_, "  [Nonlinear Refinement] Optimization complete:");
    RCLCPP_INFO(logger_, "    Iterations: %d", static_cast<int>(summary.iterations.size()));
    RCLCPP_INFO(logger_, "    Initial cost: %.6e", summary.initial_cost);
    RCLCPP_INFO(logger_, "    Final cost: %.6e", summary.final_cost);
    double improvement = 0.0;
    if (summary.initial_cost > 0) {
        improvement = (summary.initial_cost - summary.final_cost) / summary.initial_cost * 100.0;
    }
    RCLCPP_INFO(logger_, "    Improvement: %.2f%%", improvement);
    RCLCPP_INFO(logger_, "    Termination: %s",
               ceres::TerminationTypeToString(summary.termination_type));
    Eigen::Matrix4d X_refined = Eigen::Matrix4d::Identity();
    // Set translation
    X_refined(0, 3) = params_translation[0];
    X_refined(1, 3) = params_translation[1];
    X_refined(2, 3) = params_translation[2];
    double refined_rotation_matrix[9];
    ceres::AngleAxisToRotationMatrix(params_rotation, refined_rotation_matrix);
    X_refined(0, 0) = refined_rotation_matrix[0];
    X_refined(0, 1) = refined_rotation_matrix[1];
    X_refined(0, 2) = refined_rotation_matrix[2];
    X_refined(1, 0) = refined_rotation_matrix[3];
    X_refined(1, 1) = refined_rotation_matrix[4];
    X_refined(1, 2) = refined_rotation_matrix[5];
    X_refined(2, 0) = refined_rotation_matrix[6];
    X_refined(2, 1) = refined_rotation_matrix[7];
    X_refined(2, 2) = refined_rotation_matrix[8];
    double refined_trans_norm = X_refined.block<3, 1>(0, 3).norm() * 1000.0;
    RCLCPP_INFO(logger_, "  Refined transformation norm: %.3f mm", refined_trans_norm);
    RCLCPP_INFO(logger_, "==============================================================\n");
    return X_refined;
}
} // namespace eye_in_hand_calibration