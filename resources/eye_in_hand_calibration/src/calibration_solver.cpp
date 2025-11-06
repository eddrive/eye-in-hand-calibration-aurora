#include "eye_in_hand_calibration/calibration_solver.hpp"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <map>
#include <algorithm>
#include <numeric>

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

        // Prepare calibration data
        std::vector<cv::Mat> R_gripper2base, t_gripper2base;
        std::vector<cv::Mat> R_target2cam, t_target2cam;

        prepareCalibrationData(samples, selected_indices,
                              R_gripper2base, t_gripper2base,
                              R_target2cam, t_target2cam);

        // Perform calibration
        cv::Mat R_cam2gripper, t_cam2gripper;

        cv::HandEyeCalibrationMethod cv_method = toCvMethod(method_);

        if (verbose) {
            RCLCPP_INFO(logger_, "Running OpenCV calibrateHandEye...");
        }

        cv::calibrateHandEye(R_gripper2base, t_gripper2base,
                            R_target2cam, t_target2cam,
                            R_cam2gripper, t_cam2gripper,
                            cv_method);

        // Convert result
        result.transformation = cvToEigen(R_cam2gripper, t_cam2gripper);

        // Evaluate calibration quality
        result.average_error = evaluateCalibration(samples, selected_indices,
                                                   result.transformation);

        // Calculate detailed error statistics
        std::vector<double> errors;
        for (size_t i = 0; i < selected_indices.size(); ++i) {
            for (size_t j = i + 1; j < selected_indices.size(); ++j) {
                size_t idx_i = selected_indices[i];
                size_t idx_j = selected_indices[j];

                Eigen::Matrix4d A = samples[idx_j].sensor_pose *
                                   samples[idx_i].sensor_pose.inverse();
                Eigen::Matrix4d B = samples[idx_j].camera_pose *
                                   samples[idx_i].camera_pose.inverse();

                double error = calculatePairError(A, B, result.transformation);
                errors.push_back(error);
            }
        }

        if (!errors.empty()) {
            result.min_error = *std::min_element(errors.begin(), errors.end());
            result.max_error = *std::max_element(errors.begin(), errors.end());
        }

        // Validate result
        result.success = validateResult(result);

        // Print result (only if verbose)
        if (verbose) {
            RCLCPP_INFO(logger_, "\n========== CALIBRATION RESULT ==========");
            RCLCPP_INFO(logger_, "Success: %s", result.success ? "YES" : "NO");
            RCLCPP_INFO(logger_, "Samples used: %zu", result.num_samples_used);

            RCLCPP_INFO(logger_, "\n--- AX=XB Errors (Relative Motion Consistency) ---");
            RCLCPP_INFO(logger_, "Number of pairs: %zu", errors.size());
            RCLCPP_INFO(logger_, "Average error: %.6f m", result.average_error);
            RCLCPP_INFO(logger_, "Min error: %.6f m", result.min_error);
            RCLCPP_INFO(logger_, "Max error: %.6f m", result.max_error);

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
        
        // ========================================
        // GRIPPER TO BASE (Sensor pose in Aurora frame)
        // ========================================
        // sample.sensor_pose is T_aurora_to_sensor
        // OpenCV expects T_base_to_gripper, which is the same thing
        cv::Mat R_gb, t_gb;
        eigenToCv(sample.sensor_pose, R_gb, t_gb);
        R_gripper2base.push_back(R_gb);
        t_gripper2base.push_back(t_gb);
        
        // ========================================
        // TARGET TO CAMERA - REQUIRES INVERSION!
        // ========================================
        // sample.camera_pose is T_aurora_to_camera (camera in Aurora frame)
        // But OpenCV expects T_camera_to_target (T_camera_to_aurora in our case)
        // So we need to INVERT it!
        
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

double CalibrationSolver::evaluateCalibration(
    const std::vector<CalibrationSample>& samples,
    const std::vector<size_t>& selected_indices,
    const Eigen::Matrix4d& transformation) {
    
    if (selected_indices.size() < 2) {
        return 0.0;
    }
    
    double total_error = 0.0;
    int num_pairs = 0;
    
    // Evaluate all pairs: AX = XB
    // where X is the hand-eye transformation
    for (size_t i = 0; i < selected_indices.size(); ++i) {
        for (size_t j = i + 1; j < selected_indices.size(); ++j) {
            size_t idx_i = selected_indices[i];
            size_t idx_j = selected_indices[j];
            
            // A = gripper_j * gripper_i^-1
            Eigen::Matrix4d A = samples[idx_j].sensor_pose * 
                               samples[idx_i].sensor_pose.inverse();
            
            // B = camera_j * camera_i^-1
            Eigen::Matrix4d B = samples[idx_j].camera_pose * 
                               samples[idx_i].camera_pose.inverse();
            
            double error = calculatePairError(A, B, transformation);
            total_error += error;
            num_pairs++;
        }
    }
    
    return num_pairs > 0 ? total_error / num_pairs : 0.0;
}

double CalibrationSolver::calculatePairError(const Eigen::Matrix4d& A,
                                             const Eigen::Matrix4d& B,
                                             const Eigen::Matrix4d& X) const {
    // Calculate AX = XB error
    // Error metric: ||AX - XB|| (Frobenius norm of translation difference)
    
    Eigen::Matrix4d AX = A * X;
    Eigen::Matrix4d XB = X * B;
    
    // Translation error
    Eigen::Vector3d t_error = AX.block<3, 1>(0, 3) - XB.block<3, 1>(0, 3);
    double translation_error = t_error.norm();
    
    // Rotation error (optional, less commonly used)
    // Eigen::Matrix3d R_error = AX.block<3,3>(0,0) * XB.block<3,3>(0,0).transpose();
    // double rotation_error = std::acos((R_error.trace() - 1.0) / 2.0);
    
    return translation_error;
}

void CalibrationSolver::printTransformation(const Eigen::Matrix4d& transform) const {
    Eigen::Vector3d translation = transform.block<3, 1>(0, 3);
    Eigen::Matrix3d rotation = transform.block<3, 3>(0, 0);
    
    // Euler angles
    Eigen::Vector3d euler = rotation.eulerAngles(0, 1, 2) * 180.0 / M_PI;
    
    // Quaternion
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
    
    // Check for NaN or Inf
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (!std::isfinite(result.transformation(i, j))) {
                RCLCPP_ERROR(logger_, "Transformation contains NaN or Inf");
                return false;
            }
        }
    }
    
    // Check rotation matrix properties
    Eigen::Matrix3d R = result.transformation.block<3, 3>(0, 0);
    
    // Determinant should be +1
    double det = R.determinant();
    if (std::abs(det - 1.0) > 0.1) {
        RCLCPP_WARN(logger_, "Rotation matrix determinant: %.6f (expected 1.0)", det);
    }
    
    // Should be orthogonal: R * R^T = I
    Eigen::Matrix3d I = R * R.transpose();
    double orthogonality_error = (I - Eigen::Matrix3d::Identity()).norm();
    if (orthogonality_error > 0.1) {
        RCLCPP_WARN(logger_, "Rotation matrix orthogonality error: %.6f", 
                   orthogonality_error);
    }
    
    // Check translation magnitude (sanity check)
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
        
        // Metadata
        out << YAML::Key << "timestamp" << YAML::Value << std::time(nullptr);
        out << YAML::Key << "method" << YAML::Value << getMethodName(static_cast<Method>(result.method_used));
        out << YAML::Key << "samples_used" << YAML::Value << result.num_samples_used;
        out << YAML::Key << "result_frame_id" << YAML::Value << result_frame_id;
        out << YAML::Key << "target_frame_id" << YAML::Value << target_frame_id;
        out << YAML::Key << "success" << YAML::Value << result.success;
        
        // Transformation matrix
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
        
        // Translation
        Eigen::Vector3d translation = result.transformation.block<3, 1>(0, 3);
        out << YAML::Key << "translation";
        out << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "x" << YAML::Value << translation.x();
        out << YAML::Key << "y" << YAML::Value << translation.y();
        out << YAML::Key << "z" << YAML::Value << translation.z();
        out << YAML::EndMap;
        
        // Rotation (quaternion)
        Eigen::Matrix3d rotation = result.transformation.block<3, 3>(0, 0);
        Eigen::Quaterniond quat(rotation);
        out << YAML::Key << "rotation";
        out << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "x" << YAML::Value << quat.x();
        out << YAML::Key << "y" << YAML::Value << quat.y();
        out << YAML::Key << "z" << YAML::Value << quat.z();
        out << YAML::Key << "w" << YAML::Value << quat.w();
        out << YAML::EndMap;
        
        // Quality metrics
        out << YAML::Key << "quality";
        out << YAML::Value << YAML::BeginMap;
        out << YAML::Key << "average_error" << YAML::Value << result.average_error;
        out << YAML::Key << "min_error" << YAML::Value << result.min_error;
        out << YAML::Key << "max_error" << YAML::Value << result.max_error;
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

// ============================================
// ITERATIVE REFINEMENT (Python script logic)
// ============================================

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

    // Track worst pairs history for pattern detection
    std::vector<std::pair<size_t, size_t>> worst_pair_history;
    const int PATTERN_THRESHOLD = 3;

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        if (static_cast<int>(current_indices.size()) - 1 <= target_pairs) {
            RCLCPP_INFO(logger_, "\n✓ Reached target of %d pairs", target_pairs);
            break;
        }

        // Calibrate with current samples (verbose=false to avoid printing during refinement)
        CalibrationResult result = solve(samples, current_indices, false);

        if (!result.success) {
            RCLCPP_WARN(logger_, "Calibration failed during refinement iteration %d",
                       iteration + 1);
            break;
        }

        // Evaluate error for each consecutive pair
        struct PairError {
            size_t k;
            size_t idx_i;
            size_t idx_j;
            double trans_err_mm;
            double rot_err_deg;
            double combined_err;
        };

        std::vector<PairError> pair_errors;

        for (size_t k = 0; k < current_indices.size() - 1; ++k) {
            size_t idx_i = current_indices[k];
            size_t idx_j = current_indices[k + 1];

            const auto& sample_i = samples[idx_i];
            const auto& sample_j = samples[idx_j];

            // A = sensor_j * sensor_i^-1
            Eigen::Matrix4d A = sample_j.sensor_pose * sample_i.sensor_pose.inverse();

            // B = camera_j * camera_i^-1
            Eigen::Matrix4d B = sample_j.camera_pose * sample_i.camera_pose.inverse();

            // Calculate AX and XB
            Eigen::Matrix4d AX = A * result.transformation;
            Eigen::Matrix4d XB = result.transformation * B;
            Eigen::Matrix4d Delta = AX * XB.inverse();

            // Errors
            double trans_err_mm = Delta.block<3, 1>(0, 3).norm() * 1000.0;

            double trace = Delta.block<3, 3>(0, 0).trace();
            double cos_angle = std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0));
            double rot_err_deg = std::acos(cos_angle) * 180.0 / M_PI;

            // Combined error (weight rotation more)
            double combined_err = trans_err_mm + rot_err_deg * 5.0;

            pair_errors.push_back({k, idx_i, idx_j, trans_err_mm, rot_err_deg, combined_err});
        }

        // Find worst pair
        auto worst_it = std::max_element(pair_errors.begin(), pair_errors.end(),
            [](const PairError& a, const PairError& b) {
                return a.combined_err < b.combined_err;
            });

        if (worst_it == pair_errors.end()) {
            RCLCPP_WARN(logger_, "No worst pair found, stopping refinement");
            break;
        }

        PairError worst = *worst_it;
        worst_pair_history.push_back({worst.idx_i, worst.idx_j});

        // Keep only recent history
        if (worst_pair_history.size() > static_cast<size_t>(PATTERN_THRESHOLD)) {
            worst_pair_history.erase(worst_pair_history.begin());
        }

        // Analyze if a sample appears repeatedly in worst pairs
        size_t sample_to_remove = worst.idx_j;  // Default: remove second element
        std::string removal_reason = "worst_pair";

        if (worst_pair_history.size() >= static_cast<size_t>(PATTERN_THRESHOLD)) {
            // Count how many times each sample appears
            std::map<size_t, int> sample_counts;
            for (const auto& [si, sj] : worst_pair_history) {
                sample_counts[si]++;
                sample_counts[sj]++;
            }

            // Find problematic samples
            std::vector<size_t> problematic_samples;
            for (const auto& [sample, count] : sample_counts) {
                if (count >= PATTERN_THRESHOLD) {
                    problematic_samples.push_back(sample);
                }
            }

            if (!problematic_samples.empty()) {
                // If multiple problematic, choose one with highest average error
                if (problematic_samples.size() == 1) {
                    sample_to_remove = problematic_samples[0];
                } else {
                    std::map<size_t, double> sample_errors;
                    for (size_t s : problematic_samples) {
                        std::vector<double> errors;
                        for (const auto& pe : pair_errors) {
                            if (pe.idx_i == s || pe.idx_j == s) {
                                errors.push_back(pe.combined_err);
                            }
                        }
                        if (!errors.empty()) {
                            double avg = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
                            sample_errors[s] = avg;
                        }
                    }

                    // Find max error sample
                    auto max_it = std::max_element(sample_errors.begin(), sample_errors.end(),
                        [](const auto& a, const auto& b) { return a.second < b.second; });

                    if (max_it != sample_errors.end()) {
                        sample_to_remove = max_it->first;
                        removal_reason = "outlier (appeared in " +
                                       std::to_string(sample_counts[sample_to_remove]) + " worst pairs)";
                    }
                }

                // Clear history after removing outlier
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
            "[Iter %2d] Removed sample %3zu (pair %3zu→%3zu): "
            "err=%5.1fmm/%5.1f° | Reason: %s | Remaining: %zu samples",
            iteration + 1, sample_to_remove, worst.idx_i, worst.idx_j,
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

    // For each sample, compute: T_predicted = T_sensor @ X, compare with T_measured
    for (size_t idx : selected_indices) {
        const auto& sample = samples[idx];

        // Predicted camera pose in Aurora frame
        Eigen::Matrix4d T_aurora_to_camera_pred = sample.sensor_pose * transformation;

        // Measured camera pose in Aurora frame
        const Eigen::Matrix4d& T_aurora_to_camera_meas = sample.camera_pose;

        // Extract rotations
        Eigen::Matrix3d R_pred = T_aurora_to_camera_pred.block<3, 3>(0, 0);
        Eigen::Matrix3d R_meas = T_aurora_to_camera_meas.block<3, 3>(0, 0);

        // Extract translations
        Eigen::Vector3d t_pred = T_aurora_to_camera_pred.block<3, 1>(0, 3);
        Eigen::Vector3d t_meas = T_aurora_to_camera_meas.block<3, 1>(0, 3);

        // Rotation error: angle of relative rotation R_pred.T @ R_meas
        Eigen::Matrix3d R_rel = R_pred.transpose() * R_meas;

        // Convert to angle-axis representation to get rotation angle
        Eigen::AngleAxisd angle_axis(R_rel);
        double rot_err_rad = std::abs(angle_axis.angle());

        // Normalize to [0, pi]
        if (rot_err_rad > M_PI) {
            rot_err_rad = 2.0 * M_PI - rot_err_rad;
        }

        double rot_err_deg = rot_err_rad * 180.0 / M_PI;

        // Translation error: euclidean distance
        double trans_err_m = (t_meas - t_pred).norm();
        double trans_err_mm = trans_err_m * 1000.0;

        rotation_errors_deg.push_back(rot_err_deg);
        translation_errors_mm.push_back(trans_err_mm);
    }

    // Compute statistics for rotation errors
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

        // Mean
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

    // Print results in format matching Python script
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

} // namespace eye_in_hand_calibration