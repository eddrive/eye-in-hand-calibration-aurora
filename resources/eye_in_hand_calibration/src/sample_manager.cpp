#include "eye_in_hand_calibration/sample_manager.hpp"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>

namespace eye_in_hand_calibration {

SampleManager::SampleManager(double min_movement_threshold,
                             double min_rotation_threshold,
                             double max_reprojection_error,
                             rclcpp::Logger logger)
    : next_sample_id_(0),
      min_movement_threshold_(min_movement_threshold),
      min_rotation_threshold_(min_rotation_threshold),
      max_reprojection_error_(max_reprojection_error),
      logger_(logger)
{
    RCLCPP_INFO(logger_, 
                "SampleManager initialized: min_movement=%.3fm, min_rotation=%.3frad, max_error=%.1fpx",
                min_movement_threshold_, min_rotation_threshold_, max_reprojection_error_);
}

bool SampleManager::shouldSaveSample(const Eigen::Matrix4d& sensor_pose,
                                     [[maybe_unused]] const Eigen::Matrix4d& camera_pose) {
    std::lock_guard<std::mutex> lock(samples_mutex_);
    
    // Always save first sample
    if (samples_.empty()) {
        return true;
    }
    
    // Check diversity against last sample (quick filter)
    const auto& last_sample = samples_.back();
    
    double translation_dist = calculateTranslationDistance(sensor_pose, 
                                                          last_sample.sensor_pose);
    double rotation_angle = calculateRotationAngle(sensor_pose, 
                                                   last_sample.sensor_pose);
    
    // Require minimum movement OR rotation
    bool is_diverse = (translation_dist >= min_movement_threshold_) ||
                     (rotation_angle >= min_rotation_threshold_);
    
    if (!is_diverse) {
        RCLCPP_DEBUG(logger_,
                    "Sample too similar to last: trans=%.3fm, rot=%.3frad",
                    translation_dist, rotation_angle);
    }
    
    return is_diverse;
}

int SampleManager::addSample(const Eigen::Matrix4d& sensor_pose,
                             const Eigen::Matrix4d& camera_pose,
                             double reprojection_error, 
                             double distance_to_target) {
    std::lock_guard<std::mutex> lock(samples_mutex_);
    
    CalibrationSample sample;
    sample.sensor_pose = sensor_pose;
    sample.camera_pose = camera_pose;
    sample.reprojection_error = reprojection_error;
    sample.distance_to_target = distance_to_target;
    sample.sample_id = next_sample_id_++;
    
    samples_.push_back(sample);
    
    RCLCPP_DEBUG(logger_,
                "Sample %d added: reproj=%.3fpx, dist=%.3fm, total=%zu", 
                sample.sample_id, reprojection_error, distance_to_target, 
                samples_.size());
    
    return sample.sample_id;
}

std::vector<size_t> SampleManager::selectDiverseSamples(int num_samples) {
    std::lock_guard<std::mutex> lock(samples_mutex_);
    
    if (samples_.empty()) {
        RCLCPP_ERROR(logger_, "No samples available for selection");
        return {};
    }
    
    RCLCPP_INFO(logger_,
               "\n========== SELECTING BEST %d SAMPLES FROM %zu ==========",
               num_samples, samples_.size());
    
    // Step 1: Filter by quality (reprojection error)
    std::vector<size_t> valid_indices = filterByQuality();
    
    if (valid_indices.empty()) {
        RCLCPP_ERROR(logger_, "No samples meet quality criteria!");
        return {};
    }
    
    RCLCPP_INFO(logger_,
               "Quality filter: %zu valid samples (error < %.1fpx)",
               valid_indices.size(), max_reprojection_error_);
    
    // Adjust num_samples if we have fewer valid samples
    if (static_cast<int>(valid_indices.size()) < num_samples) {
        RCLCPP_WARN(logger_,
                   "Only %zu valid samples available (requested %d), using all",
                   valid_indices.size(), num_samples);
        num_samples = valid_indices.size();
    }
    
    std::vector<size_t> selected;
    std::vector<bool> used(samples_.size(), false);
    
    // Step 2: Start with lowest reprojection error
    size_t best_idx = valid_indices[0];
    double best_error = samples_[valid_indices[0]].reprojection_error;
    
    for (size_t idx : valid_indices) {
        if (samples_[idx].reprojection_error < best_error) {
            best_error = samples_[idx].reprojection_error;
            best_idx = idx;
        }
    }
    
    selected.push_back(best_idx);
    used[best_idx] = true;
    
    RCLCPP_INFO(logger_, 
               "Starting with sample %d (error: %.3fpx)",
               samples_[best_idx].sample_id, best_error);
    
    // Step 3: Iteratively add most diverse samples
    while (static_cast<int>(selected.size()) < num_samples) {
        double max_diversity = -1.0;
        size_t best_candidate = valid_indices[0];
        
        for (size_t idx : valid_indices) {
            if (used[idx]) continue;
            
            double diversity = calculateDiversityScore(idx, selected);
            
            if (diversity > max_diversity) {
                max_diversity = diversity;
                best_candidate = idx;
            }
        }
        
        selected.push_back(best_candidate);
        used[best_candidate] = true;
        
        if (selected.size() % 5 == 0) {
            RCLCPP_INFO(logger_, "Selected %zu/%d samples...",
                       selected.size(), num_samples);
        }
    }
    
    RCLCPP_INFO(logger_, "✓ Selected %zu diverse samples", selected.size());
    
    return selected;
}

std::vector<size_t> SampleManager::filterByQuality() const {
    std::vector<size_t> valid_indices;
    
    for (size_t i = 0; i < samples_.size(); ++i) {
        if (samples_[i].reprojection_error < max_reprojection_error_) {
            valid_indices.push_back(i);
        }
    }
    
    return valid_indices;
}

double SampleManager::calculateDiversityScore(
    size_t candidate_idx,
    const std::vector<size_t>& selected_indices) const {
    
    // Find minimum distance to any selected sample
    double min_distance = std::numeric_limits<double>::max();
    
    for (size_t sel_idx : selected_indices) {
        // Translation distance
        double trans_dist = calculateTranslationDistance(
            samples_[candidate_idx].sensor_pose,
            samples_[sel_idx].sensor_pose
        );
        
        // Rotation distance
        double rot_angle = calculateRotationAngle(
            samples_[candidate_idx].sensor_pose,
            samples_[sel_idx].sensor_pose
        );
        
        // Combined metric: translation + weighted rotation
        // Weight rotation by 0.1 to convert radians to approximate meter scale
        double combined_distance = trans_dist + rot_angle * 0.1;
        
        min_distance = std::min(min_distance, combined_distance);
    }
    
    // Penalize high reprojection error
    double quality_factor = 1.0 / (1.0 + samples_[candidate_idx].reprojection_error * 0.1);
    
    // Diversity score: how far from nearest selected sample, weighted by quality
    return min_distance * quality_factor;
}

double SampleManager::calculateTranslationDistance(
    const Eigen::Matrix4d& pose1,
    const Eigen::Matrix4d& pose2) const {
    
    Eigen::Vector3d t1 = pose1.block<3, 1>(0, 3);
    Eigen::Vector3d t2 = pose2.block<3, 1>(0, 3);
    
    return (t1 - t2).norm();
}

double SampleManager::calculateRotationAngle(
    const Eigen::Matrix4d& pose1,
    const Eigen::Matrix4d& pose2) const {
    
    Eigen::Matrix3d R1 = pose1.block<3, 3>(0, 0);
    Eigen::Matrix3d R2 = pose2.block<3, 3>(0, 0);
    
    Eigen::Matrix3d R_diff = R1.transpose() * R2;
    
    // Extract angle from rotation matrix
    // trace(R) = 1 + 2*cos(theta)
    double trace = R_diff.trace();
    double cos_angle = (trace - 1.0) / 2.0;
    
    // Clamp to valid range to avoid numerical issues
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
    
    return std::acos(cos_angle);
}

const std::vector<CalibrationSample>& SampleManager::getSamples() const {
    return samples_;
}

size_t SampleManager::getNumSamples() const {
    std::lock_guard<std::mutex> lock(samples_mutex_);
    return samples_.size();
}

void SampleManager::clearSamples() {
    std::lock_guard<std::mutex> lock(samples_mutex_);
    samples_.clear();
    next_sample_id_ = 0;
    RCLCPP_INFO(logger_, "All samples cleared");
}

bool SampleManager::saveAllSamples(const std::string& filename) {
    std::lock_guard<std::mutex> lock(samples_mutex_);
    
    try {
        YAML::Emitter out;
        
        out << YAML::BeginMap;
        out << YAML::Key << "collected_samples";
        out << YAML::Value << YAML::BeginMap;
        
        out << YAML::Key << "timestamp" << YAML::Value << std::time(nullptr);
        out << YAML::Key << "total_samples" << YAML::Value << samples_.size();
        
        out << YAML::Key << "samples";
        out << YAML::Value << YAML::BeginSeq;
        
        for (const auto& sample : samples_) {
            writeSampleToYAML(&out, sample);
        }
        
        out << YAML::EndSeq;
        out << YAML::EndMap;
        out << YAML::EndMap;
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            RCLCPP_ERROR(logger_, "Failed to open file: %s", filename.c_str());
            return false;
        }
        
        file << out.c_str();
        file.close();
        
        RCLCPP_INFO(logger_, "✓ Saved %zu samples to: %s",
                   samples_.size(), filename.c_str());
        
        return true;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger_, "Failed to save samples: %s", e.what());
        return false;
    }
}

bool SampleManager::savePosePairs(const std::string& filename,
                                  const std::vector<size_t>& selected_indices) {
    std::lock_guard<std::mutex> lock(samples_mutex_);
    
    try {
        std::ofstream file(filename);
        if (!file.is_open()) {
            RCLCPP_ERROR(logger_, "Failed to open file: %s", filename.c_str());
            return false;
        }
        
        file << "# Selected pose pairs for visualization\n";
        file << "# Both poses are in Aurora frame:\n";
        file << "#   sensor: T_aurora_to_sensor (sensor position in Aurora frame)\n";
        file << "#   camera: T_aurora_to_camera (camera position in Aurora frame)\n";
        file << "pose_pairs:\n";
        
        for (size_t idx : selected_indices) {
            if (idx >= samples_.size()) {
                RCLCPP_WARN(logger_, "Invalid sample index: %zu", idx);
                continue;
            }
            
            const auto& sample = samples_[idx];
            
            // ========================================
            // CAMERA POSE (T_aurora_to_camera)
            // ========================================
            // This is already correct: camera position in Aurora frame
            // NO INVERSION NEEDED!
            auto cam_pos = sample.camera_pose.block<3, 1>(0, 3);
            Eigen::Matrix3d cam_rot = sample.camera_pose.block<3, 3>(0, 0);
            Eigen::Quaterniond cam_quat(cam_rot);
            
            // ========================================
            // SENSOR POSE (T_aurora_to_sensor)
            // ========================================
            auto sens_pos = sample.sensor_pose.block<3, 1>(0, 3);
            Eigen::Matrix3d sens_rot = sample.sensor_pose.block<3, 3>(0, 0);
            Eigen::Quaterniond sens_quat(sens_rot);
            
            file << "  - id: " << sample.sample_id << "\n";
            file << "    reprojection_error: " << sample.reprojection_error << "\n";
            file << "    camera:\n";
            file << "      position: [" << cam_pos(0) << ", " 
                 << cam_pos(1) << ", " << cam_pos(2) << "]\n";
            file << "      orientation: [" << cam_quat.x() << ", "
                 << cam_quat.y() << ", " << cam_quat.z() << ", " 
                 << cam_quat.w() << "]\n";
            file << "    sensor:\n";
            file << "      position: [" << sens_pos(0) << ", "
                 << sens_pos(1) << ", " << sens_pos(2) << "]\n";
            file << "      orientation: [" << sens_quat.x() << ", "
                 << sens_quat.y() << ", " << sens_quat.z() << ", "
                 << sens_quat.w() << "]\n";
        }
        
        file.close();
        
        RCLCPP_INFO(logger_, "✓ Saved %zu pose pairs to: %s",
                   selected_indices.size(), filename.c_str());
        
        return true;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger_, "Failed to save pose pairs: %s", e.what());
        return false;
    }
}

void SampleManager::writeSampleToYAML(void* emitter_ptr, 
                                      const CalibrationSample& sample) const {
    YAML::Emitter& out = *static_cast<YAML::Emitter*>(emitter_ptr);
    
    out << YAML::BeginMap;
    out << YAML::Key << "sample_id" << YAML::Value << sample.sample_id;
    out << YAML::Key << "reprojection_error" << YAML::Value << sample.reprojection_error;
    out << YAML::Key << "distance_to_target" << YAML::Value << sample.distance_to_target;
    
    // ========================================
    // CAMERA POSE (T_aurora_to_camera)
    // ========================================
    // Extract position and orientation (same format as savePosePairs)
    auto cam_pos = sample.camera_pose.block<3, 1>(0, 3);
    Eigen::Matrix3d cam_rot = sample.camera_pose.block<3, 3>(0, 0);
    Eigen::Quaterniond cam_quat(cam_rot);
    
    out << YAML::Key << "camera";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "position" << YAML::Value << YAML::Flow;
    out << YAML::BeginSeq;
    out << cam_pos(0) << cam_pos(1) << cam_pos(2);
    out << YAML::EndSeq;
    out << YAML::Key << "orientation" << YAML::Value << YAML::Flow;
    out << YAML::BeginSeq;
    out << cam_quat.x() << cam_quat.y() << cam_quat.z() << cam_quat.w();
    out << YAML::EndSeq;
    out << YAML::EndMap;
    
    // ========================================
    // SENSOR POSE (T_aurora_to_sensor)
    // ========================================
    auto sens_pos = sample.sensor_pose.block<3, 1>(0, 3);
    Eigen::Matrix3d sens_rot = sample.sensor_pose.block<3, 3>(0, 0);
    Eigen::Quaterniond sens_quat(sens_rot);
    
    out << YAML::Key << "sensor";
    out << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "position" << YAML::Value << YAML::Flow;
    out << YAML::BeginSeq;
    out << sens_pos(0) << sens_pos(1) << sens_pos(2);
    out << YAML::EndSeq;
    out << YAML::Key << "orientation" << YAML::Value << YAML::Flow;
    out << YAML::BeginSeq;
    out << sens_quat.x() << sens_quat.y() << sens_quat.z() << sens_quat.w();
    out << YAML::EndSeq;
    out << YAML::EndMap;
    
    out << YAML::EndMap;
}

SampleManager::SampleStats SampleManager::getStatistics() const {
    std::lock_guard<std::mutex> lock(samples_mutex_);
    
    SampleStats stats;
    stats.total_samples = samples_.size();
    stats.high_quality_samples = 0;
    stats.avg_reprojection_error = 0.0;
    stats.min_reprojection_error = std::numeric_limits<double>::max();
    stats.max_reprojection_error = 0.0;
    stats.translation_range = 0.0;
    stats.rotation_range = 0.0;
    
    if (samples_.empty()) {
        return stats;
    }
    
    // Calculate reprojection error statistics
    for (const auto& sample : samples_) {
        stats.avg_reprojection_error += sample.reprojection_error;
        stats.min_reprojection_error = std::min(stats.min_reprojection_error, 
                                               sample.reprojection_error);
        stats.max_reprojection_error = std::max(stats.max_reprojection_error, 
                                               sample.reprojection_error);
        
        if (sample.reprojection_error < 2.0) {
            stats.high_quality_samples++;
        }
    }
    stats.avg_reprojection_error /= samples_.size();
    
    // Calculate spatial diversity
    for (size_t i = 0; i < samples_.size(); ++i) {
        for (size_t j = i + 1; j < samples_.size(); ++j) {
            double trans_dist = calculateTranslationDistance(
                samples_[i].sensor_pose,
                samples_[j].sensor_pose
            );
            double rot_angle = calculateRotationAngle(
                samples_[i].sensor_pose,
                samples_[j].sensor_pose
            );
            
            stats.translation_range = std::max(stats.translation_range, trans_dist);
            stats.rotation_range = std::max(stats.rotation_range, rot_angle);
        }
    }
    
    return stats;
}

// ============================================
// ADVANCED FILTERING (Python script logic)
// ============================================

std::vector<size_t> SampleManager::selectSamplesAdvanced(
    double max_reproj_error,
    double min_sensor_camera_dist,
    double max_sensor_camera_dist,
    double max_movement_ratio,
    double max_rotation_diff,
    bool use_spatial_diversity,
    double min_trans_dist_mm,
    double min_rot_dist_deg,
    int target_diverse_samples) {

    std::lock_guard<std::mutex> lock(samples_mutex_);

    if (samples_.empty()) {
        RCLCPP_ERROR(logger_, "No samples available for selection");
        return {};
    }

    RCLCPP_INFO(logger_,
               "\n========== APPLYING FILTERS ==========");
    RCLCPP_INFO(logger_, "Total samples: %zu\n", samples_.size());

    // ========================================
    // Filter 1: Reprojection error
    // ========================================
    std::vector<size_t> valid_indices;
    for (size_t i = 0; i < samples_.size(); ++i) {
        if (samples_[i].reprojection_error <= max_reproj_error) {
            valid_indices.push_back(i);
        }
    }

    RCLCPP_INFO(logger_,
               "[Filter 1/3] Reprojection error < %.1fpx: %zu/%zu samples",
               max_reproj_error, valid_indices.size(), samples_.size());

    if (valid_indices.empty()) {
        RCLCPP_ERROR(logger_, "No samples passed reprojection error filter!");
        return {};
    }

    // ========================================
    // Filter 2: Sensor-camera distance (MIN and MAX)
    // ========================================
    std::vector<size_t> distance_filtered;
    for (size_t idx : valid_indices) {
        const auto& sample = samples_[idx];

        // Calculate sensor-camera distance
        Eigen::Vector3d sensor_pos = sample.sensor_pose.block<3, 1>(0, 3);
        Eigen::Vector3d camera_pos = sample.camera_pose.block<3, 1>(0, 3);
        double dist_m = (camera_pos - sensor_pos).norm();
        double dist_mm = dist_m * 1000.0;

        if (dist_m >= min_sensor_camera_dist && dist_m <= max_sensor_camera_dist) {
            distance_filtered.push_back(idx);
        } else {
            if (dist_m < min_sensor_camera_dist) {
                RCLCPP_DEBUG(logger_,
                    "  [Discarding sample %d] Sensor-camera dist: %.1fmm < %.1fmm (too close)",
                    sample.sample_id, dist_mm, min_sensor_camera_dist * 1000.0);
            } else {
                RCLCPP_DEBUG(logger_,
                    "  [Discarding sample %d] Sensor-camera dist: %.1fmm > %.1fmm (too far)",
                    sample.sample_id, dist_mm, max_sensor_camera_dist * 1000.0);
            }
        }
    }

    RCLCPP_INFO(logger_,
               "[Filter 2/3] Sensor-camera distance %.1fmm - %.1fmm: %zu/%zu samples",
               min_sensor_camera_dist * 1000.0, max_sensor_camera_dist * 1000.0,
               distance_filtered.size(), valid_indices.size());

    if (distance_filtered.empty()) {
        RCLCPP_ERROR(logger_, "No samples passed distance filter!");
        return {};
    }

    // ========================================
    // Filter 3: Movement coherence
    // PYTHON STRATEGY: Bidirectional chain expansion
    // - Start from sample with lowest reprojection error
    // - Expand forward: compare with LAST VALID in forward chain
    // - Expand backward: compare with LAST VALID in backward chain
    // ========================================
    std::vector<size_t> coherence_filtered;

    if (distance_filtered.empty()) {
        return {};
    }

    // Find sample with lowest reprojection error (starting point)
    size_t best_idx = distance_filtered[0];
    double best_error = samples_[best_idx].reprojection_error;
    size_t best_pos = 0;

    for (size_t i = 0; i < distance_filtered.size(); ++i) {
        size_t idx = distance_filtered[i];
        if (samples_[idx].reprojection_error < best_error) {
            best_error = samples_[idx].reprojection_error;
            best_idx = idx;
            best_pos = i;
        }
    }

    RCLCPP_INFO(logger_,
        "  Starting from sample %d (lowest reproj error: %.3fpx)",
        samples_[best_idx].sample_id, best_error);

    // Lambda to check coherence between two samples
    auto check_coherence = [&](size_t idx_prev, size_t idx_curr) -> std::pair<bool, std::string> {
        const auto& sample_prev = samples_[idx_prev];
        const auto& sample_curr = samples_[idx_curr];

        // Calculate translation movements
        Eigen::Vector3d sens_pos_prev = sample_prev.sensor_pose.block<3, 1>(0, 3);
        Eigen::Vector3d sens_pos_curr = sample_curr.sensor_pose.block<3, 1>(0, 3);
        double sens_move_mm = (sens_pos_curr - sens_pos_prev).norm() * 1000.0;

        Eigen::Vector3d cam_pos_prev = sample_prev.camera_pose.block<3, 1>(0, 3);
        Eigen::Vector3d cam_pos_curr = sample_curr.camera_pose.block<3, 1>(0, 3);
        double cam_move_mm = (cam_pos_curr - cam_pos_prev).norm() * 1000.0;

        // Calculate rotation movements
        double sens_rot_deg = calculateRotationAngle(
            sample_curr.sensor_pose, sample_prev.sensor_pose) * 180.0 / M_PI;
        double cam_rot_deg = calculateRotationAngle(
            sample_curr.camera_pose, sample_prev.camera_pose) * 180.0 / M_PI;

        // Check translation coherence
        bool trans_ok;
        double ratio;

        if (sens_move_mm < 1e-3 && cam_move_mm < 1e-3) {
            trans_ok = true;
            ratio = 1.0;
        } else if (sens_move_mm < 1e-3 || cam_move_mm < 1e-3) {
            trans_ok = false;
            ratio = std::numeric_limits<double>::infinity();

            std::stringstream reason;
            reason << "sens=" << std::fixed << std::setprecision(1) << sens_move_mm
                   << "mm/" << sens_rot_deg << "°, cam=" << cam_move_mm
                   << "mm/" << cam_rot_deg << "°, ratio=inf";
            return {false, reason.str()};
        } else {
            ratio = std::max(sens_move_mm, cam_move_mm) /
                   std::min(sens_move_mm, cam_move_mm);
            trans_ok = (ratio <= max_movement_ratio);
        }

        // Check rotation coherence
        double rot_diff_deg = std::abs(sens_rot_deg - cam_rot_deg);
        bool rot_ok = (rot_diff_deg <= max_rotation_diff);

        if (trans_ok && rot_ok) {
            return {true, ""};
        } else {
            std::stringstream reason;
            reason << "sens=" << std::fixed << std::setprecision(1) << sens_move_mm
                   << "mm/" << sens_rot_deg << "°, cam=" << cam_move_mm
                   << "mm/" << cam_rot_deg << "°, ratio=" << std::setprecision(2) << ratio
                   << ", Δrot=" << std::setprecision(1) << rot_diff_deg << "°";
            return {false, reason.str()};
        }
    };

    // Start with best sample
    coherence_filtered.push_back(best_idx);

    // Expand forward (increasing indices)
    size_t last_valid_forward = best_idx;
    for (size_t i = best_pos + 1; i < distance_filtered.size(); ++i) {
        size_t idx_curr = distance_filtered[i];
        auto [is_valid, reason] = check_coherence(last_valid_forward, idx_curr);

        if (is_valid) {
            coherence_filtered.push_back(idx_curr);
            last_valid_forward = idx_curr;
        } else {
            RCLCPP_DEBUG(logger_,
                "  [Discarding sample %d] Incoherent with %d: %s",
                samples_[idx_curr].sample_id,
                samples_[last_valid_forward].sample_id,
                reason.c_str());
        }
    }

    // Expand backward (decreasing indices)
    size_t last_valid_backward = best_idx;
    for (int i = static_cast<int>(best_pos) - 1; i >= 0; --i) {
        size_t idx_curr = distance_filtered[i];
        auto [is_valid, reason] = check_coherence(idx_curr, last_valid_backward);

        if (is_valid) {
            coherence_filtered.insert(coherence_filtered.begin(), idx_curr);
            last_valid_backward = idx_curr;
        } else {
            RCLCPP_DEBUG(logger_,
                "  [Discarding sample %d] Incoherent with %d: %s",
                samples_[idx_curr].sample_id,
                samples_[last_valid_backward].sample_id,
                reason.c_str());
        }
    }

    RCLCPP_INFO(logger_,
               "[Filter 3/3] Movement coherence (ratio<%.1f, Δrot<%.1f°): %zu/%zu samples",
               max_movement_ratio, max_rotation_diff,
               coherence_filtered.size(), distance_filtered.size());

    if (coherence_filtered.empty()) {
        RCLCPP_ERROR(logger_, "No samples passed coherence filter!");
        return {};
    }

    // ========================================
    // Filter 4: Spatial diversity (OPTIONAL)
    // ========================================
    std::vector<size_t> final_indices = coherence_filtered;

    if (use_spatial_diversity && static_cast<int>(coherence_filtered.size()) > target_diverse_samples) {
        RCLCPP_INFO(logger_,
            "\n[Filter 4/4] Selecting %d spatially diverse samples:",
            target_diverse_samples);
        RCLCPP_INFO(logger_,
            "  Strategy: Maximize distance between consecutive samples");
        RCLCPP_INFO(logger_,
            "  Min translation: %.1fmm, Min rotation: %.1f°\n",
            min_trans_dist_mm, min_rot_dist_deg);

        std::vector<size_t> selected;
        std::vector<bool> used(samples_.size(), false);

        // Start with first sample (arbitrary)
        selected.push_back(coherence_filtered[0]);
        used[coherence_filtered[0]] = true;

        // Greedy farthest-point selection
        while (static_cast<int>(selected.size()) < target_diverse_samples && selected.size() < coherence_filtered.size()) {
            size_t last_idx = selected.back();
            const auto& last_pose = samples_[last_idx].sensor_pose;

            double best_dist = -1.0;
            size_t best_candidate = coherence_filtered[0];

            for (size_t candidate_idx : coherence_filtered) {
                if (used[candidate_idx]) continue;

                const auto& candidate_pose = samples_[candidate_idx].sensor_pose;

                // Translation distance
                double trans_dist_mm = calculateTranslationDistance(
                    candidate_pose, last_pose) * 1000.0;

                // Rotation distance
                double rot_dist_deg = calculateRotationAngle(
                    candidate_pose, last_pose) * 180.0 / M_PI;

                // Combined distance (normalize and weight equally)
                double combined_dist = trans_dist_mm / min_trans_dist_mm +
                                     rot_dist_deg / min_rot_dist_deg;

                if (combined_dist > best_dist) {
                    best_dist = combined_dist;
                    best_candidate = candidate_idx;
                }
            }

            if (!used[best_candidate]) {
                selected.push_back(best_candidate);
                used[best_candidate] = true;

                double best_trans = calculateTranslationDistance(
                    samples_[best_candidate].sensor_pose,
                    samples_[last_idx].sensor_pose) * 1000.0;
                double best_rot = calculateRotationAngle(
                    samples_[best_candidate].sensor_pose,
                    samples_[last_idx].sensor_pose) * 180.0 / M_PI;

                RCLCPP_INFO(logger_,
                    "  [%2zu] Sample %3d (Δ from %3d: trans=%5.1fmm, rot=%5.1f°)",
                    selected.size(),
                    samples_[best_candidate].sample_id,
                    samples_[last_idx].sample_id,
                    best_trans, best_rot);
            }
        }

        final_indices = selected;
        RCLCPP_INFO(logger_,
            "\n✓ Selected %zu spatially diverse samples\n",
            final_indices.size());
    } else if (use_spatial_diversity) {
        RCLCPP_INFO(logger_,
            "[Filter 4/4] Spatial diversity: %zu samples (no filtering needed)\n",
            coherence_filtered.size());
    }

    RCLCPP_INFO(logger_,
               "✓ Filters completed: %zu samples selected\n",
               final_indices.size());

    return final_indices;
}

} // namespace eye_in_hand_calibration