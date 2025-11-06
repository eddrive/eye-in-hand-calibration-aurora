#ifndef EYE_IN_HAND_CALIBRATION_SAMPLE_MANAGER_HPP
#define EYE_IN_HAND_CALIBRATION_SAMPLE_MANAGER_HPP

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <mutex>
#include <rclcpp/rclcpp.hpp>

namespace eye_in_hand_calibration {

/**
 * @brief Calibration sample containing synchronized poses
 */
struct CalibrationSample {
    Eigen::Matrix4d sensor_pose;      // Aurora sensor pose (4x4 transform)
    Eigen::Matrix4d camera_pose;      // Camera-to-target pose (4x4 transform)
    double reprojection_error;        // Reprojection error in pixels
    double distance_to_target;        // Distance to chessboard (meters)
    int sample_id;                    // Unique sample identifier
    
    CalibrationSample()
        : sensor_pose(Eigen::Matrix4d::Identity()),
          camera_pose(Eigen::Matrix4d::Identity()),
          reprojection_error(0.0),
          distance_to_target(0.0),
          sample_id(-1) {}
};

/**
 * @brief Manages collection, filtering, and selection of calibration samples
 * 
 * This class handles the storage of calibration samples with diversity
 * filtering to ensure good spatial coverage and quality-based selection
 * for final calibration.
 */
class SampleManager {
public:
    /**
     * @brief Constructor
     * @param min_movement_threshold Minimum translation between samples (meters)
     * @param min_rotation_threshold Minimum rotation between samples (radians)
     * @param max_reprojection_error Maximum acceptable reprojection error (pixels)
     * @param logger ROS2 logger for diagnostics
     */
    SampleManager(double min_movement_threshold,
                  double min_rotation_threshold,
                  double max_reprojection_error,
                  rclcpp::Logger logger);
    
    /**
     * @brief Check if a new sample should be saved
     * @param sensor_pose New sensor pose
     * @param camera_pose New camera pose
     * @return true if sample is sufficiently different from existing samples
     */
    bool shouldSaveSample(const Eigen::Matrix4d& sensor_pose,
                          const Eigen::Matrix4d& camera_pose);
    
    /**
     * @brief Add a calibration sample
     * @param sensor_pose Sensor transformation
     * @param camera_pose Camera transformation
     * @param reprojection_error Reprojection error in pixels
     * @param distance_to_target Distance to chessboard in meters
     * @return Sample ID
     */
    int addSample(const Eigen::Matrix4d& sensor_pose,
                  const Eigen::Matrix4d& camera_pose,
                  double reprojection_error,
                  double distance_to_target);
    
    /**
     * @brief Select diverse samples for calibration (old greedy method)
     * @param num_samples Number of samples to select
     * @return Indices of selected samples
     */
    std::vector<size_t> selectDiverseSamples(int num_samples);

    /**
     * @brief Select samples using advanced filtering (Python script logic)
     * @param max_reproj_error Maximum reprojection error (pixels)
     * @param max_sensor_camera_dist Maximum sensor-camera distance (meters)
     * @param max_movement_ratio Maximum ratio between sensor/camera movement
     * @param max_rotation_diff Maximum rotation difference (degrees)
     * @return Indices of selected samples (in order)
     */
    std::vector<size_t> selectSamplesAdvanced(double max_reproj_error,
                                               double max_sensor_camera_dist,
                                               double max_movement_ratio,
                                               double max_rotation_diff);
    
    /**
     * @brief Get all collected samples
     */
    const std::vector<CalibrationSample>& getSamples() const;
    
    /**
     * @brief Get number of collected samples
     */
    size_t getNumSamples() const;
    
    /**
     * @brief Clear all samples
     */
    void clearSamples();
    
    /**
     * @brief Save all samples to YAML file
     * @param filename Output file path
     * @return true if successful
     */
    bool saveAllSamples(const std::string& filename);
    
    /**
     * @brief Save selected pose pairs for visualization
     * @param filename Output file path
     * @param selected_indices Indices of samples to save
     * @return true if successful
     */
    bool savePosePairs(const std::string& filename,
                       const std::vector<size_t>& selected_indices);
    
    /**
     * @brief Get statistics about collected samples
     */
    struct SampleStats {
        size_t total_samples;
        size_t high_quality_samples;    // reprojection_error < 2.0px
        double avg_reprojection_error;
        double min_reprojection_error;
        double max_reprojection_error;
        double translation_range;       // Max distance between any two samples
        double rotation_range;          // Max rotation between any two samples
    };
    
    SampleStats getStatistics() const;
    
private:
    /**
     * @brief Calculate translation distance between two poses
     */
    double calculateTranslationDistance(const Eigen::Matrix4d& pose1,
                                        const Eigen::Matrix4d& pose2) const;
    
    /**
     * @brief Calculate rotation angle between two poses
     */
    double calculateRotationAngle(const Eigen::Matrix4d& pose1,
                                  const Eigen::Matrix4d& pose2) const;
    
    /**
     * @brief Calculate diversity score for a candidate sample
     * @param candidate_idx Index of candidate sample
     * @param selected_indices Already selected sample indices
     * @return Diversity score (higher is more diverse)
     */
    double calculateDiversityScore(size_t candidate_idx,
                                   const std::vector<size_t>& selected_indices) const;
    
    /**
     * @brief Filter samples by reprojection error quality
     * @return Indices of valid samples
     */
    std::vector<size_t> filterByQuality() const;
    
    /**
     * @brief Write sample to YAML emitter
     */
    void writeSampleToYAML(void* emitter, const CalibrationSample& sample) const;
    
    /**
     * @brief Write pose pair to YAML for RViz visualization
     */
    void writePosePairToYAML(void* emitter, const CalibrationSample& sample) const;
    
    // Sample storage
    std::vector<CalibrationSample> samples_;
    mutable std::mutex samples_mutex_;
    int next_sample_id_;
    
    // Diversity filtering parameters
    double min_movement_threshold_;    // meters
    double min_rotation_threshold_;    // radians
    
    // Quality filtering parameters
    double max_reprojection_error_;    // pixels
    
    // Logger
    rclcpp::Logger logger_;
};

} // namespace eye_in_hand_calibration

#endif // EYE_IN_HAND_CALIBRATION_SAMPLE_MANAGER_HPP