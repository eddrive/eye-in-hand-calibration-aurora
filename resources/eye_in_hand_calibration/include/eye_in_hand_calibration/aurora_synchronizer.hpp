#ifndef EYE_IN_HAND_CALIBRATION_AURORA_SYNCHRONIZER_HPP
#define EYE_IN_HAND_CALIBRATION_AURORA_SYNCHRONIZER_HPP

#include <aurora_ndi_ros2_driver/msg/aurora_data.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <deque>
#include <mutex>
#include <optional>

namespace eye_in_hand_calibration {

/**
 * @brief Cached Aurora data with timestamp
 */
struct AuroraDataCached {
    aurora_ndi_ros2_driver::msg::AuroraData data;
    rclcpp::Time timestamp;
};

/**
 * @brief Synchronizes Aurora tracking data with image timestamps
 * 
 * This class maintains a buffer of Aurora measurements and provides
 * time-synchronized pose queries with configurable tolerance and
 * coordinate frame transformations.
 */
class AuroraSynchronizer {
public:
    /**
     * @brief Constructor
     * @param max_buffer_size Maximum number of Aurora samples to keep
     * @param max_time_diff_ms Maximum time difference for sync (milliseconds)
     * @param max_rms_error_mm Maximum RMS error threshold (millimeters)
     * @param logger ROS2 logger for diagnostics
     */
    AuroraSynchronizer(size_t max_buffer_size,
                       int max_time_diff_ms,
                       double max_rms_error_mm,
                       rclcpp::Logger logger);
    
    /**
     * @brief Add Aurora measurement to buffer
     * @param data Aurora data message
     */
    void addMeasurement(const aurora_ndi_ros2_driver::msg::AuroraData::SharedPtr data);
    
    /**
     * @brief Get synchronized Aurora pose for given timestamp
     * @param target_time Target timestamp to synchronize with
     * @return 4x4 transformation matrix if sync successful, nullopt otherwise
     */
    std::optional<Eigen::Matrix4d> getPoseAt(const rclcpp::Time& target_time);
    
    /**
     * @brief Get the closest Aurora data to target time
     * @param target_time Target timestamp
     * @return Cached Aurora data if found within tolerance
     */
    std::optional<AuroraDataCached> getClosestData(const rclcpp::Time& target_time);
    
    /**
     * @brief Get current buffer size
     */
    size_t getBufferSize() const;
    
    /**
     * @brief Clear all buffered data
     */
    void clearBuffer();
    
    /**
     * @brief Get statistics about synchronization performance
     */
    struct SyncStats {
        size_t total_queries;
        size_t successful_syncs;
        size_t failed_visibility;
        size_t failed_quality;
        size_t failed_timeout;
        double avg_time_diff_ms;
        double max_time_diff_ms;
    };
    
    SyncStats getStatistics() const;
    
    /**
     * @brief Reset statistics counters
     */
    void resetStatistics();
    
private:
    /**
     * @brief Validate Aurora data quality
     * @param data Aurora measurement
     * @return true if data meets quality criteria
     */
    bool validateQuality(const aurora_ndi_ros2_driver::msg::AuroraData& data) const;
    
    /**
     * @brief Convert Aurora data to transformation matrix
     * @param data Aurora measurement
     * @return 4x4 transformation matrix in calibrated frame
     */
    Eigen::Matrix4d dataToTransform(const aurora_ndi_ros2_driver::msg::AuroraData& data) const;
    
    // /**
    //  * @brief Transform from Aurora NED frame to standard frame
    //  * 
    //  * Aurora NED: X=Down, Y=Right, Z=Forward
    //  * Standard:   X=Down, Y=Left, Z=Backward
    //  * 
    //  * @param ned_transform Transform in NED frame
    //  * @return Transform in standard frame
    //  */
    // Eigen::Matrix4d nedToStandard(const Eigen::Matrix4d& ned_transform) const;
    
    // Buffer management
    std::deque<AuroraDataCached> buffer_;
    mutable std::mutex buffer_mutex_;
    size_t max_buffer_size_;
    
    // Synchronization parameters
    int max_time_diff_ms_;        // Maximum acceptable time difference
    double max_rms_error_mm_;     // Maximum RMS tracking error
    
    // Statistics tracking
    mutable std::mutex stats_mutex_;
    SyncStats stats_;
    
    // // Frame transformation
    // Eigen::Matrix4d T_ned_to_std_;  // NED to Standard frame transform
    
    // Logger
    rclcpp::Logger logger_;
};

} // namespace eye_in_hand_calibration

#endif // EYE_IN_HAND_CALIBRATION_AURORA_SYNCHRONIZER_HPP