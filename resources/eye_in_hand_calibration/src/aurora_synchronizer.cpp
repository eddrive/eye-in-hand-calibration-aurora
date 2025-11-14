#include "eye_in_hand_calibration/aurora_synchronizer.hpp"
#include <limits>
#include <cmath>

namespace eye_in_hand_calibration {

AuroraSynchronizer::AuroraSynchronizer(size_t max_buffer_size,
                                       int max_time_diff_ms,
                                       double max_rms_error_mm,
                                       rclcpp::Logger logger)
    : max_buffer_size_(max_buffer_size),
      max_time_diff_ms_(max_time_diff_ms),
      max_rms_error_mm_(max_rms_error_mm),
      logger_(logger)
{
    stats_.total_queries = 0;
    stats_.successful_syncs = 0;
    stats_.failed_visibility = 0;
    stats_.failed_quality = 0;
    stats_.failed_timeout = 0;
    stats_.avg_time_diff_ms = 0.0;
    stats_.max_time_diff_ms = 0.0;

    RCLCPP_INFO(logger_,
                "AuroraSynchronizer initialized: buffer=%zu, tolerance=%dms, max_error=%.1fmm",
                max_buffer_size_, max_time_diff_ms_, max_rms_error_mm_);
}

void AuroraSynchronizer::addMeasurement(const aurora_ndi_ros2_driver::msg::AuroraData::SharedPtr data) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);

    AuroraDataCached cached;
    cached.data = *data;
    cached.timestamp = rclcpp::Time(data->header.stamp);

    buffer_.push_back(cached);

    while (buffer_.size() > max_buffer_size_) {
        buffer_.pop_front();
    }
}

std::optional<AuroraDataCached> AuroraSynchronizer::getClosestData(
    const rclcpp::Time& target_time) {

    std::lock_guard<std::mutex> lock(buffer_mutex_);

    if (buffer_.empty()) {
        RCLCPP_WARN_THROTTLE(logger_, *rclcpp::Clock::make_shared(), 1000,
                            "Aurora buffer is empty");
        return std::nullopt;
    }

    double min_time_diff = std::numeric_limits<double>::max();
    std::optional<AuroraDataCached> closest;

    for (const auto& cached : buffer_) {
        double time_diff = std::abs((target_time - cached.timestamp).seconds());
        if (time_diff < min_time_diff) {
            min_time_diff = time_diff;
            closest = cached;
        }
    }

    if (!closest.has_value()) {
        return std::nullopt;
    }

    double time_diff_ms = min_time_diff * 1000.0;

    RCLCPP_DEBUG(logger_,
                "Closest Aurora: Δt=%.1fms (threshold=%dms) | "
                "image_time=%.3f, aurora_time=%.3f, buffer_size=%zu",
                time_diff_ms, max_time_diff_ms_,
                target_time.seconds(), closest->timestamp.seconds(),
                buffer_.size());

    if (time_diff_ms > max_time_diff_ms_) {
        RCLCPP_WARN_THROTTLE(logger_, *rclcpp::Clock::make_shared(), 1000,
                            "Aurora time difference too large: %.1fms > %dms | "
                            "image_time=%.6f, aurora_time=%.6f",
                            time_diff_ms, max_time_diff_ms_,
                            target_time.seconds(), closest->timestamp.seconds());

        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.failed_timeout++;

        return std::nullopt;
    }

    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.avg_time_diff_ms = (stats_.avg_time_diff_ms * stats_.total_queries + time_diff_ms) /
                                  (stats_.total_queries + 1);
        stats_.max_time_diff_ms = std::max(stats_.max_time_diff_ms, time_diff_ms);
    }

    return closest;
}

std::optional<Eigen::Matrix4d> AuroraSynchronizer::getPoseAt(
    const rclcpp::Time& target_time) {

    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.total_queries++;
    }

    auto closest_opt = getClosestData(target_time);
    if (!closest_opt.has_value()) {
        return std::nullopt;
    }

    const auto& data = closest_opt->data;

    if (!data.visible) {
        RCLCPP_WARN_THROTTLE(logger_, *rclcpp::Clock::make_shared(), 1000,
                            "Aurora sensor not visible");

        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.failed_visibility++;

        return std::nullopt;
    }

    if (!validateQuality(data)) {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.failed_quality++;

        return std::nullopt;
    }


    Eigen::Matrix4d transform = dataToTransform(data);

    {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.successful_syncs++;
    }

    RCLCPP_DEBUG(logger_,
                "✅ Aurora synchronized: pos=[%.3f, %.3f, %.3f]m, error=%.2fmm",
                transform(0, 3), transform(1, 3), transform(2, 3),
                data.error);

    return transform;
}

bool AuroraSynchronizer::validateQuality(const aurora_ndi_ros2_driver::msg::AuroraData& data) const {
    if (data.error > max_rms_error_mm_) {
        RCLCPP_WARN_THROTTLE(logger_, *rclcpp::Clock::make_shared(), 1000,
                            "Aurora RMS error too high: %.2fmm > %.2fmm",
                            data.error, max_rms_error_mm_);
        return false;
    }

    if (!std::isfinite(data.position.x) ||
        !std::isfinite(data.position.y) ||
        !std::isfinite(data.position.z)) {
        RCLCPP_ERROR(logger_, "Aurora position contains NaN or Inf");
        return false;
    }

    if (!std::isfinite(data.orientation.w) ||
        !std::isfinite(data.orientation.x) ||
        !std::isfinite(data.orientation.y) ||
        !std::isfinite(data.orientation.z)) {
        RCLCPP_ERROR(logger_, "Aurora orientation contains NaN or Inf");
        return false;
    }

    Eigen::Quaterniond q(data.orientation.w,
                        data.orientation.x,
                        data.orientation.y,
                        data.orientation.z);

    double norm = q.norm();
    if (std::abs(norm - 1.0) > 0.01) {
        RCLCPP_WARN_THROTTLE(logger_, *rclcpp::Clock::make_shared(), 1000,
                            "Aurora quaternion not normalized: %.4f", norm);
        return false;
    }

    return true;
}

Eigen::Matrix4d AuroraSynchronizer::dataToTransform(
    const aurora_ndi_ros2_driver::msg::AuroraData& data) const {

    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();

    transform(0, 3) = data.position.x / 1000.0;
    transform(1, 3) = data.position.y / 1000.0;
    transform(2, 3) = data.position.z / 1000.0;

    Eigen::Quaterniond q(data.orientation.w,
                        data.orientation.x,
                        data.orientation.y,
                        data.orientation.z);

    q.normalize();

    transform.block<3, 3>(0, 0) = q.toRotationMatrix();

    return transform;
}


size_t AuroraSynchronizer::getBufferSize() const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    return buffer_.size();
}

void AuroraSynchronizer::clearBuffer() {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    buffer_.clear();
    RCLCPP_INFO(logger_, "Aurora buffer cleared");
}

AuroraSynchronizer::SyncStats AuroraSynchronizer::getStatistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void AuroraSynchronizer::resetStatistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    stats_.total_queries = 0;
    stats_.successful_syncs = 0;
    stats_.failed_visibility = 0;
    stats_.failed_quality = 0;
    stats_.failed_timeout = 0;
    stats_.avg_time_diff_ms = 0.0;
    stats_.max_time_diff_ms = 0.0;
    
    RCLCPP_INFO(logger_, "Aurora synchronizer statistics reset");
}

} // namespace eye_in_hand_calibration