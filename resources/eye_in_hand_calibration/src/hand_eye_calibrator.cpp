#include "eye_in_hand_calibration/hand_eye_calibrator.hpp"
#include <cv_bridge/cv_bridge.h>
#include <filesystem>
#include <ctime>
#include <iomanip>
#include <sstream>
namespace eye_in_hand_calibration {
HandEyeCalibrator::HandEyeCalibrator() : Node("hand_eye_calibrator") {
    RCLCPP_INFO(this->get_logger(), "Initializing Hand-Eye Calibrator...");
    collection_complete_ = false;
    calibration_started_ = false;
    total_images_processed_ = 0;
    successful_detections_ = 0;
    samples_saved_ = 0;
    images_dropped_ = 0;
    should_stop_ = false;
    final_transformation_ = Eigen::Matrix4d::Identity();
    config_ = CalibrationConfig::loadFromNode(this);
    if (config_.use_measured_object_points) {
        detector_ = std::make_unique<ChessboardDetector>(
            config_.chessboard_size,
            config_.measured_points_file,
            this->get_logger()
        );
    } else {
        detector_ = std::make_unique<ChessboardDetector>(
            config_.chessboard_size,
            config_.square_size,
            this->get_logger()
        );
    }
    pose_estimator_ = std::make_unique<PoseEstimator>(
        config_.camera_matrix,
        config_.dist_coeffs,
        config_.is_fisheye,
        this->get_logger()
    );
    aurora_sync_ = std::make_unique<AuroraSynchronizer>(
        config_.aurora_buffer_size,
        config_.max_pose_age_ms,
        5.0,  // 5mm max RMS error
        this->get_logger()
    );
    sample_manager_ = std::make_unique<SampleManager>(
        config_.min_movement_threshold,
        config_.min_rotation_threshold,
        10.0,  // 10.0px max reprojection error
        config_.stillness_buffer_size,
        config_.max_stillness_translation,
        config_.max_stillness_rotation,
        config_.max_sensor_stillness_translation,
        config_.max_sensor_stillness_rotation,
        this->get_logger()
    );
    solver_ = std::make_unique<CalibrationSolver>(
        static_cast<CalibrationSolver::Method>(config_.calibration_method),
        this->get_logger()
    );
    auto image_cb_group = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    auto aurora_cb_group = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions image_opts;
    image_opts.callback_group = image_cb_group;
    rclcpp::SubscriptionOptions aurora_opts;
    aurora_opts.callback_group = aurora_cb_group;
    image_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
        config_.image_topic,
        rclcpp::SensorDataQoS().keep_last(10),
        std::bind(&HandEyeCalibrator::imageCallback, this, std::placeholders::_1),
        image_opts);
    aurora_sub_ = this->create_subscription<aurora_ndi_ros2_driver::msg::AuroraData>(
        config_.aurora_topic,
        rclcpp::SensorDataQoS().keep_last(100),
        std::bind(&HandEyeCalibrator::auroraCallback, this, std::placeholders::_1),
        aurora_opts);
    num_processing_threads_ = 4;
    for (int i = 0; i < num_processing_threads_; ++i) {
        processing_threads_.emplace_back(
            &HandEyeCalibrator::processingThreadFunction, this);
    }
    RCLCPP_INFO(this->get_logger(), "=================================================");
    RCLCPP_INFO(this->get_logger(), "Hand-Eye Calibrator - COLLECTION MODE");
    RCLCPP_INFO(this->get_logger(), "=================================================");
    RCLCPP_INFO(this->get_logger(), "Target: Collect %d samples", config_.max_samples);
    RCLCPP_INFO(this->get_logger(), "Final calibration will use best %d samples", 
                config_.final_poses);
    RCLCPP_INFO(this->get_logger(), "Processing threads: %d", num_processing_threads_);
    RCLCPP_INFO(this->get_logger(), "=================================================");
}
HandEyeCalibrator::~HandEyeCalibrator() {
    RCLCPP_INFO(this->get_logger(), "Shutting down Hand-Eye Calibrator...");
    should_stop_ = true;
    queue_cv_.notify_all();
    for (auto& thread : processing_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    RCLCPP_INFO(this->get_logger(), "\n========== FINAL STATISTICS ==========");
    RCLCPP_INFO(this->get_logger(), "Total images received: %d", 
                total_images_processed_.load());
    RCLCPP_INFO(this->get_logger(), "Images dropped: %d", images_dropped_.load());
    RCLCPP_INFO(this->get_logger(), "Successful detections: %d", 
                successful_detections_.load());
    RCLCPP_INFO(this->get_logger(), "Samples saved: %d", samples_saved_.load());
    auto aurora_stats = aurora_sync_->getStatistics();
    RCLCPP_INFO(this->get_logger(), "\n========== AURORA SYNC STATISTICS ==========");
    RCLCPP_INFO(this->get_logger(), "Total queries: %zu", aurora_stats.total_queries);
    RCLCPP_INFO(this->get_logger(), "Successful syncs: %zu (%.1f%%)",
                aurora_stats.successful_syncs,
                100.0 * aurora_stats.successful_syncs / 
                std::max<size_t>(1, aurora_stats.total_queries));
    RCLCPP_INFO(this->get_logger(), "Failed - visibility: %zu", 
                aurora_stats.failed_visibility);
    RCLCPP_INFO(this->get_logger(), "Failed - quality: %zu", 
                aurora_stats.failed_quality);
    RCLCPP_INFO(this->get_logger(), "Failed - timeout: %zu", 
                aurora_stats.failed_timeout);
    RCLCPP_INFO(this->get_logger(), "Avg time diff: %.2fms", 
                aurora_stats.avg_time_diff_ms);
    auto sample_stats = sample_manager_->getStatistics();
    RCLCPP_INFO(this->get_logger(), "\n========== SAMPLE STATISTICS ==========");
    RCLCPP_INFO(this->get_logger(), "Total samples: %zu", sample_stats.total_samples);
    RCLCPP_INFO(this->get_logger(), "High quality: %zu", 
                sample_stats.high_quality_samples);
    RCLCPP_INFO(this->get_logger(), "Avg reprojection error: %.2fpx",
                sample_stats.avg_reprojection_error);
    RCLCPP_INFO(this->get_logger(), "Spatial coverage: %.3fm translation, %.2frad rotation",
                sample_stats.translation_range, sample_stats.rotation_range);
    RCLCPP_INFO(this->get_logger(), "======================================");
}
void HandEyeCalibrator::imageCallback(
    const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
    if (collection_complete_) {
        return;
    }
    total_images_processed_++;
    int count = total_images_processed_.load();
    if (count == 10 || count % 1000 == 0) {
        RCLCPP_INFO(this->get_logger(),
            "📸 Received %d images (detections: %d, samples: %d)",
            count, successful_detections_.load(), samples_saved_.load());
    }
    if (total_images_processed_ % 10 != 0) {
        return;
    }
    ImageProcessingTask task;
    task.image_msg = msg;
    task.image_timestamp = rclcpp::Time(msg->header.stamp);
    {
        std::unique_lock<std::mutex> lock(queue_mutex_, std::try_to_lock);
        if (lock.owns_lock()) {
            const size_t MAX_QUEUE_SIZE = 20;
            if (processing_queue_.size() < MAX_QUEUE_SIZE) {
                processing_queue_.push(task);
                queue_cv_.notify_one();
            } else {
                images_dropped_++;
            }
        } else {
            images_dropped_++;
        }
    }
}
void HandEyeCalibrator::auroraCallback(
    const aurora_ndi_ros2_driver::msg::AuroraData::SharedPtr msg) {
    aurora_sync_->addMeasurement(msg);
    // Log periodically
    static std::atomic<int> aurora_count{0};
    int count = ++aurora_count;
    if (count == 10 || count % 1000 == 0) {
        RCLCPP_INFO(this->get_logger(),
            "🔭 Received %d Aurora messages (buffer: %zu)",
            count, aurora_sync_->getBufferSize());
    }
}
void HandEyeCalibrator::processingThreadFunction() {
    while (!should_stop_) {
        ImageProcessingTask task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !processing_queue_.empty() || should_stop_;
            });
            if (should_stop_) {
                break;
            }
            if (!processing_queue_.empty()) {
                task = processing_queue_.front();
                processing_queue_.pop();
            } else {
                continue;
            }
        }
        processImage(task);
        if (samples_saved_ >= config_.max_samples) {
            bool expected = false;
            if (calibration_started_.compare_exchange_strong(expected, true)) {
                collection_complete_ = true;
                RCLCPP_INFO(this->get_logger(),
                    "\n🎉 COLLECTION COMPLETE! Collected %d samples",
                    samples_saved_.load());
                RCLCPP_INFO(this->get_logger(), "Starting selection and calibration...");
                selectBestSamplesAndCalibrate();
            }
            break;
        }
    }
}
void HandEyeCalibrator::processImage(const ImageProcessingTask& task) {
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(
            task.image_msg,
            sensor_msgs::image_encodings::BGR8);
        cv::Mat image = cv_ptr->image;
        std::vector<cv::Point2f> corners;
        if (!detector_->detectPattern(image, corners)) {
            return;
        }
        successful_detections_++;
        double quality = detector_->calculateCornerQuality(corners);
        RCLCPP_INFO(this->get_logger(),
            "✅ Chessboard detected! (detection #%d, quality: %.2f)",
            successful_detections_.load(), quality);
        cv::Mat rvec, tvec;
        if (!pose_estimator_->estimatePose(corners, detector_->getObjectPoints(),
                                          rvec, tvec)) {
            RCLCPP_WARN(this->get_logger(), "Pose estimation failed");
            return;
        }
        double reproj_error = pose_estimator_->computeReprojectionError(
            corners, detector_->getObjectPoints(), rvec, tvec);
        Eigen::Matrix4d T_camera_to_aurora = pose_estimator_->poseToMatrix(rvec, tvec);
        // We need the camera position in Aurora frame for hand-eye calibration
        Eigen::Matrix4d T_aurora_to_camera = T_camera_to_aurora.inverse();
        // Extract camera position in Aurora frame for logging
        Eigen::Vector3d camera_pos_aurora = T_aurora_to_camera.block<3,1>(0,3);
        double tx_mm = camera_pos_aurora.x() * 1000.0;
        double ty_mm = camera_pos_aurora.y() * 1000.0;
        double tz_mm = camera_pos_aurora.z() * 1000.0;
        double rotation_angle_deg = cv::norm(rvec) * 180.0 / M_PI;
        // Calculate distance to board using real board position in Aurora frame
        cv::Point3f board_center = detector_->getBoardCenter();
        Eigen::Vector3d board_pos_aurora(board_center.x, board_center.y, board_center.z);
        double distance_to_target = (camera_pos_aurora - board_pos_aurora).norm();
        RCLCPP_INFO(this->get_logger(),
            "📍 Camera pos (Aurora frame): [%.1f, %.1f, %.1f]mm | dist to board: %.1fmm",
            tx_mm, ty_mm, tz_mm, distance_to_target * 1000.0);
        RCLCPP_INFO(this->get_logger(), 
            "🎯 Reprojection error: %.3fpx", reproj_error);
        // GET SYNCHRONIZED AURORA POSE
        auto sensor_pose_opt = aurora_sync_->getPoseAt(task.image_timestamp);
        if (!sensor_pose_opt) {
            RCLCPP_WARN(this->get_logger(),
                "⚠️ No synchronized Aurora pose available");
            return;
        }
        Eigen::Matrix4d T_aurora_to_sensor = *sensor_pose_opt;
        Eigen::Vector3d sensor_pos_aurora = T_aurora_to_sensor.block<3,1>(0,3);
        RCLCPP_INFO(this->get_logger(),
            "📍 Sensor pos (Aurora frame): [%.1f, %.1f, %.1f]mm",
            sensor_pos_aurora.x() * 1000.0,
            sensor_pos_aurora.y() * 1000.0,
            sensor_pos_aurora.z() * 1000.0);
        // Calculate distance between sensor and camera (should be ~5cm for endoscope)
        double dist_sensor_camera = (camera_pos_aurora - sensor_pos_aurora).norm();
        RCLCPP_INFO(this->get_logger(),
            "📏 Distance sensor↔camera: %.1fmm %s",
            dist_sensor_camera * 1000.0,
            (dist_sensor_camera > 0.15) ? "⚠️ (seems large!)" : "");
        // SAVE SAMPLE WITH CORRECTED POSES
        // Both poses are now in Aurora frame:
        // - T_aurora_to_sensor: sensor pose in Aurora frame
        // - T_aurora_to_camera: camera pose in Aurora frame (inverted from PnP result)
        // // ========================================
        // // CRITICAL DEBUG: Log what we're saving
        // // ========================================
        // RCLCPP_ERROR(this->get_logger(),
        //     "🔵 DEBUG SAVE: Camera pose elements[0-3]: [%.4f, %.4f, %.4f, %.4f]",
        //     T_aurora_to_camera(0,0), T_aurora_to_camera(0,1), 
        //     T_aurora_to_camera(0,2), T_aurora_to_camera(0,3));
        // RCLCPP_ERROR(this->get_logger(),
        //     "🔵 DEBUG SAVE: Camera pos from matrix: [%.4f, %.4f, %.4f]",
        //     T_aurora_to_camera(0,3), T_aurora_to_camera(1,3), T_aurora_to_camera(2,3));

        // Filter by reprojection error before saving
        if (reproj_error > config_.max_reproj_error_filter) {
            RCLCPP_WARN(this->get_logger(),
                "❌ Sample discarded: reprojection error %.3fpx > %.3fpx threshold",
                reproj_error, config_.max_reproj_error_filter);
            return;
        }

        // Filter by sensor-camera distance
        if (dist_sensor_camera < config_.min_sensor_camera_distance) {
            RCLCPP_WARN(this->get_logger(),
                "❌ Sample discarded: sensor↔camera distance %.1fmm < %.1fmm threshold",
                dist_sensor_camera * 1000.0, config_.min_sensor_camera_distance * 1000.0);
            return;
        }
        if (dist_sensor_camera > config_.max_sensor_camera_distance) {
            RCLCPP_WARN(this->get_logger(),
                "❌ Sample discarded: sensor↔camera distance %.1fmm > %.1fmm threshold",
                dist_sensor_camera * 1000.0, config_.max_sensor_camera_distance * 1000.0);
            return;
        }

        // Check stillness if enabled (always update buffers, even if we don't save)
        Eigen::Matrix4d T_camera_to_use = T_aurora_to_camera;  // Default: use current pose
        Eigen::Matrix4d T_sensor_to_use = T_aurora_to_sensor;  // Default: use current pose

        if (config_.enable_stillness_check) {
            bool is_camera_still = sample_manager_->isCameraStill(T_aurora_to_camera);
            bool is_sensor_still = sample_manager_->isSensorStill(T_aurora_to_sensor);

            if (!is_camera_still) {
                RCLCPP_DEBUG(this->get_logger(),
                    "⏸️  Camera is moving, waiting for stillness...");
                return;
            }

            if (!is_sensor_still) {
                RCLCPP_DEBUG(this->get_logger(),
                    "⏸️  Sensor is moving, waiting for stillness...");
                return;
            }

            // Both camera and sensor are still - use averaged poses from buffers (with outlier rejection)
            T_camera_to_use = sample_manager_->getAveragedCameraPose();
            T_sensor_to_use = sample_manager_->getAveragedSensorPose();
            RCLCPP_DEBUG(this->get_logger(),
                "Using averaged camera and sensor poses from %d frames",
                static_cast<int>(config_.stillness_buffer_size));
        }

        // Check if sample is diverse enough from last saved sample
        if (sample_manager_->shouldSaveSample(T_sensor_to_use, T_camera_to_use)) {
            int sample_id = sample_manager_->addSample(
                T_sensor_to_use,        // Sensor pose (averaged if stillness enabled)
                T_camera_to_use,        // Camera pose (averaged if stillness enabled)
                reproj_error,
                distance_to_target
            );
            samples_saved_++;
            RCLCPP_INFO(this->get_logger(),
                "✓ Sample %d saved (total: %zu, reproj: %.3fpx, dist S↔C: %.1fmm)",
                sample_id, sample_manager_->getNumSamples(),
                reproj_error, dist_sensor_camera * 1000.0);
        } else {
            RCLCPP_DEBUG(this->get_logger(),
                "Sample skipped (too similar to previous)");
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error processing image: %s", e.what());
    }
}
void HandEyeCalibrator::selectBestSamplesAndCalibrate() {
    // Create output directory
    std::string output_dir = "/workspace/src/eye_in_hand_calibration/output";
    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }
    // Generate timestamp for filenames
    auto now = std::time(nullptr);
    auto local_time = std::localtime(&now);
    std::stringstream timestamp;
    timestamp << std::put_time(local_time, "%Y%m%d_%H%M%S");
    // Filters in order:
    // 1. Reprojection error < 0.8px
    // 2. Sensor-camera distance 8.0mm - 15.0mm
    // 3. Movement coherence (ratio<1.2, Δrot<15.0°) - ALL vs REFERENCE
    // 4. Spatial diversity (optional, if enabled)
    auto selected = sample_manager_->selectSamplesAdvanced(
        config_.max_reproj_error_filter,
        config_.min_sensor_camera_distance,
        config_.max_sensor_camera_distance,
        config_.max_movement_ratio,
        config_.max_rotation_diff_deg,
        config_.use_spatial_diversity,
        config_.min_trans_dist_mm,
        config_.min_rot_dist_deg,
        config_.target_diverse_samples
    );
    if (selected.empty()) {
        RCLCPP_ERROR(this->get_logger(), "No samples passed advanced filtering!");
        return;
    }
    // Save all collected samples
    std::string all_samples_file = output_dir + "/collected_samples_" +
                                   timestamp.str() + ".yaml";
    sample_manager_->saveAllSamples(all_samples_file);
    // Perform final calibration
    RCLCPP_INFO(this->get_logger(),
        "\n========== FINAL CALIBRATION ==========");
    RCLCPP_INFO(this->get_logger(), "Using %zu samples (%zu pairs)",
               selected.size(), selected.size() - 1);
    auto result = solver_->solve(sample_manager_->getSamples(), selected);
    if (!result.success) {
        RCLCPP_ERROR(this->get_logger(), "❌ Calibration failed!");
        return;
    }
    RCLCPP_INFO(this->get_logger(), "✅ Calibration successful!");
    // Nonlinear refinement (Bundle Adjustment) - MATCHING PYTHON SCRIPT
    if (config_.use_nonlinear_refinement) {
        RCLCPP_INFO(this->get_logger(),
            "\n========== NONLINEAR REFINEMENT (Bundle Adjustment) ==========");
        RCLCPP_INFO(this->get_logger(), "Rotation weight: %.1f", config_.rotation_weight);
        RCLCPP_INFO(this->get_logger(), "Max iterations: %d\n", config_.refinement_max_iterations);
        Eigen::Matrix4d X_initial = result.transformation;
        result.transformation = solver_->refineHandEyeNonlinear(
            sample_manager_->getSamples(),
            selected,
            X_initial,
            config_.refinement_max_iterations,
            config_.rotation_weight
        );
        RCLCPP_INFO(this->get_logger(), "✅ Nonlinear refinement completed!");
    }
    // Compute and print absolute errors (matching Python script)
    solver_->computeAndPrintAbsoluteErrors(
        sample_manager_->getSamples(),
        selected,
        result.transformation
    );
    // Save calibration result
    if (config_.save_result) {
        std::string result_file = output_dir + "/eye_in_hand_calibration_" +
                                 timestamp.str() + ".yaml";
        solver_->saveResult(
            result_file,
            result,
            config_.result_frame_id,
            config_.target_frame_id,
            total_images_processed_.load(),
            successful_detections_.load()
        );
    }
    // Save selected pose pairs for RViz visualization
    std::string poses_file = output_dir + "/selected_pose_pairs_" +
                            timestamp.str() + ".yaml";
    sample_manager_->savePosePairs(poses_file, selected);
    // Store final transformation
    final_transformation_ = result.transformation;
}
} // namespace eye_in_hand_calibration