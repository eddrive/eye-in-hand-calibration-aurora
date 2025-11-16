#include "eye_in_hand_calibration/hand_eye_calibrator.hpp"
#include <cv_bridge/cv_bridge.h>
#include <filesystem>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <numeric>
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>
namespace eye_in_hand_calibration {
HandEyeCalibrator::HandEyeCalibrator() : Node("hand_eye_calibrator") {
    RCLCPP_INFO(this->get_logger(), "Initializing Hand-Eye Calibrator (Interactive Mode)...");
    collection_complete_ = false;
    calibration_started_ = false;
    total_images_processed_ = 0;
    successful_detections_ = 0;
    samples_saved_ = 0;
    images_dropped_ = 0;
    should_stop_ = false;
    final_transformation_ = Eigen::Matrix4d::Identity();

    // Interactive mode initialization
    current_state_ = CalibrationState::IDLE;
    current_position_index_ = 0;
    chessboard_detected_ = false;
    last_corner_quality_ = 0.0;
    last_detection_time_ = this->now();
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

    // Start keyboard listener thread
    keyboard_thread_ = std::thread(&HandEyeCalibrator::keyboardListenerThread, this);

    RCLCPP_INFO(this->get_logger(), "=================================================");
    RCLCPP_INFO(this->get_logger(), "Hand-Eye Calibrator - INTERACTIVE MODE");
    RCLCPP_INFO(this->get_logger(), "=================================================");
    RCLCPP_INFO(this->get_logger(), "Positions to collect: %d", config_.num_positions);
    RCLCPP_INFO(this->get_logger(), "Samples per position: %d", config_.samples_per_position);
    RCLCPP_INFO(this->get_logger(), "Detection rate: %.1f Hz", config_.detection_rate_hz);
    RCLCPP_INFO(this->get_logger(), "Quality threshold: %.2f", config_.chessboard_quality_threshold);
    RCLCPP_INFO(this->get_logger(), "Processing threads: %d", num_processing_threads_);
    RCLCPP_INFO(this->get_logger(), "=================================================");
    RCLCPP_INFO(this->get_logger(), "Commands: [s] Start collecting | [p] Pause | [x] Stop");
    RCLCPP_INFO(this->get_logger(), "=================================================\n");

    // Transition to DETECTING state
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        current_state_ = CalibrationState::DETECTING;
    }

    RCLCPP_INFO(this->get_logger(), "==> STATE: DETECTING");
    RCLCPP_INFO(this->get_logger(), "Analyzing frames at %.1f Hz for chessboard detection...", config_.detection_rate_hz);
    RCLCPP_INFO(this->get_logger(), "Waiting for image topic: %s\n", config_.image_topic.c_str());
}
HandEyeCalibrator::~HandEyeCalibrator() {
    RCLCPP_INFO(this->get_logger(), "Shutting down Hand-Eye Calibrator...");
    should_stop_ = true;
    queue_cv_.notify_all();

    // Join processing threads
    for (auto& thread : processing_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Join keyboard thread
    if (keyboard_thread_.joinable()) {
        keyboard_thread_.join();
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

    // Log periodically to confirm images are arriving
    int count = total_images_processed_.load();
    if (count == 1 || count == 10 || count % 100 == 0) {
        RCLCPP_INFO(this->get_logger(), "📸 Received %d images", count);
    }

    CalibrationState state;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state = current_state_;
    }

    // Apply throttling based on state
    bool should_process = false;

    if (state == CalibrationState::DETECTING) {
        // Throttle to detection_rate_hz (e.g., 1 Hz)
        auto now = this->now();
        double dt = (now - last_detection_time_).seconds();
        double min_interval = 1.0 / config_.detection_rate_hz;

        if (dt >= min_interval) {
            should_process = true;
            last_detection_time_ = now;
        }
    } else if (state == CalibrationState::COLLECTING) {
        // Process all frames during collection
        should_process = true;
    }

    if (!should_process) {
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
    CalibrationState state;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state = current_state_;
    }

    if (state == CalibrationState::DETECTING) {
        processDetection(task);
    } else if (state == CalibrationState::COLLECTING) {
        processCollection(task);
    }
}

void HandEyeCalibrator::processDetection(const ImageProcessingTask& task) {
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(
            task.image_msg,
            sensor_msgs::image_encodings::BGR8);
        cv::Mat image = cv_ptr->image;

        std::vector<cv::Point2f> corners;
        if (!detector_->detectPattern(image, corners)) {
            // Chessboard not detected
            RCLCPP_INFO(this->get_logger(), "[DETECTING] Frame analyzed - Chessboard NOT detected");
            if (chessboard_detected_) {
                chessboard_detected_ = false;
                RCLCPP_WARN(this->get_logger(), "Chessboard lost");
            }
            return;
        }

        // Chessboard detected - calculate quality
        double quality = detector_->calculateCornerQuality(corners);
        last_corner_quality_ = quality;

        // Print quality with comparison to threshold
        if (quality >= config_.chessboard_quality_threshold) {
            RCLCPP_INFO(this->get_logger(),
                "[DETECTING] Frame analyzed - Chessboard DETECTED | Quality: %.3f [ABOVE threshold %.3f] ✓",
                quality, config_.chessboard_quality_threshold);
        } else {
            RCLCPP_INFO(this->get_logger(),
                "[DETECTING] Frame analyzed - Chessboard DETECTED | Quality: %.3f [BELOW threshold %.3f]",
                quality, config_.chessboard_quality_threshold);
        }

        if (quality >= config_.chessboard_quality_threshold) {
            if (!chessboard_detected_) {
                chessboard_detected_ = true;
                RCLCPP_INFO(this->get_logger(),
                    "\n*** Chessboard quality sufficient! Quality: %.3f >= %.3f ***",
                    quality, config_.chessboard_quality_threshold);
                RCLCPP_INFO(this->get_logger(), "Press 's' to start collecting samples at this position\n");
            }
        } else {
            if (chessboard_detected_) {
                chessboard_detected_ = false;
                RCLCPP_WARN(this->get_logger(),
                    "Chessboard quality dropped below threshold: %.3f < %.3f",
                    quality, config_.chessboard_quality_threshold);
            }
        }

    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error in detection: %s", e.what());
    }
}

void HandEyeCalibrator::processCollection(const ImageProcessingTask& task) {
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

        cv::Mat rvec, tvec;
        if (!pose_estimator_->estimatePose(corners, detector_->getObjectPoints(),
                                          rvec, tvec)) {
            RCLCPP_WARN(this->get_logger(), "Pose estimation failed");
            return;
        }

        double reproj_error = pose_estimator_->computeReprojectionError(
            corners, detector_->getObjectPoints(), rvec, tvec);
        double distance_to_target = cv::norm(tvec);

        Eigen::Matrix4d T_camera_to_aurora = pose_estimator_->poseToMatrix(rvec, tvec);
        Eigen::Matrix4d T_aurora_to_camera = T_camera_to_aurora.inverse();

        // GET SYNCHRONIZED AURORA POSE
        auto sensor_pose_opt = aurora_sync_->getPoseAt(task.image_timestamp);
        if (!sensor_pose_opt) {
            RCLCPP_WARN(this->get_logger(), "No synchronized Aurora pose available");
            return;
        }

        Eigen::Matrix4d T_aurora_to_sensor = *sensor_pose_opt;

        // Create sample
        CalibrationSample sample;
        sample.sensor_pose = T_aurora_to_sensor;
        sample.camera_pose = T_aurora_to_camera;
        sample.reprojection_error = reproj_error;
        sample.distance_to_target = distance_to_target;
        sample.sample_id = current_bucket_.samples.size();

        // Add to current bucket
        {
            std::lock_guard<std::mutex> lock(bucket_mutex_);
            current_bucket_.samples.push_back(sample);

            RCLCPP_INFO(this->get_logger(),
                "Sample %zu/%d collected (reproj: %.3fpx, quality: %.3f)",
                current_bucket_.samples.size(), config_.samples_per_position,
                reproj_error, quality);

            // Check if we have enough samples for this position
            if (static_cast<int>(current_bucket_.samples.size()) >= config_.samples_per_position) {
                RCLCPP_INFO(this->get_logger(), "\nPosition complete! Validating...");
                validateAndSavePosition();
            }
        }

    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error processing collection: %s", e.what());
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

    RCLCPP_INFO(this->get_logger(), "\n========== SELECTING BEST SAMPLES ==========");
    RCLCPP_INFO(this->get_logger(), "Position buckets collected: %zu", position_buckets_.size());

    // Select best sample from each position bucket (lowest reprojection error)
    std::vector<CalibrationSample> best_samples;

    for (const auto& bucket : position_buckets_) {
        if (bucket.samples.empty()) {
            continue;
        }

        // Find sample with lowest reprojection error
        auto best_it = std::min_element(bucket.samples.begin(), bucket.samples.end(),
            [](const CalibrationSample& a, const CalibrationSample& b) {
                return a.reprojection_error < b.reprojection_error;
            });

        best_samples.push_back(*best_it);

        RCLCPP_INFO(this->get_logger(),
            "Position %d: Best sample has reproj error %.3fpx (from %zu samples)",
            bucket.position_id + 1, best_it->reprojection_error, bucket.samples.size());
    }

    RCLCPP_INFO(this->get_logger(), "Total best samples selected: %zu", best_samples.size());

    if (best_samples.empty()) {
        RCLCPP_ERROR(this->get_logger(), "No samples available for calibration!");
        return;
    }

    // Add best samples to sample_manager for calibration
    sample_manager_->clearSamples();
    std::vector<size_t> selected_indices;

    for (size_t i = 0; i < best_samples.size(); ++i) {
        const auto& sample = best_samples[i];
        sample_manager_->addSample(
            sample.sensor_pose,
            sample.camera_pose,
            sample.reprojection_error,
            sample.distance_to_target
        );
        selected_indices.push_back(i);
    }

    RCLCPP_INFO(this->get_logger(), "\n========== APPLYING FILTERS ==========");

    // Apply advanced filters to the selected best samples
    auto filtered = sample_manager_->selectSamplesAdvanced(
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

    if (filtered.empty()) {
        RCLCPP_ERROR(this->get_logger(), "No samples passed advanced filtering!");
        return;
    }

    // ITERATIVE REFINEMENT (optional)
    if (config_.use_iterative_refinement &&
        static_cast<int>(filtered.size()) - 1 > config_.target_pairs) {
        RCLCPP_INFO(this->get_logger(),
            "Applying iterative refinement to reduce from %zu to %d pairs",
            filtered.size() - 1, config_.target_pairs);
        filtered = solver_->refineByError(
            sample_manager_->getSamples(),
            filtered,
            config_.target_pairs,
            config_.max_refinement_iterations
        );
    }

    if (filtered.empty()) {
        RCLCPP_ERROR(this->get_logger(), "No samples after refinement!");
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
               filtered.size(), filtered.size() - 1);
    auto result = solver_->solve(sample_manager_->getSamples(), filtered);
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
            filtered,
            X_initial,
            config_.refinement_max_iterations,
            config_.rotation_weight
        );
        RCLCPP_INFO(this->get_logger(), "✅ Nonlinear refinement completed!");
    }
    // Compute and print absolute errors (matching Python script)
    solver_->computeAndPrintAbsoluteErrors(
        sample_manager_->getSamples(),
        filtered,
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
    sample_manager_->savePosePairs(poses_file, filtered);

    // Store final transformation
    final_transformation_ = result.transformation;
}

// ========================================
// INTERACTIVE MODE FUNCTIONS
// ========================================

void HandEyeCalibrator::keyboardListenerThread() {
    RCLCPP_INFO(this->get_logger(), "Keyboard listener started");

    // Set terminal to raw mode for immediate key capture
    struct termios old_tio, new_tio;
    tcgetattr(STDIN_FILENO, &old_tio);
    new_tio = old_tio;
    new_tio.c_lflag &= (~ICANON & ~ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_tio);

    while (!should_stop_) {
        fd_set set;
        struct timeval timeout;
        FD_ZERO(&set);
        FD_SET(STDIN_FILENO, &set);

        timeout.tv_sec = 0;
        timeout.tv_usec = 100000; // 100ms timeout

        int ret = select(STDIN_FILENO + 1, &set, NULL, NULL, &timeout);

        if (ret > 0) {
            char c;
            if (read(STDIN_FILENO, &c, 1) > 0) {
                handleUserCommand(c);
            }
        }
    }

    // Restore terminal settings
    tcsetattr(STDIN_FILENO, TCSANOW, &old_tio);
}

void HandEyeCalibrator::handleUserCommand(char cmd) {
    std::lock_guard<std::mutex> lock(state_mutex_);

    switch (cmd) {
        case 's':
        case 'S':
            if (current_state_ == CalibrationState::DETECTING && chessboard_detected_) {
                startCollecting();
            } else if (current_state_ == CalibrationState::PAUSED) {
                // Resume from pause - restart detection for next position
                current_state_ = CalibrationState::DETECTING;
                chessboard_detected_ = false;
                RCLCPP_INFO(this->get_logger(), "\n==> STATE: DETECTING");
                RCLCPP_INFO(this->get_logger(), "Move camera to next position and wait for detection...");
            } else {
                if (current_state_ == CalibrationState::DETECTING) {
                    RCLCPP_WARN(this->get_logger(),
                        "Cannot start: chessboard quality insufficient (current: %.3f, required: %.3f)",
                        last_corner_quality_, config_.chessboard_quality_threshold);
                } else {
                    RCLCPP_WARN(this->get_logger(), "Cannot start: invalid state");
                }
            }
            break;

        case 'p':
        case 'P':
            if (current_state_ == CalibrationState::COLLECTING) {
                pauseCollecting();
            }
            break;

        case 'x':
        case 'X':
            RCLCPP_INFO(this->get_logger(), "Stop command received");
            should_stop_ = true;
            break;

        default:
            // Ignore other keys
            break;
    }
}

void HandEyeCalibrator::startCollecting() {
    current_state_ = CalibrationState::COLLECTING;
    current_bucket_.samples.clear();
    current_bucket_.position_id = current_position_index_;

    RCLCPP_INFO(this->get_logger(), "\n==> STATE: COLLECTING");
    RCLCPP_INFO(this->get_logger(), "Position %d/%d - Collecting %d samples...",
                current_position_index_ + 1, config_.num_positions,
                config_.samples_per_position);
}

void HandEyeCalibrator::pauseCollecting() {
    current_state_ = CalibrationState::PAUSED;
    RCLCPP_INFO(this->get_logger(), "\n==> STATE: PAUSED");
    RCLCPP_INFO(this->get_logger(), "Collection paused manually");
}

void HandEyeCalibrator::validateAndSavePosition() {
    std::lock_guard<std::mutex> lock(bucket_mutex_);

    if (current_bucket_.samples.empty()) {
        RCLCPP_WARN(this->get_logger(), "No samples collected for this position");
        return;
    }

    // Calculate statistics
    std::vector<Eigen::Vector3d> sensor_positions;
    std::vector<double> camera_sensor_distances;
    double total_reproj_error = 0.0;

    for (const auto& sample : current_bucket_.samples) {
        Eigen::Vector3d sensor_pos = sample.sensor_pose.block<3,1>(0,3);
        Eigen::Vector3d camera_pos = sample.camera_pose.block<3,1>(0,3);

        sensor_positions.push_back(sensor_pos);
        camera_sensor_distances.push_back((camera_pos - sensor_pos).norm());
        total_reproj_error += sample.reprojection_error;
    }

    // Compute sensor movement (std dev of positions)
    Eigen::Vector3d mean_sensor_pos = Eigen::Vector3d::Zero();
    for (const auto& pos : sensor_positions) {
        mean_sensor_pos += pos;
    }
    mean_sensor_pos /= sensor_positions.size();

    double sensor_movement_variance = 0.0;
    for (const auto& pos : sensor_positions) {
        sensor_movement_variance += (pos - mean_sensor_pos).squaredNorm();
    }
    double sensor_movement_std = std::sqrt(sensor_movement_variance / sensor_positions.size());

    // Validate sensor movement
    bool valid = (sensor_movement_std * 1000.0) <= config_.max_sensor_movement_mm;

    printPositionRecap(current_bucket_, valid);

    if (valid) {
        position_buckets_.push_back(current_bucket_);
        current_position_index_++;

        RCLCPP_INFO(this->get_logger(), "Position %d/%d collected successfully!",
                    current_position_index_, config_.num_positions);

        if (current_position_index_ >= config_.num_positions) {
            // All positions collected - start calibration
            RCLCPP_INFO(this->get_logger(), "\n==> All %d positions collected!", config_.num_positions);
            RCLCPP_INFO(this->get_logger(), "Starting final calibration...");
            current_state_ = CalibrationState::CALIBRATING;
            collection_complete_ = true;

            // Trigger calibration
            bool expected = false;
            if (calibration_started_.compare_exchange_strong(expected, true)) {
                selectBestSamplesAndCalibrate();
            }
        } else {
            // Go to paused state, waiting for user to move camera
            current_state_ = CalibrationState::PAUSED;
        }
    } else {
        RCLCPP_ERROR(this->get_logger(), "Samples DISCARDED due to excessive sensor movement!");
        RCLCPP_INFO(this->get_logger(), "Returning to DETECTING state...");
        current_state_ = CalibrationState::DETECTING;
        chessboard_detected_ = false;
    }

    // Clear current bucket
    current_bucket_.samples.clear();
}

void HandEyeCalibrator::printPositionRecap(const PositionBucket& bucket, bool valid) {
    // Calculate statistics for display
    std::vector<Eigen::Vector3d> sensor_positions;
    std::vector<double> camera_sensor_distances;
    double total_reproj_error = 0.0;

    for (const auto& sample : bucket.samples) {
        Eigen::Vector3d sensor_pos = sample.sensor_pose.block<3,1>(0,3);
        Eigen::Vector3d camera_pos = sample.camera_pose.block<3,1>(0,3);

        sensor_positions.push_back(sensor_pos);
        camera_sensor_distances.push_back((camera_pos - sensor_pos).norm());
        total_reproj_error += sample.reprojection_error;
    }

    Eigen::Vector3d mean_sensor_pos = Eigen::Vector3d::Zero();
    for (const auto& pos : sensor_positions) {
        mean_sensor_pos += pos;
    }
    mean_sensor_pos /= sensor_positions.size();

    double sensor_movement_variance = 0.0;
    for (const auto& pos : sensor_positions) {
        sensor_movement_variance += (pos - mean_sensor_pos).squaredNorm();
    }
    double sensor_movement_std = std::sqrt(sensor_movement_variance / sensor_positions.size());

    double mean_distance = std::accumulate(camera_sensor_distances.begin(),
                                          camera_sensor_distances.end(), 0.0) /
                          camera_sensor_distances.size();

    double distance_variance = 0.0;
    for (double d : camera_sensor_distances) {
        distance_variance += (d - mean_distance) * (d - mean_distance);
    }
    double distance_std = std::sqrt(distance_variance / camera_sensor_distances.size());

    double avg_reproj_error = total_reproj_error / bucket.samples.size();

    RCLCPP_INFO(this->get_logger(), "\n========== POSITION %d RECAP ==========", bucket.position_id + 1);
    RCLCPP_INFO(this->get_logger(), "Samples collected: %zu", bucket.samples.size());
    RCLCPP_INFO(this->get_logger(), "Avg reprojection error: %.3f px", avg_reproj_error);
    RCLCPP_INFO(this->get_logger(), "Camera-Sensor distance: %.2f ± %.2f mm",
                mean_distance * 1000.0, distance_std * 1000.0);
    RCLCPP_INFO(this->get_logger(), "Sensor movement (std dev): %.3f mm %s",
                sensor_movement_std * 1000.0,
                valid ? "[OK]" : "[EXCEEDED THRESHOLD]");
    RCLCPP_INFO(this->get_logger(), "Status: %s", valid ? "VALID" : "INVALID - DISCARDED");
    RCLCPP_INFO(this->get_logger(), "======================================");
}

} // namespace eye_in_hand_calibration