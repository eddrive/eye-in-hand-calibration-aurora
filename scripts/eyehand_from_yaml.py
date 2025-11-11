#!/usr/bin/env python3
"""
Hand-Eye Calibration with automatic selection of best sample pairs.

Selection criteria:
1. Reprojection error < threshold
2. Sensor-camera distance < threshold (e.g., 60mm for endoscope)
3. Relative movement coherence (sensor and camera move together)
"""

import yaml
import numpy as np
import cv2
import os
import sys
from math import acos
from typing import List, Dict, Tuple
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# ============================================
# DIRECTORY PATHS
# ============================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_RESULTS_DIR = REPO_ROOT / "data" / "results"
CHESSBOARD_DIR = REPO_ROOT / "hardware" / "chessboards" / "measured_points"

# ============================================
# CONFIGURABLE PARAMETERS
# ============================================
CONFIG = {
    # Quality filters
    "MAX_REPROJ_ERROR_PX": 0.8,           # Maximum reprojection error (pixels)
    "MAX_SENSOR_CAMERA_DIST_MM": 20.0,    # Maximum sensor-camera distance (mm)

    # Relative movement coherence
    "MAX_MOVEMENT_RATIO": 2.3,            # Max ratio between sensor/camera movement
    "MAX_ROTATION_DIFF_DEG": 25.0,        # Max difference in rotation angles (degrees)

    # Spatial diversity filter
    "USE_SPATIAL_DIVERSITY": True,        # Enable spatial diversity filter
    "MIN_TRANS_DIST_MM": 15.0,            # Minimum translation distance between consecutive samples (mm)
    "MIN_ROT_DIST_DEG": 10.0,             # Minimum rotation distance between consecutive samples (deg)
    "TARGET_DIVERSE_SAMPLES": 25,         # Target number of spatially diverse samples

    # Hand-eye method
    "METHOD": "tsai",  # Options: "tsai", "park", "horaud", "andreff", "daniilidis"

    # Minimum number of samples for calibration
    "MIN_SAMPLES": 10,

    # Iterative refinement
    "USE_ITERATIVE_REFINEMENT": False,    # Enable iterative pair selection based on error (DISABLED for now)
    "TARGET_PAIRS": 20,                   # Target number of pairs after refinement
}
# ============================================

# ----------------------------
# Interactive file selection
# ----------------------------
def select_file_interactive(directory: Path, pattern: str, file_type: str) -> Path:
    """Shows list of files in directory and asks user to choose one."""
    if not directory.exists():
        print(f"❌ ERROR: Directory '{directory}' not found!")
        sys.exit(1)

    # Find all files matching the pattern
    files = sorted(directory.glob(pattern))

    if not files:
        print(f"❌ ERROR: No {pattern} files found in '{directory}'")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Available {file_type} files in: {directory.relative_to(REPO_ROOT)}")
    print(f"{'='*70}")

    for i, file_path in enumerate(files, start=1):
        # Show file size
        size_kb = file_path.stat().st_size / 1024
        print(f"  [{i}] {file_path.name} ({size_kb:.1f} KB)")

    print(f"{'='*70}\n")

    # Ask user
    while True:
        try:
            choice = input(f"Select a file (1-{len(files)}) or 'q' to quit: ").strip()

            if choice.lower() == 'q':
                print("❌ Operation cancelled by user")
                sys.exit(0)

            idx = int(choice) - 1
            if 0 <= idx < len(files):
                selected = files[idx]
                print(f"✓ Selected: {selected.name}\n")
                return selected
            else:
                print(f"❌ Invalid choice. Enter a number between 1 and {len(files)}")
        except ValueError:
            print(f"❌ Invalid input. Enter a number between 1 and {len(files)} or 'q'")
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user")
            sys.exit(0)

# ----------------------------
# SE(3) utilities
# ----------------------------
def matrix_from_pos_quat(position: List[float], quaternion: List[float]) -> np.ndarray:
    """Creates 4x4 matrix from position [x,y,z] and quaternion [x,y,z,w]."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = position
    rot = R.from_quat(quaternion)  # [x, y, z, w]
    T[:3, :3] = rot.as_matrix()
    return T

def T_inv(T: np.ndarray) -> np.ndarray:
    """Inverse of homogeneous transformation."""
    R_mat = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R_mat.T
    Ti[:3, 3] = -R_mat.T @ t
    return Ti

def rot_angle(R_mat: np.ndarray) -> float:
    """Rotation angle (rad) from 3x3 matrix."""
    tr = np.clip((np.trace(R_mat) - 1.0) / 2.0, -1.0, 1.0)
    return acos(tr)

def to_R_t(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts rotation and translation."""
    return T[:3, :3].copy(), T[:3, 3].reshape(3, 1).copy()

# ----------------------------
# Data loading
# ----------------------------
def load_samples(path: str) -> List[Dict]:
    """Loads samples from YAML with new format."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    if 'collected_samples' not in data:
        raise ValueError("YAML does not contain 'collected_samples'")

    samples = data['collected_samples'].get('samples', [])
    if not samples:
        raise ValueError("No samples found")

    return samples

# ----------------------------
# Selection filters
# ----------------------------
def filter_by_reprojection_error(samples: List[Dict], max_error: float) -> List[int]:
    """Filters by reprojection error."""
    valid = []
    for i, sample in enumerate(samples):
        reproj = sample.get('reprojection_error', float('inf'))
        if reproj <= max_error:
            valid.append(i)

    print(f"[Filter 1/3] Reprojection error < {max_error:.1f}px: {len(valid)}/{len(samples)} samples")
    return valid

def filter_by_sensor_camera_distance(samples: List[Dict], indices: List[int],
                                     max_dist_mm: float) -> List[int]:
    """Filters by sensor-camera distance."""
    valid = []

    for idx in indices:
        sample = samples[idx]

        # Extract positions from new format
        sensor_pos = np.array(sample['sensor']['position'])
        camera_pos = np.array(sample['camera']['position'])

        # Calculate distance
        dist_mm = np.linalg.norm(camera_pos - sensor_pos) * 1000.0

        if dist_mm <= max_dist_mm:
            valid.append(idx)
        else:
            sample_id = sample.get('sample_id', idx)
            print(f"  [Discarding sample {sample_id}] "
                  f"Sensor-camera dist: {dist_mm:.1f}mm > {max_dist_mm:.1f}mm")

    print(f"[Filter 2/3] Sensor-camera distance < {max_dist_mm:.1f}mm: "
          f"{len(valid)}/{len(indices)} samples")
    return valid

def filter_by_movement_coherence(samples: List[Dict], indices: List[int],
                                 max_ratio: float, max_rot_diff_deg: float) -> List[int]:
    """
    Filters by coherence of relative movements between successive pairs.
    """
    if len(indices) < 2:
        return indices

    valid = [indices[0]]  # Always keep the first one

    for i in range(1, len(indices)):
        idx_prev = indices[i-1]
        idx_curr = indices[i]

        sample_prev = samples[idx_prev]
        sample_curr = samples[idx_curr]

        # Create matrices from new format
        T_sens_prev = matrix_from_pos_quat(
            sample_prev['sensor']['position'],
            sample_prev['sensor']['orientation']
        )
        T_sens_curr = matrix_from_pos_quat(
            sample_curr['sensor']['position'],
            sample_curr['sensor']['orientation']
        )
        T_cam_prev = matrix_from_pos_quat(
            sample_prev['camera']['position'],
            sample_prev['camera']['orientation']
        )
        T_cam_curr = matrix_from_pos_quat(
            sample_curr['camera']['position'],
            sample_curr['camera']['orientation']
        )

        # Calculate movements
        sens_move_mm = np.linalg.norm(T_sens_curr[:3, 3] - T_sens_prev[:3, 3]) * 1000.0
        cam_move_mm = np.linalg.norm(T_cam_curr[:3, 3] - T_cam_prev[:3, 3]) * 1000.0

        # Calculate relative rotations
        dR_sens = T_sens_curr[:3, :3] @ T_sens_prev[:3, :3].T
        dR_cam = T_cam_curr[:3, :3] @ T_cam_prev[:3, :3].T

        sens_rot_deg = np.degrees(rot_angle(dR_sens))
        cam_rot_deg = np.degrees(rot_angle(dR_cam))

        # Check translation coherence
        if sens_move_mm < 1e-3 and cam_move_mm < 1e-3:
            trans_ok = True
            ratio = 1.0
        elif sens_move_mm < 1e-3 or cam_move_mm < 1e-3:
            trans_ok = False
            ratio = float('inf')
        else:
            ratio = max(sens_move_mm, cam_move_mm) / min(sens_move_mm, cam_move_mm)
            trans_ok = (ratio <= max_ratio)

        # Check rotation coherence
        rot_diff_deg = abs(sens_rot_deg - cam_rot_deg)
        rot_ok = (rot_diff_deg <= max_rot_diff_deg)

        # Decision
        if trans_ok and rot_ok:
            valid.append(idx_curr)
        else:
            sid_prev = sample_prev.get('sample_id', idx_prev)
            sid_curr = sample_curr.get('sample_id', idx_curr)
            print(f"  [Discarding sample {sid_curr}] Incoherent with {sid_prev}: "
                  f"sens={sens_move_mm:.1f}mm/{sens_rot_deg:.1f}°, "
                  f"cam={cam_move_mm:.1f}mm/{cam_rot_deg:.1f}°, "
                  f"ratio={ratio:.2f}, Δrot={rot_diff_deg:.1f}°")

    print(f"[Filter 3/3] Movement coherence (ratio<{max_ratio:.1f}, Δrot<{max_rot_diff_deg:.1f}°): "
          f"{len(valid)}/{len(indices)} samples")
    return valid

def filter_by_spatial_diversity(samples: List[Dict], indices: List[int],
                                 min_trans_dist_mm: float = 15.0,
                                 min_rot_dist_deg: float = 10.0,
                                 target_samples: int = 25) -> List[int]:
    """
    Selects spatially diverse samples using greedy farthest-point algorithm.

    This ensures consecutive pairs have large relative movements, reducing
    calibration errors caused by small movements amplifying noise.

    Args:
        samples: All samples
        indices: Current sample indices
        min_trans_dist_mm: Minimum translation distance between consecutive samples (mm)
        min_rot_dist_deg: Minimum rotation distance between consecutive samples (deg)
        target_samples: Target number of samples to select

    Returns:
        List of indices with maximum spatial diversity
    """
    if len(indices) <= target_samples:
        print(f"[Filter 4/4] Spatial diversity: {len(indices)} samples (no filtering needed)")
        return indices

    # Extract all poses
    poses = []
    for idx in indices:
        sample = samples[idx]
        T_sensor = matrix_from_pos_quat(
            sample['sensor']['position'],
            sample['sensor']['orientation']
        )
        poses.append((idx, T_sensor))

    # Greedy farthest-point selection
    selected = []
    remaining = list(range(len(poses)))

    # Start with first sample (arbitrary)
    selected.append(0)
    remaining.remove(0)

    print(f"\n[Filter 4/4] Selecting {target_samples} spatially diverse samples:")
    print(f"  Strategy: Maximize distance between consecutive samples")
    print(f"  Min translation: {min_trans_dist_mm:.1f}mm, Min rotation: {min_rot_dist_deg:.1f}°\n")

    while len(selected) < target_samples and remaining:
        last_idx = selected[-1]
        last_pose = poses[last_idx][1]

        # Find farthest sample from last selected
        best_dist = -1
        best_candidate = None

        for candidate_idx in remaining:
            candidate_pose = poses[candidate_idx][1]

            # Translation distance
            trans_dist_mm = np.linalg.norm(
                candidate_pose[:3, 3] - last_pose[:3, 3]
            ) * 1000.0

            # Rotation distance
            dR = candidate_pose[:3, :3] @ last_pose[:3, :3].T
            rot_dist_deg = np.degrees(rot_angle(dR))

            # Combined distance (weight translation and rotation equally)
            combined_dist = trans_dist_mm / min_trans_dist_mm + rot_dist_deg / min_rot_dist_deg

            if combined_dist > best_dist:
                best_dist = combined_dist
                best_candidate = candidate_idx
                best_trans = trans_dist_mm
                best_rot = rot_dist_deg

        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)

            sample_id_prev = poses[last_idx][0]
            sample_id_curr = poses[best_candidate][0]

            print(f"  [{len(selected):2d}] Sample {sample_id_curr:3d} "
                  f"(Δ from {sample_id_prev:3d}: "
                  f"trans={best_trans:5.1f}mm, rot={best_rot:5.1f}°)")

    # Convert local indices back to original indices
    result = [poses[i][0] for i in selected]

    print(f"\n✓ Selected {len(result)} spatially diverse samples")
    print(f"  Original indices: {result}\n")

    return result

# ----------------------------
# Iterative pair selection based on calibration error
# ----------------------------
def refine_samples_by_error(samples: List[Dict], indices: List[int], method_name: str,
                            target_pairs: int = 20, max_iterations: int = 50) -> List[int]:
    """
    Iteratively removes samples that create pairs with highest AX≈XB error.

    Improved algorithm that detects problematic samples appearing repeatedly
    in worst pairs and removes them intelligently.

    Args:
        samples: All samples
        indices: Current sample indices
        method_name: Hand-eye calibration method
        target_pairs: Target number of pairs (samples - 1)
        max_iterations: Maximum refinement iterations

    Returns:
        Refined list of sample indices
    """
    current_indices = indices.copy()

    # Track which samples appear in worst pairs
    worst_pair_history = []  # List of (sample_i, sample_j) tuples
    PATTERN_THRESHOLD = 3  # If a sample appears in N consecutive worst pairs, remove it

    print("\n" + "="*70)
    print("ITERATIVE SAMPLE REFINEMENT (Smart Mode)")
    print("="*70)
    print(f"Starting with {len(current_indices)} samples ({len(current_indices)-1} pairs)")
    print(f"Target: {target_pairs} pairs\n")

    for iteration in range(max_iterations):
        if len(current_indices) - 1 <= target_pairs:
            print(f"\n✓ Reached target of {target_pairs} pairs")
            break

        # Calibrate with current samples
        X = run_handeye(samples, current_indices, method_name)

        # Evaluate error for each consecutive pair
        pair_errors = []
        for k in range(len(current_indices) - 1):
            idx_i = current_indices[k]
            idx_j = current_indices[k + 1]

            T_sens_i = matrix_from_pos_quat(
                samples[idx_i]['sensor']['position'],
                samples[idx_i]['sensor']['orientation']
            )
            T_sens_j = matrix_from_pos_quat(
                samples[idx_j]['sensor']['position'],
                samples[idx_j]['sensor']['orientation']
            )
            T_cam_i = matrix_from_pos_quat(
                samples[idx_i]['camera']['position'],
                samples[idx_i]['camera']['orientation']
            )
            T_cam_j = matrix_from_pos_quat(
                samples[idx_j]['camera']['position'],
                samples[idx_j]['camera']['orientation']
            )

            A = T_sens_j @ T_inv(T_sens_i)
            B = T_cam_j @ T_inv(T_cam_i)

            AX = A @ X
            XB = X @ B
            Delta = AX @ T_inv(XB)

            trans_err_mm = np.linalg.norm(Delta[:3, 3]) * 1000.0
            rot_err_deg = np.degrees(rot_angle(Delta[:3, :3]))

            # Combined error (weight rotation errors more)
            combined_err = trans_err_mm + rot_err_deg * 5.0

            pair_errors.append((k, idx_i, idx_j, trans_err_mm, rot_err_deg, combined_err))

        # Find worst pair
        pair_errors.sort(key=lambda x: x[5], reverse=True)
        worst = pair_errors[0]
        _, worst_i, worst_j, worst_trans, worst_rot, _ = worst

        # Add worst pair to history
        worst_pair_history.append((worst_i, worst_j))

        # Keep only recent history (last PATTERN_THRESHOLD iterations)
        if len(worst_pair_history) > PATTERN_THRESHOLD:
            worst_pair_history.pop(0)

        # Analyze if a sample appears repeatedly in worst pairs
        sample_to_remove = None
        removal_reason = "worst_pair"

        if len(worst_pair_history) >= PATTERN_THRESHOLD:
            # Count how many times each sample appears in recent worst pairs
            sample_counts = {}
            for (si, sj) in worst_pair_history:
                sample_counts[si] = sample_counts.get(si, 0) + 1
                sample_counts[sj] = sample_counts.get(sj, 0) + 1

            # Find samples that appear in all recent worst pairs
            problematic_samples = [s for s, count in sample_counts.items()
                                  if count >= PATTERN_THRESHOLD]

            if problematic_samples:
                # If multiple problematic samples, choose the one with highest average error
                if len(problematic_samples) == 1:
                    sample_to_remove = problematic_samples[0]
                else:
                    # Calculate average error for each problematic sample
                    sample_errors = {}
                    for s in problematic_samples:
                        errors = []
                        for (_, idx_i, idx_j, _, _, comb_err) in pair_errors:
                            if idx_i == s or idx_j == s:
                                errors.append(comb_err)
                        if errors:
                            sample_errors[s] = np.mean(errors)

                    # Remove the one with highest average error
                    sample_to_remove = max(sample_errors.keys(), key=lambda s: sample_errors[s])

                removal_reason = f"outlier (appeared in {sample_counts[sample_to_remove]} worst pairs)"

                # Clear history after removing an outlier
                worst_pair_history.clear()

        # If no pattern detected, use default strategy: remove second element of worst pair
        if sample_to_remove is None:
            sample_to_remove = worst_j
            removal_reason = "worst_pair"

        # Remove the selected sample
        current_indices.remove(sample_to_remove)

        print(f"[Iter {iteration+1:2d}] Removed sample {sample_to_remove:3d} "
              f"(pair {worst_i:3d}→{worst_j:3d}): "
              f"err={worst_trans:5.1f}mm/{worst_rot:5.1f}° | "
              f"Reason: {removal_reason} | "
              f"Remaining: {len(current_indices)} samples")

    print(f"\n✓ Refinement complete: {len(current_indices)} samples selected\n")
    return current_indices

# ----------------------------
# Hand-Eye Calibration
# ----------------------------
def run_handeye(samples: List[Dict], indices: List[int], method_name: str) -> np.ndarray:
    """
    Performs hand-eye calibration with OpenCV.

    CRITICAL: Samples contain (in new format):
      - sensor: {position, orientation} = T_aurora_to_sensor
      - camera: {position, orientation} = T_aurora_to_camera

    Since both sensor and camera are in the same Aurora frame,
    we can directly use their poses for calibration:
      - R_gripper2base = R_aurora_to_sensor (direct)
      - R_target2cam = R_aurora_to_camera (direct, NO inversion!)

    OpenCV will solve: T_aurora_to_camera = T_aurora_to_sensor @ X
    Where X = T_sensor_to_camera (the transformation we want)
    """
    method_map = {
        'tsai': cv2.CALIB_HAND_EYE_TSAI,
        'park': cv2.CALIB_HAND_EYE_PARK,
        'horaud': cv2.CALIB_HAND_EYE_HORAUD,
        'andreff': cv2.CALIB_HAND_EYE_ANDREFF,
        'daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    
    if method_name not in method_map:
        raise ValueError(f"Method '{method_name}' not supported: {list(method_map.keys())}")

    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []

    for idx in indices:
        sample = samples[idx]

        # Gripper to base (sensor - direct use)
        T_gripper2base = matrix_from_pos_quat(
            sample['sensor']['position'],
            sample['sensor']['orientation']
        )
        R_gb, t_gb = to_R_t(T_gripper2base)

        # Target to camera (camera - INVERSION!)
        T_aurora_to_camera = matrix_from_pos_quat(
            sample['camera']['position'],
            sample['camera']['orientation']
        )
        T_camera_to_aurora = T_inv(T_aurora_to_camera)
        R_tc, t_tc = to_R_t(T_camera_to_aurora)

        R_g2b.append(R_gb)
        t_g2b.append(t_gb)
        R_t2c.append(R_tc)
        t_t2c.append(t_tc)

    # Calibration
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=method_map[method_name]
    )

    # Build 4x4 matrix
    X = np.eye(4, dtype=np.float64)
    X[:3, :3] = R_cam2gripper
    X[:3, 3] = t_cam2gripper.flatten()

    return X

# ----------------------------
# Movement statistics
# ----------------------------
def analyze_movements(samples: List[Dict], indices: List[int]) -> None:
    """Analyzes and prints movement statistics between consecutive pairs."""
    movements_A = []
    movements_B = []

    print("\nMovement analysis between consecutive pairs:")
    print("-" * 80)

    # ONLY consecutive pairs
    for k in range(len(indices) - 1):
        idx_i = indices[k]
        idx_j = indices[k + 1]

        # Create matrices from new format
        T_sens_i = matrix_from_pos_quat(
            samples[idx_i]['sensor']['position'],
            samples[idx_i]['sensor']['orientation']
        )
        T_sens_j = matrix_from_pos_quat(
            samples[idx_j]['sensor']['position'],
            samples[idx_j]['sensor']['orientation']
        )

        T_cam_i = matrix_from_pos_quat(
            samples[idx_i]['camera']['position'],
            samples[idx_i]['camera']['orientation']
        )
        T_cam_j = matrix_from_pos_quat(
            samples[idx_j]['camera']['position'],
            samples[idx_j]['camera']['orientation']
        )

        # A = sensor_j * sensor_i^-1
        A = T_sens_j @ T_inv(T_sens_i)

        # B = camera_j * camera_i^-1
        B = T_cam_j @ T_inv(T_cam_i)

        # Analyze movements
        trans_A = np.linalg.norm(A[:3, 3]) * 1000.0  # mm
        rot_A = np.degrees(rot_angle(A[:3, :3]))     # deg
        trans_B = np.linalg.norm(B[:3, 3]) * 1000.0  # mm
        rot_B = np.degrees(rot_angle(B[:3, :3]))     # deg

        movements_A.append((trans_A, rot_A))
        movements_B.append((trans_B, rot_B))

        # Print for each pair
        print(f"  Pair {k:2d} (sample {idx_i:3d}→{idx_j:3d}): "
              f"sensor={trans_A:6.1f}mm/{rot_A:5.1f}°  |  "
              f"camera={trans_B:6.1f}mm/{rot_B:5.1f}°")

    movements_A = np.array(movements_A)
    movements_B = np.array(movements_B)

    print("-" * 80)
    print("\nSENSOR movement statistics:")
    print(f"  Translation: mean={np.mean(movements_A[:, 0]):6.1f}mm, "
          f"median={np.median(movements_A[:, 0]):6.1f}mm, "
          f"min={np.min(movements_A[:, 0]):6.1f}mm, "
          f"max={np.max(movements_A[:, 0]):6.1f}mm")
    print(f"  Rotation:    mean={np.mean(movements_A[:, 1]):6.1f}°, "
          f"median={np.median(movements_A[:, 1]):6.1f}°, "
          f"min={np.min(movements_A[:, 1]):6.1f}°, "
          f"max={np.max(movements_A[:, 1]):6.1f}°")

    print("\nCAMERA movement statistics:")
    print(f"  Translation: mean={np.mean(movements_B[:, 0]):6.1f}mm, "
          f"median={np.median(movements_B[:, 0]):6.1f}mm, "
          f"min={np.min(movements_B[:, 0]):6.1f}mm, "
          f"max={np.max(movements_B[:, 0]):6.1f}mm")
    print(f"  Rotation:    mean={np.mean(movements_B[:, 1]):6.1f}°, "
          f"median={np.median(movements_B[:, 1]):6.1f}°, "
          f"min={np.min(movements_B[:, 1]):6.1f}°, "
          f"max={np.max(movements_B[:, 1]):6.1f}°")

    # Ratio between movements
    ratios_trans = np.maximum(movements_A[:, 0], movements_B[:, 0]) / (np.minimum(movements_A[:, 0], movements_B[:, 0]) + 1e-6)
    ratios_rot = np.maximum(movements_A[:, 1], movements_B[:, 1]) / (np.minimum(movements_A[:, 1], movements_B[:, 1]) + 1e-6)

    print("\nMovement ratio (sensor/camera):")
    print(f"  Translation: mean={np.mean(ratios_trans):.2f}, median={np.median(ratios_trans):.2f}")
    print(f"  Rotation:    mean={np.mean(ratios_rot):.2f}, median={np.median(ratios_rot):.2f}")
    print()

# ----------------------------
# Error evaluation
# ----------------------------
def compute_camera_chessboard_distance(samples: List[Dict], indices: List[int],
                                       chessboard_center: np.ndarray) -> Dict:
    """
    Calculates the average distance between camera and chessboard center.

    Args:
        samples: All samples
        indices: Indices of samples to use
        chessboard_center: 3D position of chessboard center in Aurora frame [x, y, z] in meters

    Returns:
        Dict with 'mean', 'median', 'min', 'max' distances in mm
    """
    distances = []

    for idx in indices:
        sample = samples[idx]
        T_aurora_to_camera = matrix_from_pos_quat(
            sample['camera']['position'],
            sample['camera']['orientation']
        )
        # Camera position in Aurora frame
        camera_pos = T_aurora_to_camera[:3, 3]
        # Distance from camera to chessboard center
        dist_m = np.linalg.norm(camera_pos - chessboard_center)
        distances.append(dist_m * 1000.0)  # Convert to mm

    distances = np.array(distances)

    return {
        'mean': float(np.mean(distances)),
        'median': float(np.median(distances)),
        'min': float(np.min(distances)),
        'max': float(np.max(distances)),
        'std': float(np.std(distances, ddof=1)) if len(distances) > 1 else 0.0,
    }

def compute_errors_alternative(samples: List[Dict], indices: List[int], X: np.ndarray) -> Dict:
    """
    Calculates errors by comparing predicted vs measured camera pose in base frame (Aurora).
    - Predicted:  T_aurora_to_camera_pred = T_aurora_to_sensor @ X
    - Measured:   T_aurora_to_camera_meas = (from sample)

    Rotation error: angle between R_pred and R_meas [deg]
    Translation error: ||t_meas - t_pred|| [mm]
    """

    def _projR(R):
        """Projects an almost-rotation matrix onto SO(3) via SVD."""
        U, _, Vt = np.linalg.svd(R)
        return U @ Vt
    
    rot_errs = []
    trans_errs = []
    
    for idx in indices:
        sample = samples[idx]

        # Build T_aurora_to_sensor (A) and measured T_aurora_to_camera (B_meas)
        T_aurora_to_sensor = matrix_from_pos_quat(
            sample['sensor']['position'],
            sample['sensor']['orientation']
        )
        T_aurora_to_camera_meas = matrix_from_pos_quat(
            sample['camera']['position'],
            sample['camera']['orientation']
        )

        # Predict camera position using X
        T_aurora_to_camera_pred = T_aurora_to_sensor @ X

        # Extract rotations and project onto SO(3)
        R_pred = _projR(T_aurora_to_camera_pred[:3, :3])
        R_meas = _projR(T_aurora_to_camera_meas[:3, :3])
        t_pred = T_aurora_to_camera_pred[:3, 3]
        t_meas = T_aurora_to_camera_meas[:3, 3]

        # Rotation error: angle of relative rotation
        R_rel = _projR(R_pred.T @ R_meas)
        rot_err_rad = rot_angle(R_rel)

        # Translation error: euclidean distance
        trans_err_m = np.linalg.norm(t_meas - t_pred)
        
        rot_errs.append(rot_err_rad)
        trans_errs.append(trans_err_m)
    
    rot_errs = np.array(rot_errs)
    trans_errs = np.array(trans_errs)
    
    def _stats(arr):
        if arr.size == 0:
            return dict(min=np.nan, median=np.nan, mean=np.nan, std=np.nan,
                       rms=np.nan, max=np.nan, count=0)
        return dict(
            min=float(arr.min()),
            median=float(np.median(arr)),
            mean=float(arr.mean()),
            std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            rms=float(np.sqrt((arr**2).mean())),
            max=float(arr.max()),
            count=int(arr.size),
        )
    
    return {
        'rotation_deg': _stats(np.degrees(rot_errs)),
        'translation_mm': _stats(trans_errs * 1000.0),
        'per_sample': {
            'rotation_deg': np.degrees(rot_errs),
            'translation_mm': trans_errs * 1000.0,
        }
    }

# ----------------------------
# Save results
# ----------------------------
def save_results(filename: Path, X: np.ndarray, stats_pred: Dict,
                indices: List[int], config: Dict, input_file: str, chessboard_file: str,
                avg_camera_chessboard_dist_mm: float = None):
    """Saves results to file."""
    with open(filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("HAND-EYE CALIBRATION RESULTS\n")
        f.write("="*70 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Input file:              {input_file}\n")
        f.write(f"  Chessboard file:         {chessboard_file}\n")
        f.write(f"  Method:                  {config['METHOD']}\n")
        f.write(f"  Max reproj error:        {config['MAX_REPROJ_ERROR_PX']:.1f} px\n")
        f.write(f"  Max sensor-camera dist:  {config['MAX_SENSOR_CAMERA_DIST_MM']:.1f} mm\n")
        f.write(f"  Max movement ratio:      {config['MAX_MOVEMENT_RATIO']:.2f}\n")
        f.write(f"  Max rotation diff:       {config['MAX_ROTATION_DIFF_DEG']:.1f} deg\n")
        if config.get('USE_SPATIAL_DIVERSITY', False):
            f.write(f"  Spatial diversity:       ENABLED\n")
            f.write(f"    Min trans distance:    {config['MIN_TRANS_DIST_MM']:.1f} mm\n")
            f.write(f"    Min rot distance:      {config['MIN_ROT_DIST_DEG']:.1f} deg\n")
            f.write(f"    Target samples:        {config['TARGET_DIVERSE_SAMPLES']}\n")
        f.write(f"  Samples used:            {len(indices)}\n")
        f.write(f"  Sample indices:          {indices}\n")
        if avg_camera_chessboard_dist_mm is not None:
            f.write(f"  Avg camera-chessboard:   {avg_camera_chessboard_dist_mm:.2f} mm\n")
        f.write("\n")

        # Metriche trasformazione X
        trans_norm_mm = np.linalg.norm(X[:3, 3]) * 1000.0
        rot_deg = np.degrees(rot_angle(X[:3, :3]))

        f.write("Transformation X (sensor -> camera):\n")
        f.write(f"  Translation norm:  {trans_norm_mm:.3f} mm\n")
        f.write(f"  Rotation angle:    {rot_deg:.4f} deg\n")
        f.write("\n")

        f.write("Matrix X:\n")
        for row in X:
            f.write(f"  {' '.join(f'{x:12.6f}' for x in row)}\n")
        f.write("\n")

        # Quaternion
        quat = R.from_matrix(X[:3, :3]).as_quat()  # [x, y, z, w]
        f.write("Rotation (Quaternion [x, y, z, w]):\n")
        f.write(f"  [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]\n")
        f.write("\n")

        # Errori predizione diretta
        f.write("="*70 + "\n")
        f.write("Calibration Errors:\n")
        f.write("="*70 + "\n")
        f.write(f"  Number of samples:   {stats_pred['rotation_deg']['count']}\n")
        f.write(f"  Rotation min:        {stats_pred['rotation_deg']['min']:.4f} deg\n")
        f.write(f"  Rotation median:     {stats_pred['rotation_deg']['median']:.4f} deg\n")
        f.write(f"  Rotation max:        {stats_pred['rotation_deg']['max']:.4f} deg\n")
        f.write(f"  Rotation mean:       {stats_pred['rotation_deg']['mean']:.4f} deg\n")
        f.write(f"  Rotation std:        {stats_pred['rotation_deg']['std']:.4f} deg\n")
        f.write(f"  Rotation RMS:        {stats_pred['rotation_deg']['rms']:.4f} deg\n")
        f.write(f"  Translation min:     {stats_pred['translation_mm']['min']:.3f} mm\n")
        f.write(f"  Translation median:  {stats_pred['translation_mm']['median']:.3f} mm\n")
        f.write(f"  Translation max:     {stats_pred['translation_mm']['max']:.3f} mm\n")
        f.write(f"  Translation mean:    {stats_pred['translation_mm']['mean']:.3f} mm\n")
        f.write(f"  Translation std:     {stats_pred['translation_mm']['std']:.3f} mm\n")
        f.write(f"  Translation RMS:     {stats_pred['translation_mm']['rms']:.3f} mm\n")
        f.write("\n")
        f.write("="*70 + "\n")

    print(f"\n✓ Results saved to: {filename}")

class CalibrationViewer:
    def __init__(self, samples, indices, X, chessboard_corners=None):
        self.samples = samples
        self.indices = indices
        self.X = X
        self.current_index = 0
        self.chessboard_corners = chessboard_corners

        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        print("\n=== CONTROLS ===")
        print("SPACE → show next sample")
        print("ESC → close visualization")
        print("===================\n")

        # Draw chessboard if available
        if self.chessboard_corners is not None:
            self.plot_chessboard(self.chessboard_corners)

        # Show first sample
        self.update_plot()
        plt.show()

    def matrix_from_pos_quat(self, position, quaternion):
        T = np.eye(4)
        T[:3, 3] = position
        T[:3, :3] = R.from_quat(quaternion).as_matrix()
        return T

    def draw_frame(self, T, label, color, length=0.02):
        """Draws a 3D coordinate frame from a 4x4 matrix."""
        origin = T[:3, 3]
        Rm = T[:3, :3]
        colors = ['r', 'g', 'b']
        for i in range(3):
            self.ax.plot(
                [origin[0], origin[0] + Rm[0, i]*length],
                [origin[1], origin[1] + Rm[1, i]*length],
                [origin[2], origin[2] + Rm[2, i]*length],
                color=colors[i], linewidth=2, alpha=0.9
            )
        self.ax.text(origin[0], origin[1], origin[2], label, color=color, fontsize=10)

    def plot_chessboard(self, chessboard):
        """Draws the chessboard in green (remains fixed)."""
        corners = np.array([[c['x'], c['y'], c['z']] for c in chessboard['points']])
        rows, cols = chessboard['rows'], chessboard['cols']

        # Corner points
        self.ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2],
                        c='green', marker='s', s=50, alpha=0.7,
                        edgecolors='darkgreen', linewidths=1,
                        label=f'Chessboard ({rows}x{cols})')

        # Grid lines
        for r in range(rows):
            row_pts = corners[r*cols:(r+1)*cols]
            self.ax.plot(row_pts[:, 0], row_pts[:, 1], row_pts[:, 2],
                         'g-', alpha=0.3, linewidth=1)
        for c in range(cols):
            col_pts = corners[c::cols]
            self.ax.plot(col_pts[:, 0], col_pts[:, 1], col_pts[:, 2],
                         'g-', alpha=0.3, linewidth=1)

        # Center
        center = np.mean(corners, axis=0)
        self.ax.scatter(center[0], center[1], center[2],
                        c='darkgreen', marker='X', s=200,
                        edgecolors='black', linewidths=1.5,
                        label='Chessboard center')

        print(f"✓ Chessboard loaded: {rows}x{cols} ({len(corners)} points)")

    def update_plot(self):
        """Shows only the current sample (one frame at a time)."""
        self.ax.cla()

        # Redraw chessboard if present
        if self.chessboard_corners is not None:
            self.plot_chessboard(self.chessboard_corners)

        idx = self.indices[self.current_index]
        sample = self.samples[idx]

        T_sensor = self.matrix_from_pos_quat(
            sample['sensor']['position'], sample['sensor']['orientation'])
        T_camera = self.matrix_from_pos_quat(
            sample['camera']['position'], sample['camera']['orientation'])
        T_est = T_sensor @ self.X

        # Draw frames
        self.draw_frame(T_sensor, "Sensor", "red")
        self.draw_frame(T_camera, "Camera (measured)", "blue")
        self.draw_frame(T_est, "Camera (estimated)", "magenta")

        # Lines between frames
        self.ax.plot([T_sensor[0, 3], T_camera[0, 3]],
                     [T_sensor[1, 3], T_camera[1, 3]],
                     [T_sensor[2, 3], T_camera[2, 3]],
                     'gray', linewidth=1.5, alpha=0.6)
        self.ax.plot([T_sensor[0, 3], T_est[0, 3]],
                     [T_sensor[1, 3], T_est[1, 3]],
                     [T_sensor[2, 3], T_est[2, 3]],
                     'magenta', linewidth=1.5, alpha=0.6)

        # Distances
        dist_meas = np.linalg.norm(T_camera[:3, 3] - T_sensor[:3, 3])
        dist_est = np.linalg.norm(T_est[:3, 3] - T_sensor[:3, 3])

        # Title
        self.ax.set_title(
            f"Sample {self.current_index+1}/{len(self.indices)} | "
            f"Dist. Sensor→Camera: measured={dist_meas*1000:.1f} mm, estimated={dist_est*1000:.1f} mm",
            fontsize=12, fontweight='bold')

        # Axes labels
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")

        # Default view orientation
        self.ax.view_init(elev=30, azim=-45)

        self.ax.legend()
        self.ax.grid(True)

        # Dynamic centered limits
        all_pts = [T_sensor[:3, 3], T_camera[:3, 3], T_est[:3, 3]]
        if self.chessboard_corners is not None:
            corners = np.array([[c['x'], c['y'], c['z']] for c in self.chessboard_corners['points']])
            all_pts.extend(corners)
        all_pts = np.vstack(all_pts)

        center = np.mean(all_pts, axis=0)
        max_range = np.max(np.linalg.norm(all_pts - center, axis=1)) * 2
        self.ax.set_xlim(center[0] - max_range, center[0] + max_range)
        self.ax.set_ylim(center[1] - max_range, center[1] + max_range)
        self.ax.set_zlim(center[2] - max_range, center[2] + max_range)

        self.fig.canvas.draw()



    def on_key_press(self, event):
        if event.key == ' ':
            self.current_index = (self.current_index + 1) % len(self.indices)
            self.update_plot()
        elif event.key == 'escape':
            plt.close(self.fig)


# ----------------------------
# Main
# ----------------------------
def main():
    print("\n" + "="*70)
    print("HAND-EYE CALIBRATION")
    print("="*70 + "\n")

    # Selezione interattiva del file collected_samples
    samples_file = select_file_interactive(
        DATA_RESULTS_DIR,
        "collected_samples*.yaml",
        "collected samples"
    )

    # Interactive chessboard selection
    chessboard_file = select_file_interactive(
        CHESSBOARD_DIR,
        "*.yaml",
        "chessboard"
    )

    # Load samples
    print(f"Loading samples from: {samples_file.relative_to(REPO_ROOT)}")
    samples = load_samples(str(samples_file))
    print(f"✓ Loaded {len(samples)} samples\n")

    # Apply filters
    print("="*70)
    print("APPLYING FILTERS")
    print("="*70 + "\n")

    # Filter 1: Reprojection error
    indices = filter_by_reprojection_error(samples, CONFIG["MAX_REPROJ_ERROR_PX"])

    if len(indices) < CONFIG["MIN_SAMPLES"]:
        print(f"\n❌ ERROR: Too few samples after filter 1 ({len(indices)} < {CONFIG['MIN_SAMPLES']})")
        sys.exit(1)

    # Filter 2: Sensor-camera distance
    indices = filter_by_sensor_camera_distance(samples, indices,
                                               CONFIG["MAX_SENSOR_CAMERA_DIST_MM"])

    if len(indices) < CONFIG["MIN_SAMPLES"]:
        print(f"\n❌ ERROR: Too few samples after filter 2 ({len(indices)} < {CONFIG['MIN_SAMPLES']})")
        sys.exit(1)

    # Filter 3: Movement coherence
    indices = filter_by_movement_coherence(samples, indices,
                                          CONFIG["MAX_MOVEMENT_RATIO"],
                                          CONFIG["MAX_ROTATION_DIFF_DEG"])

    if len(indices) < CONFIG["MIN_SAMPLES"]:
        print(f"\n❌ ERROR: Too few samples after filter 3 ({len(indices)} < {CONFIG['MIN_SAMPLES']})")
        sys.exit(1)

    # Filter 4: Spatial diversity (NEW!)
    if CONFIG["USE_SPATIAL_DIVERSITY"]:
        indices = filter_by_spatial_diversity(
            samples, indices,
            min_trans_dist_mm=CONFIG["MIN_TRANS_DIST_MM"],
            min_rot_dist_deg=CONFIG["MIN_ROT_DIST_DEG"],
            target_samples=CONFIG["TARGET_DIVERSE_SAMPLES"]
        )

        if len(indices) < CONFIG["MIN_SAMPLES"]:
            print(f"\n❌ ERROR: Too few samples after filter 4 ({len(indices)} < {CONFIG['MIN_SAMPLES']})")
            sys.exit(1)

    print(f"\n✓ Filters completed: {len(indices)} samples selected\n")

    # Iterative refinement (optional)
    if CONFIG["USE_ITERATIVE_REFINEMENT"]:
        indices = refine_samples_by_error(
            samples, indices, CONFIG["METHOD"],
            target_pairs=CONFIG["TARGET_PAIRS"]
        )

    # Calibration
    print("="*70)
    print("HAND-EYE CALIBRATION")
    print("="*70 + "\n")

    print(f"Method: {CONFIG['METHOD'].upper()}")
    print(f"Samples used: {len(indices)}")
    print(f"Indices: {indices}\n")

    try:
        X = run_handeye(samples, indices, CONFIG["METHOD"])
    except Exception as e:
        print(f"\n❌ ERROR during calibration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("✓ Calibration completed\n")

    # Print transformation results
    trans_norm_mm = np.linalg.norm(X[:3, 3]) * 1000.0
    rot_deg = np.degrees(rot_angle(X[:3, :3]))

    print("Transformation X (sensor -> camera):")
    print(f"  Translation norm:  {trans_norm_mm:.3f} mm")
    print(f"  Rotation angle:    {rot_deg:.4f} deg\n")

    print("Matrix X:")
    np.set_printoptions(precision=6, suppress=True)
    print(X)
    print()

    # Movement statistics
    print("="*70)
    print("MOVEMENT STATISTICS")
    print("="*70)
    analyze_movements(samples, indices)

    # Evaluation with direct prediction method
    print("="*70)
    print("ERROR EVALUATION")
    print("="*70 + "\n")

    stats_pred = compute_errors_alternative(samples, indices, X)

    print("Camera prediction errors in Aurora frame:")
    print(f"  Samples evaluated:   {stats_pred['rotation_deg']['count']}")
    print(f"  Rotation min:        {stats_pred['rotation_deg']['min']:.4f} deg")
    print(f"  Rotation median:     {stats_pred['rotation_deg']['median']:.4f} deg")
    print(f"  Rotation max:        {stats_pred['rotation_deg']['max']:.4f} deg")
    print(f"  Rotation mean:       {stats_pred['rotation_deg']['mean']:.4f} deg")
    print(f"  Rotation std:        {stats_pred['rotation_deg']['std']:.4f} deg")
    print(f"  Rotation RMS:        {stats_pred['rotation_deg']['rms']:.4f} deg")
    print(f"  Translation min:     {stats_pred['translation_mm']['min']:.3f} mm")
    print(f"  Translation median:  {stats_pred['translation_mm']['median']:.3f} mm")
    print(f"  Translation max:     {stats_pred['translation_mm']['max']:.3f} mm")
    print(f"  Translation mean:    {stats_pred['translation_mm']['mean']:.3f} mm")
    print(f"  Translation std:     {stats_pred['translation_mm']['std']:.3f} mm")
    print(f"  Translation RMS:     {stats_pred['translation_mm']['rms']:.3f} mm\n")

    # Load selected chessboard
    chessboard_corners = None
    try:
        with open(chessboard_file, "r") as f:
            chessboard_data = yaml.safe_load(f)
            chessboard_corners = chessboard_data.get('chessboard_corners')
            if chessboard_corners:
                print(f"✓ Chessboard loaded: {chessboard_file.name}")
    except Exception as e:
        print(f"⚠ Unable to load chessboard: {e}")

    # Calculate chessboard center from loaded corners
    chessboard_center = np.array([0.0, 0.0, 0.0])  # Default to origin if no chessboard
    if chessboard_corners is not None and 'points' in chessboard_corners:
        corners = np.array([[c['x'], c['y'], c['z']] for c in chessboard_corners['points']])
        chessboard_center = np.mean(corners, axis=0)
        print(f"Chessboard center in Aurora frame: [{chessboard_center[0]:.6f}, "
              f"{chessboard_center[1]:.6f}, {chessboard_center[2]:.6f}] m\n")

    # Calculate camera-chessboard distance statistics
    print("="*70)
    print("CAMERA-CHESSBOARD DISTANCE")
    print("="*70 + "\n")

    camera_chessboard_stats = compute_camera_chessboard_distance(samples, indices, chessboard_center)

    print(f"Distance from camera to chessboard center:")
    print(f"  Mean:    {camera_chessboard_stats['mean']:.2f} mm")
    print(f"  Median:  {camera_chessboard_stats['median']:.2f} mm")
    print(f"  Min:     {camera_chessboard_stats['min']:.2f} mm")
    print(f"  Max:     {camera_chessboard_stats['max']:.2f} mm")
    print(f"  Std:     {camera_chessboard_stats['std']:.2f} mm\n")

    # Generate output file name with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = DATA_RESULTS_DIR / f"handeye_result_{timestamp}.txt"

    # Save results
    save_results(
        output_file, X, stats_pred, indices, CONFIG,
        str(samples_file.relative_to(REPO_ROOT)),
        str(chessboard_file.relative_to(REPO_ROOT)),
        avg_camera_chessboard_dist_mm=camera_chessboard_stats['mean']
    )

    print("="*70)
    print("✓ CALIBRATION COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")

    # Start visualization
    CalibrationViewer(samples, indices, X, chessboard_corners=chessboard_corners)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
