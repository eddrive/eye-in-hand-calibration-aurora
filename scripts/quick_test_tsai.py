#!/usr/bin/env python3

import yaml
import numpy as np
import cv2
import random
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from math import acos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_RESULTS_DIR = REPO_ROOT / "data" / "results"
CHESSBOARD_DIR = REPO_ROOT / "hardware" / "chessboards" / "measured_points"

# Configuration
NUM_SAMPLES = 15
NUM_ITERATIONS = 1000

def select_file_interactive(directory: Path, pattern: str, file_type: str) -> Path:
    files = sorted(directory.glob(pattern))

    print(f"\n{'='*70}")
    print(f"Available {file_type} files in: {directory.relative_to(REPO_ROOT)}")
    print(f"{'='*70}")

    for i, file_path in enumerate(files, start=1):
        size_kb = file_path.stat().st_size / 1024
        print(f"  [{i}] {file_path.name} ({size_kb:.1f} KB)")

    print(f"{'='*70}\n")

    while True:
        choice = input(f"Select a file (1-{len(files)}): ").strip()
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            selected = files[idx]
            print(f"✓ Selected: {selected.name}\n")
            return selected

def matrix_from_pos_quat(position, quaternion):
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = position
    T[:3, :3] = R.from_quat(quaternion).as_matrix()
    return T

def T_inv(T):
    R_mat = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R_mat.T
    Ti[:3, 3] = -R_mat.T @ t
    return Ti

def rot_angle(R_mat):
    tr = np.clip((np.trace(R_mat) - 1.0) / 2.0, -1.0, 1.0)
    return acos(tr)

def to_R_t(T):
    return T[:3, :3].copy(), T[:3, 3].reshape(3, 1).copy()

def load_samples(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data['collected_samples'].get('samples', [])

def filter_samples(samples):
    valid = []
    for i, sample in enumerate(samples):
        reproj = sample.get('reprojection_error', float('inf'))
        if reproj > 0.6:
            continue

        sensor_pos = np.array(sample['sensor']['position'])
        camera_pos = np.array(sample['camera']['position'])
        dist_mm = np.linalg.norm(camera_pos - sensor_pos) * 1000.0

        if 10.0 <= dist_mm <= 16.0:
            valid.append(i)

    return valid

def run_handeye_tsai(samples, indices):
    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []

    for idx in indices:
        sample = samples[idx]

        T_gripper2base = matrix_from_pos_quat(
            sample['sensor']['position'],
            sample['sensor']['orientation']
        )
        R_gb, t_gb = to_R_t(T_gripper2base)

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

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI
    )

    X = np.eye(4, dtype=np.float64)
    X[:3, :3] = R_cam2gripper
    X[:3, 3] = t_cam2gripper.flatten()

    return X

def compute_rms(samples, indices, X):
    rot_errs = []
    trans_errs = []

    for idx in indices:
        sample = samples[idx]

        T_sensor = matrix_from_pos_quat(
            sample['sensor']['position'],
            sample['sensor']['orientation']
        )
        T_camera_meas = matrix_from_pos_quat(
            sample['camera']['position'],
            sample['camera']['orientation']
        )

        T_camera_pred = T_sensor @ X

        R_error = T_camera_meas[:3, :3] @ T_camera_pred[:3, :3].T
        rot_err_rad = rot_angle(R_error)

        trans_err_m = np.linalg.norm(T_camera_meas[:3, 3] - T_camera_pred[:3, 3])

        rot_errs.append(rot_err_rad)
        trans_errs.append(trans_err_m)

    rot_rms = np.sqrt(np.mean(np.array(rot_errs)**2))
    trans_rms = np.sqrt(np.mean(np.array(trans_errs)**2))

    return np.degrees(rot_rms), trans_rms * 1000.0

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

        if self.chessboard_corners is not None:
            self.plot_chessboard(self.chessboard_corners)

        self.update_plot()
        plt.show()

    def matrix_from_pos_quat(self, position, quaternion):
        T = np.eye(4)
        T[:3, 3] = position
        T[:3, :3] = R.from_quat(quaternion).as_matrix()
        return T

    def draw_frame(self, T, label, color, length=0.02):
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
        corners = np.array([[c['x'], c['y'], c['z']] for c in chessboard['points']])
        rows, cols = chessboard['rows'], chessboard['cols']

        self.ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2],
                        c='green', marker='s', s=50, alpha=0.7,
                        edgecolors='darkgreen', linewidths=1,
                        label=f'Chessboard ({rows}x{cols})')

        for r in range(rows):
            row_pts = corners[r*cols:(r+1)*cols]
            self.ax.plot(row_pts[:, 0], row_pts[:, 1], row_pts[:, 2],
                         'g-', alpha=0.3, linewidth=1)
        for c in range(cols):
            col_pts = corners[c::cols]
            self.ax.plot(col_pts[:, 0], col_pts[:, 1], col_pts[:, 2],
                         'g-', alpha=0.3, linewidth=1)

        center = np.mean(corners, axis=0)
        self.ax.scatter(center[0], center[1], center[2],
                        c='darkgreen', marker='X', s=200,
                        edgecolors='black', linewidths=1.5,
                        label='Chessboard center')

        print(f"✓ Chessboard loaded: {rows}x{cols} ({len(corners)} points)")

    def update_plot(self):
        self.ax.cla()

        if self.chessboard_corners is not None:
            self.plot_chessboard(self.chessboard_corners)

        idx = self.indices[self.current_index]
        sample = self.samples[idx]

        T_sensor = self.matrix_from_pos_quat(
            sample['sensor']['position'], sample['sensor']['orientation'])
        T_camera = self.matrix_from_pos_quat(
            sample['camera']['position'], sample['camera']['orientation'])
        T_est = T_sensor @ self.X

        self.draw_frame(T_sensor, "Sensor", "red")
        self.draw_frame(T_camera, "Camera (measured)", "blue")
        self.draw_frame(T_est, "Camera (estimated)", "magenta")

        self.ax.plot([T_sensor[0, 3], T_camera[0, 3]],
                     [T_sensor[1, 3], T_camera[1, 3]],
                     [T_sensor[2, 3], T_camera[2, 3]],
                     'gray', linewidth=1.5, alpha=0.6)
        self.ax.plot([T_sensor[0, 3], T_est[0, 3]],
                     [T_sensor[1, 3], T_est[1, 3]],
                     [T_sensor[2, 3], T_est[2, 3]],
                     'magenta', linewidth=1.5, alpha=0.6)

        dist_meas = np.linalg.norm(T_camera[:3, 3] - T_sensor[:3, 3])
        dist_est = np.linalg.norm(T_est[:3, 3] - T_sensor[:3, 3])

        self.ax.set_title(
            f"Sample {self.current_index+1}/{len(self.indices)} (idx={idx}) | "
            f"Dist. Sensor→Camera: measured={dist_meas*1000:.1f} mm, estimated={dist_est*1000:.1f} mm",
            fontsize=12, fontweight='bold')

        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")

        self.ax.view_init(elev=30, azim=-45)

        self.ax.legend()
        self.ax.grid(True)

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

def main():
    print("\n" + "="*70)
    print("HAND-EYE CALIBRATION")
    print("="*70 + "\n")

    samples_file = select_file_interactive(
        DATA_RESULTS_DIR,
        "collected_samples*.yaml",
        "collected samples"
    )

    chessboard_file = select_file_interactive(
        CHESSBOARD_DIR,
        "*.yaml",
        "chessboard"
    )

    samples = load_samples(str(samples_file))
    valid_indices = filter_samples(samples)

    # Load chessboard
    chessboard_corners = None
    try:
        with open(chessboard_file, "r") as f:
            chessboard_data = yaml.safe_load(f)
            chessboard_corners = chessboard_data.get('chessboard_corners')
            if chessboard_corners:
                print(f"✓ Chessboard loaded: {chessboard_file.name}\n")
    except Exception as e:
        print(f"⚠ Unable to load chessboard: {e}\n")

    print(f"Total valid samples: {len(valid_indices)}\n")
    print(f"Running {NUM_ITERATIONS} iterations with random {NUM_SAMPLES} samples each...\n")
    print(f"{'Iter':<6} {'Rot RMS (deg)':<15} {'Trans RMS (mm)':<15}")
    print("-" * 40)

    all_rot_rms = []
    all_trans_rms = []
    all_indices = []

    for i in range(NUM_ITERATIONS):
        # Random sample
        selected = random.sample(valid_indices, min(NUM_SAMPLES, len(valid_indices)))
        selected.sort()

        X = run_handeye_tsai(samples, selected)
        rot_rms, trans_rms = compute_rms(samples, selected, X)

        all_rot_rms.append(rot_rms)
        all_trans_rms.append(trans_rms)
        all_indices.append(selected)

        print(f"{i+1:<6} {rot_rms:<15.4f} {trans_rms:<15.3f}")

    print("-" * 40)
    print(f"\nSTATISTICS over {NUM_ITERATIONS} iterations:")
    print(f"  Rotation RMS:    mean={np.mean(all_rot_rms):.4f}°, std={np.std(all_rot_rms):.4f}°, "
          f"min={np.min(all_rot_rms):.4f}°, max={np.max(all_rot_rms):.4f}°")
    print(f"  Translation RMS: mean={np.mean(all_trans_rms):.3f}mm, std={np.std(all_trans_rms):.3f}mm, "
          f"min={np.min(all_trans_rms):.3f}mm, max={np.max(all_trans_rms):.3f}mm")

    # Find best iteration (minimum translation RMS)
    best_idx = np.argmin(all_trans_rms)

    # Recompute X for best indices
    X_best = run_handeye_tsai(samples, all_indices[best_idx])

    trans_norm_mm = np.linalg.norm(X_best[:3, 3]) * 1000.0
    rot_deg = np.degrees(rot_angle(X_best[:3, :3]))

    print(f"\n{'='*70}")
    print(f"BEST RESULT (minimum Translation RMS):")
    print(f"{'='*70}")
    print(f"  Iteration:       {best_idx + 1}")
    print(f"  Rotation RMS:    {all_rot_rms[best_idx]:.4f}°")
    print(f"  Translation RMS: {all_trans_rms[best_idx]:.3f}mm")
    print(f"  Indices used:    {all_indices[best_idx]}")
    print()
    print(f"Transformation X (sensor -> camera):")
    print(f"  Translation norm:  {trans_norm_mm:.3f} mm")
    print(f"  Rotation angle:    {rot_deg:.4f} deg")
    print()
    print(f"Matrix X:")
    np.set_printoptions(precision=6, suppress=True)
    print(X_best)
    print()
    quat = R.from_matrix(X_best[:3, :3]).as_quat()
    print(f"Rotation (Quaternion [x, y, z, w]):")
    print(f"  [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
    print(f"{'='*70}")

    # Launch 3D visualization
    CalibrationViewer(samples, all_indices[best_idx], X_best, chessboard_corners=chessboard_corners)

if __name__ == "__main__":
    main()
