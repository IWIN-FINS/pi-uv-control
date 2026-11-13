#!/usr/bin/env python3
"""Export quality-filtered 10/20 Hz Unity/COM training candidates from passive captures.

Force labels are estimates. This never sends commands or modifies an input bag.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from dynamics_thrust import C, ThrustEstimator, sha256

# Same float32 -> float64 constants as Underwater-Adaptive-Dynamics/scales.py.
PC, PS, VS, US = [np.array(v, dtype=np.float32).astype(np.float64) for v in (
    [0, -1.2, 0, 0, 0, 0], [.6, .2, .35, .2, .4, .2],
    [.15, .08, .15, .25, .4, .25], [15, 12, 15, 2, 2, 2])]


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def timed_rows(path, columns):
    rows = read_csv(path)
    times = np.array([int(r['stamp_ns']) for r in rows], dtype=np.int64)
    values = np.array([[float(r[c]) for c in columns] for r in rows], dtype=float)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError(f'{path.name}: timestamps must strictly increase, at least two samples required')
    if not np.isfinite(values).all():
        raise ValueError(f'{path.name}: nonfinite values')
    return times, values


def interpolate(times, values, stamp, max_gap_ns=100_000_000, quaternion=False):
    right = int(np.searchsorted(times, stamp, side='right'))
    left = right - 1
    if left < 0 or right >= len(times):
        raise ValueError('state_outside_recording')
    gap = int(times[right] - times[left])
    if gap > max_gap_ns:
        raise ValueError('state_gap')
    f = (stamp - int(times[left])) / gap
    a, b = values[left], values[right]
    if quaternion:
        if np.linalg.norm(a) < .5 or np.linalg.norm(b) < .5:
            raise ValueError('invalid_quaternion')
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        b = b if np.dot(a, b) >= 0 else -b
    value = a * (1-f) + b * f
    if quaternion:
        value /= np.linalg.norm(value)
    return value


def quaternion_matrix(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def unity_euler(rotation):
    # Unity Ry(y) Rx(x) Rz(z), matching finshybrid.geometry.euler.
    a = np.arcsin(np.clip(-rotation[1, 2], -1, 1))
    if abs(np.cos(a)) < .1:
        raise ValueError('euler_chart_singularity')
    return np.array([a, np.arctan2(rotation[0, 2], rotation[2, 2]),
                     np.arctan2(rotation[1, 0], rotation[1, 1])])


def interval_mean(start, end, snapshots, estimates, times=None):
    if times is None:
        times = np.array([s['mapped_snapshot_time_ns'] for s in snapshots], dtype=np.int64)
    first = int(np.searchsorted(times, start, side='right')) - 1
    last = int(np.searchsorted(times, end, side='left'))
    if first < 0 or last >= len(times):
        raise ValueError('actuator_outside_recording')
    integral = np.zeros(6)
    segment = snapshots[first]['clock_segment']
    for i in range(first, last):
        a, b = snapshots[i], snapshots[i+1]
        if (a['clock_segment'] != segment or b['clock_segment'] != segment or
                ((b['mcu_sequence'] - a['mcu_sequence']) & 0xffff) != 1):
            raise ValueError('actuator_sequence_or_clock_gap')
        if not 0 < times[i+1] - times[i] <= 75_000_000:
            raise ValueError('actuator_time_gap')
        if (a['valid_flags'] & 31) != 31 or max(a['output_age_ms']) > 50:
            raise ValueError('actuator_or_sensor_stale')
        if not estimates[i]['valid']:
            raise ValueError('throttle_outside_calibration')
        width = max(0, min(end, int(times[i+1])) - max(start, int(times[i])))
        integral += width * np.array(estimates[i]['wrench_body_com_unity'])
    return integral / (end-start)


class FrameQuality:
    """Require matching state publications; never interpolate across a fault."""
    def __init__(self, flags, *state_times):
        self.flags = flags
        self.times = np.array([int(row['state_stamp_ns']) for row in flags], dtype=np.int64)
        if len(self.times)<2 or np.any(np.diff(self.times)<=0):
            raise ValueError('State status timestamps must strictly increase')
        if any(not np.array_equal(self.times, times) for times in state_times):
            raise ValueError('State publications do not have identical frame timestamps')
        self.good = np.array([all(row[k]=='1' for k in
            ('ready','vision_fresh','imu_fresh','depth_fresh')) and row['vision_mode']=='fresh'
            for row in flags])

    def require(self, start, end, max_gap_ns=100_000_000):
        left = int(np.searchsorted(self.times,start,side='right'))-1
        right = int(np.searchsorted(self.times,end,side='left'))
        if left<0 or right>=len(self.times):raise ValueError('state_outside_recording')
        if not self.good[left:right+1].all():raise ValueError('state_not_fresh')
        if np.any(np.diff(self.times[left:right+1])>max_gap_ns):raise ValueError('state_gap')


def export(capture: Path, output: Path, allow_stationary=False, *, sample_hz=10):
    if isinstance(sample_hz,bool) or sample_hz not in (10,20):raise ValueError('Training rate must be 10 or 20 Hz')
    dt_ns=1_000_000_000//sample_hz
    dt_seconds=dt_ns/1e9
    # Always use the configuration/model actually frozen with this capture.
    capture_manifest = json.loads((capture/'manifest.json').read_text())
    selection=None
    if capture_manifest.get('training_selection') in ('controller_mpc_windows','manual_pid_windows'):
        from collection_recording import TrainingWindows
        manual=capture_manifest['training_selection']=='manual_pid_windows'
        selection=TrainingWindows(json.loads((capture/'training_windows.json').read_text()),
            reason='outside_continuous_manual_PID' if manual else 'outside_continuous_Fossen_MPC')
    sources = capture / 'sources'
    model_path = sources / 'ros2_ws/src/hardware_bridge/config/pid_thrust_calibration.json'
    model = json.loads(model_path.read_text())
    estimator = ThrustEstimator(model)
    pool_path = sources / 'ros2_ws/src/state_estimation/config/pool_world.yaml'
    pool = yaml.safe_load(pool_path.read_text())
    surface = float(pool['/**']['ros__parameters']['water_surface_z_m'])
    pt, pv = timed_rows(capture/'pose.csv', ('x_m','y_m','z_m','qx','qy','qz','qw'))
    vt, vv = timed_rows(capture/'velocity_body.csv', ('vx_mps','vy_mps','vz_mps'))
    wt, wv = timed_rows(capture/'imu_rates.csv', ('wx_rps','wy_rps','wz_rps'))
    flags = read_csv(capture/'state_flags.csv')
    reprocessing = capture_manifest.get('reprocessing')
    synchronized = bool(flags and 'state_stamp_ns' in flags[0])
    if reprocessing and not synchronized:raise ValueError('Reprocessed data requires exact frame status')
    if sample_hz==20 and not synchronized:
        raise ValueError('20 Hz export requires exact source-stamped fused pose/velocity/IMU/status frames')
    quality = FrameQuality(flags,pt,vt,wt) if synchronized else None
    ft = quality.times if quality else np.array([int(r['host_receive_ns']) for r in flags], dtype=np.int64)
    snapshots = [json.loads(line) for line in (capture/'actuator_snapshots.jsonl').read_text().splitlines()]
    if len(snapshots) < 2:
        raise ValueError('No usable actuator recording')
    at = np.array([s['mapped_snapshot_time_ns'] for s in snapshots], dtype=np.int64)
    if np.any(np.diff(at) <= 0):
        raise ValueError('Mapped actuator time is nonmonotonic; split recording at clock resets')
    estimates = [estimator.estimate(s['motor_throttle'], s['motor_enabled_mask']) for s in snapshots]
    com = np.asarray(model['center_of_mass_from_body_origin_ros_m'])

    def state(stamp):
        index = int(np.searchsorted(ft, stamp, side='right')) - 1
        if index < 0 or stamp - int(ft[index]) > dt_ns:
            raise ValueError('state_status_stale')
        flag = flags[index]
        if any(flag[k] != '1' for k in ('ready','vision_fresh','imu_fresh','depth_fresh')) or flag['vision_mode'] != 'fresh':
            raise ValueError('state_not_fresh')
        position = interpolate(pt, pv[:, :3], stamp,max_gap_ns=dt_ns)
        q = interpolate(pt, pv[:, 3:], stamp,max_gap_ns=dt_ns,quaternion=True)
        rotation = C @ quaternion_matrix(q) @ C.T
        angles = unity_euler(rotation)
        velocity_origin = interpolate(vt, vv, stamp,max_gap_ns=dt_ns)
        omega = interpolate(wt, wv, stamp,max_gap_ns=dt_ns)
        velocity_com = velocity_origin + np.cross(omega, com)
        pose = np.r_[C @ (position - [0., 0., surface]), angles]
        velocity = np.r_[C @ velocity_com, -C @ omega]
        features = np.zeros(42)
        features[0], features[1:7], features[7:13] = 1., (pose-PC)/PS, velocity/VS
        return features, pose, velocity

    start = max(int(pt[0]), int(vt[0]), int(wt[0]), int(at[0]))
    end = min(int(pt[-1]), int(vt[-1]), int(wt[-1]), int(at[-1]))
    start = (start // dt_ns + 1) * dt_ns
    xs, ys, inputs, segments, poses, velocities, next_poses, next_velocities, stamps = ([] for _ in range(9))
    rejected = Counter(); segment = 0; previous_end = None;previous_selection=None
    for stamp in range(start, end-dt_ns, dt_ns):
        try:
            selected=selection.require(stamp,stamp+dt_ns) if selection else None
            if quality:quality.require(stamp,stamp+dt_ns,max_gap_ns=dt_ns)
            x, p, v = state(stamp)
            y, p1, v1 = state(stamp + dt_ns)
            u = interval_mean(stamp, stamp + dt_ns, snapshots, estimates, at)
        except ValueError as error:
            rejected[str(error)] += 1
            previous_end = None
            continue
        if previous_end != stamp or selected!=previous_selection:
            segment += 1
        xs.append(x); ys.append(y); inputs.append(u/US); segments.append(segment)
        poses.append(p); velocities.append(v); next_poses.append(p1); next_velocities.append(v1); stamps.append(stamp)
        previous_end = stamp+dt_ns
        previous_selection=selected
    if len(xs) < 8:
        raise ValueError(f'Fewer than eight accepted transitions; rejected={dict(rejected)}')
    excited = bool(np.any(np.abs(inputs) > 1e-5))
    if not excited and not allow_stationary:
        raise ValueError('Stationary zero-output recording: use --allow-stationary only for pipeline validation')
    if not any(np.count_nonzero(np.array(segments) == s) >= 8 for s in set(segments)):
        raise ValueError('No continuous eight-step training window')
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'transitions.npz', x=np.array(xs), y=np.array(ys), u=np.array(inputs),
                        seg=np.array(segments), pose=np.array(poses), velocity=np.array(velocities),
                        next_pose=np.array(next_poses), next_velocity=np.array(next_velocities),
                        timestamp_ns=np.array(stamps, dtype=np.int64), dt=np.array(dt_seconds))
    metadata = {
        'capture_path': str(capture.resolve()), 'split': capture_manifest.get('split', 'diagnostic'),
        'supervision_state_source': 'EKF_estimate_not_independent_ground_truth',
        'state_status_alignment': 'exact_fused_publication_stamp' if quality else 'legacy_host_receive_time',
        'training_selection':capture_manifest.get('training_selection','fresh_state_quality'),
        'training_windows_sha256':sha256(capture/'training_windows.json') if selection else None,
        'angular_velocity_source': 'same_fused_publication' if quality else 'legacy_raw_IMU_interpolation',
        'reprocessing': reprocessing,
        'source_capture_manifest_sha256': capture_manifest.get('source_capture_manifest_sha256',
                                                               sha256(capture/'manifest.json')),
        'frozen_source_sha256': capture_manifest['source_sha256'],
        'exporter_code_sha256': sha256(Path(__file__)),
        'estimator_code_sha256': sha256(Path(__file__).with_name('dynamics_thrust.py')),
        'transitions': len(xs), 'segments': segment, 'rejected_intervals': dict(rejected),
        'purpose': 'motion_training_candidate' if excited else 'stationary_pipeline_validation_only',
        'has_nonzero_estimated_input': excited, 'training_quality_physically_validated': False,
        'force_kind': model['kind'], 'force_assumptions': model['assumptions'],
        'state_frame': 'Unity_x_forward_y_up_z_left;_world_origin_at_pool_water_surface',
        'position_reference': 'body_origin', 'linear_velocity_reference': 'body_center_of_mass',
        'angular_transform': '-C_times_ROS_omega;_wrench_moments_use_same_axial_transform',
        'dt_seconds':dt_seconds,'sample_rate_hz':sample_hz,
        'maximum_state_interpolation_gap_ns':dt_ns,
        'required_model_sample_dt_seconds':dt_seconds,
        'input_average': 'zero_order_hold_integral_of_estimated_wrench_over_completed_interval',
        'unused_legacy_features': 'columns_13_to_41_zero_unused_by_G1',
        'calibration_sha256': sha256(model_path), 'pool_world_sha256': sha256(pool_path),
        'capture_manifest_sha256': sha256(capture/'manifest.json'),
        'npz_sha256': sha256(output/'transitions.npz'),
        'remaining_physical_validation': ['underwater_metric_position_velocity_and_world_yaw',
                                          'moving_vehicle_force_estimate_and_motor_response'],
    }
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-stationary', action='store_true')
    parser.add_argument('--training-hz',type=int,choices=(10,20),default=10)
    args = parser.parse_args()
    print(json.dumps(export(args.capture,args.output,args.allow_stationary,sample_hz=args.training_hz),indent=2))


if __name__ == '__main__':
    main()
