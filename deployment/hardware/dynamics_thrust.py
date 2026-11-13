#!/usr/bin/env python3
"""Offline calibration and force reconstruction. Never opens a hardware transport."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
# Polar vectors: Unity x-forward/y-up/z-left -> ROS x-forward/y-left/z-up.
# det=-1; moments and angular velocities therefore require -C, not C.
C = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def monotone(values, weights):
    """Weighted pool-adjacent-violators fit, without an extra dependency."""
    blocks = []
    for i, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([i, i + 1, float(value), float(weight)])
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
            right, left = blocks.pop(), blocks.pop()
            weight = left[3] + right[3]
            blocks.append([left[0], right[1],
                           (left[2] * left[3] + right[2] * right[3]) / weight, weight])
    result = np.empty(len(values))
    for start, end, value, _ in blocks:
        result[start:end] = value
    return result


def build_calibration(data: Path, layout_path: Path, geometry_path: Path, coefficients_path: Path):
    layout = yaml.safe_load(layout_path.read_text())
    geometry = yaml.safe_load(geometry_path.read_text())
    coefficients = json.loads(coefficients_path.read_text())['identified_groups']
    scale = float(layout['curve_reuse']['historical_final_throttle_scale'])
    if not layout['curve_reuse']['per_channel_branch_assignment_confirmed']:
        raise ValueError('Rotation-branch assignment is not confirmed')
    sources = [layout_path, geometry_path, coefficients_path]
    curves = {}
    for group, ids in {'normal': (5, 7), 'reverse': (6, 8)}.items():
        curves[group] = {}
        for side, sign in [('positive', 1), ('negative', -1)]:
            # Equal weight per motor/bin prevents the longer sweeps dominating.
            bins = [[] for _ in range(25)]
            accepted = {}
            for motor in ids:
                path = data / f'M00{motor}_rpm_sweep/raw.csv'
                if path not in sources:
                    sources.append(path)
                motor_bins = [[] for _ in bins]
                count = 0
                with path.open(newline='') as stream:
                    for row in csv.DictReader(stream):
                        if row['phase'] not in ('ramp_negative_to_positive', 'ramp_positive_to_negative'):
                            continue
                        if row.get('sensor_ok') != '1' or not all(row.get(k) for k in ('target_command', 'target_rpm', 'rpm_age_sec')):
                            continue
                        u, rpm, age = (float(row[k]) for k in ('target_command', 'target_rpm', 'rpm_age_sec'))
                        if not np.isfinite([u, rpm, age]).all() or not 0 <= age <= .2:
                            continue
                        # Reject reversal transients whose RPM still has the old sign.
                        if not 0 < sign * u <= 1 or rpm * u <= 0:
                            continue
                        motor_bins[min(24, int(abs(u) * 25))].append((abs(u), abs(rpm)))
                        count += 1
                accepted[f'M00{motor}'] = count
                for index, points in enumerate(motor_bins):
                    if len(points) >= 3:
                        bins[index].append(np.median(points, axis=0))
            points = [np.mean(points, axis=0) for points in bins if points]
            if len(points) < 10:
                raise ValueError(f'Insufficient fresh RPM support for {group}/{side}')
            x, rpm = np.asarray(points).T
            order = np.argsort(x)
            x, rpm = x[order], monotone(rpm[order], np.ones(len(rpm)))
            c = float(coefficients[f'{group}_propeller'][f'c1_{side}_omega'])
            curves[group][side] = {
                'command_abs': [0., *x.tolist()], 'rpm_abs': [0., *rpm.tolist()],
                'c_omega': c, 'accepted_rows': accepted,
                'max_supported_final_throttle': float(x[-1] * scale),
            }
    com = np.asarray(geometry['reference_point'], dtype=float)
    geometry_by_name = {g['name']: g for g in geometry['thrusters']}
    channels = []
    branch_signs = layout['curve_reuse']['positive_throttle_rotation_sign']
    for entry, branch_sign in zip(layout['channels'], branch_signs):
        g = geometry_by_name[entry['name']]
        position = C @ (np.asarray(g['position']) - com)
        axis = C @ np.asarray(g['direction'], dtype=float)
        axis /= np.linalg.norm(axis)
        if entry['name'].startswith('V_'):
            direction = -1 if entry['positive_observation'] == 'down' else 1
            axis *= direction
        else:
            yaw_sign = 1 if entry['positive_observation'] == 'yaw_ccw' else -1
            axis *= yaw_sign * np.sign(np.cross(position, axis)[2])
        channels.append({'channel': entry['channel'], 'name': entry['name'],
                         'propeller_type': entry['propeller_type'],
                         'positive_throttle_rotation_sign': branch_sign,
                         'position_from_com_ros_m': position.tolist(),
                         'positive_throttle_axis_ros': axis.tolist()})
    return {
        'schema_version': 1, 'kind': 'estimated_static_bollard_thrust',
        'historical_final_throttle_scale': scale, 'curves': curves, 'channels': channels,
        'center_of_mass_from_body_origin_ros_m': (C @ com).tolist(),
        'source_sha256': {str(p.resolve()): sha256(p) for p in sources},
        'assumptions': [
            'historical_bridge_output_scale_1_and_mcu_throttle_scale_0.4_defaults',
            'dataset_command_sign_matches_published_rotation_branch',
            'same_platform_geometry_reused;_horizontal_axes_inferred_from_yaw',
            'vertical_thrusters_share_same_propeller_group_curves',
            ('operator_confirmed_same_supply_voltage_numeric_value_not_recorded'
             if layout['curve_reuse'].get('supply_voltage_matches_calibration') else
             'supply_voltage_not_matched_or_compensated'),
            'quasistatic_estimate_no_motor_lag_inflow_or_thruster_interaction_model',
        ],
    }


class ThrustEstimator:
    def __init__(self, model):
        self.model = model
        self.scale = model['historical_final_throttle_scale']
        self.axes = np.array([c['positive_throttle_axis_ros'] for c in model['channels']])
        self.positions = np.array([c['position_from_com_ros_m'] for c in model['channels']])
        self.matrix = np.concatenate((self.axes, np.cross(self.positions, self.axes)), axis=1).T
        if np.linalg.matrix_rank(self.matrix) != 6:
            raise ValueError('Thruster geometry must span six body wrench axes')

    def estimate(self, throttle, enabled_mask):
        t = np.asarray(throttle, dtype=float)
        if t.shape != (8,) or not np.isfinite(t).all() or np.any(np.abs(t) > 1):
            raise ValueError('Expected eight finite final throttle values in [-1,1]')
        if not 0 <= enabled_mask <= 255:
            raise ValueError('Invalid driver enable mask')
        if any(t[i] != 0 and not enabled_mask & (1 << i) for i in range(8)):
            raise ValueError('Nonzero final output behind a disabled software gate')
        forces, rpms, valid, reasons = [], [], True, []
        for i, channel in enumerate(self.model['channels']):
            if t[i] == 0:
                forces.append(0.); rpms.append(0.); continue
            sign = np.sign(t[i])
            rotation_sign = int(sign * channel['positive_throttle_rotation_sign'])
            side = 'positive' if rotation_sign > 0 else 'negative'
            curve = self.model['curves'][channel['propeller_type']][side]
            u = abs(t[i]) / self.scale
            # No extrapolation and no silent clipping outside the measured range.
            if u > curve['command_abs'][-1]:
                forces.append(None); rpms.append(None); valid = False
                reasons.append(f'channel_{i+1}_outside_calibration_range'); continue
            rpm = float(np.interp(u, curve['command_abs'], curve['rpm_abs']))
            force = curve['c_omega'] * (rpm * 2 * np.pi / 60) ** 2
            forces.append(float(sign * force)); rpms.append(rotation_sign * rpm)
        ros = self.matrix @ np.array(forces) if valid else None
        unity = np.r_[C @ ros[:3], -C @ ros[3:]] if valid else None
        return {'valid': valid, 'reasons': reasons, 'force_along_positive_throttle_axis_n': forces,
                'estimated_rotation_rpm': rpms, 'wrench_body_com_ros': None if ros is None else ros.tolist(),
                'wrench_body_com_unity': None if unity is None else unity.tolist(),
                'force_source': 'estimated_from_historical_curves_not_measured'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--layout', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--coefficients', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model = build_calibration(args.data, args.layout, args.geometry, args.coefficients)
    ThrustEstimator(model)
    with args.output.open('x') as stream:
        json.dump(model, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'output': str(args.output), 'channels': len(model['channels']), 'kind': model['kind']}))


if __name__ == '__main__':
    main()
