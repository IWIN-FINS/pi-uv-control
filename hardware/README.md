# Hardware checkpoint and replay

hardware_operator.json is the frozen hardware residual checkpoint supplied with the multi-step bias implementation. hardware_intervals.json holds 30 completed 100 ms intervals and identifies the required checkpoint by SHA-256. config.json records estimator parameters. This JSON is a parameter record; the portable implementation supplies defaults in code.

    python -m finshybrid.multistep_demo --model hardware/hardware_operator.json --replay hardware/hardware_intervals.json --output runs/hardware_replay.json

## Integration contract

- State: 12 SI components, body-origin world position, Unity Euler angles, body-frame center-of-mass linear velocity, body angular velocity. World Y points up; depth is minus Y.
- Input: six-dimensional mean body-frame force/torque over a completed 100 ms interval, reconstructed from final actuator outputs, calibrated motor maps and thruster geometry.
- Timestamps: integer nanoseconds in one clock. Both endpoint observations and interval inputs must be available before updating.
- Hardware MPC: 20 Hz replanning, eight-step horizon with 100 ms model steps. Bias fitting consumes completed 100 ms intervals.
- Encoder and learned matrices stay frozen. Fit only the six-dimensional normalized bias; copy it once before each MPC solve.
- Reset history and bias at episode start. Handle estimator gaps and visual fallback explicitly.

The deployment model uses 18,715 training transitions from 43 segments, 1,690 validation transitions, 1,383 test transitions, 4,800 updates and initialization 281.

Original g1_mpc.py and multistep_bias.py excerpts are in reproduction/multistep_depth_bias/source. They depend on original hardware host modules. The portable updater and this replay run independently.

A full live-vehicle launch additionally requires the vehicle ROS workspace, estimator, MCU bridge, motor channel mappings, calibration and supervisor. The code here supplies algorithms, C++ numerical kernels, models and interfaces. Validate coordinate conversion, timestamps, input reconstruction and actuation on the target platform.
