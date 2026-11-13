# Deployment handoff

## External artifacts

Obtain the trained `operator.json`, matching nominal physical parameters, calibrated motor thrust maps, thruster allocation matrix, sensor-to-body transforms and vehicle transport implementation separately. Record their hashes with each run. This source package does not supply field calibration or trained model weights.

## State, input and timing contract

The portable predictor consumes a 12-element state: world body-origin position and orientation followed by body-frame center-of-mass linear and angular velocity. Use meters, radians, seconds, newtons and newton-meters. The source geometry uses the Unity-compatible world convention; do not pass a ROS/NED pose without an explicit frame conversion. In the simulation convention, world XZ is horizontal and depth is negative world Y. Euler ordering and center offset must follow `geometry.py` and `scales.py` exactly.

The model interval is 0.1 seconds and the nominal rollout horizon is eight intervals. The updater consumes interval start, end and availability timestamps in nanoseconds, both endpoint states, and the actual interval-mean wrench. Use completed output intervals, not the latest requested motor command. Reset adaptation and warm starts at episode boundaries; do not bridge gaps with fabricated samples.

## Hardware assembly

1. Install the package and add `deployment/hardware` to `PYTHONPATH`.
2. Construct `G1Plant(model_path, nominal, matrix)` from `g1_mpc.py`. It checks nominal physics against the supplied checkpoint.
3. Construct the thrust estimator from the vehicle calibration. `export_dynamics_training.py` provides completed-interval wrench reconstruction utilities.
4. Construct `CausalG1Bias(plant, estimator, method="multistep_depth", depth_weight=0.25)`. Check the accepted method string in the source when adapting an existing host.
5. Supply final output packets through `put_output` and timestamped state observations through `observe`. Read the wrapper's diagnostics and bias only after causal interval processing.
6. Integrate the plant with `FossenMPC` in `fossen_mpc.py` and the host's reference generator, output dispatch and runtime limits. Preserve the intended solver budget, actuator limits and hold policy.

The files contain no automatic motor connection or arming entry point. The deploying application owns communication, state quality checks, emergency stop and scheduling. The legacy default of `CausalG1Bias` is `single_step`; explicitly choose `multistep_depth` for the current method.

## Simulator integration

Use the same checkpoint, normalization and physical-state layout with the simulator's state adapter. Keep nominal prediction physics separate from the simulator's true plant parameters. Apply the first optimized control, record the actual final motor commands and retain timing/hold events. `mpc.py` exposes a portable objective and diagnostic reference solver; native simulator integration must separately verify objective and gradient parity before making runtime comparisons.

## Validation before deployment

Run the included synthetic tests, then replay completed intervals from the target host with the exact checkpoint. Verify frame conversion, thrust sign, units, sampling interval, nominal metadata and causal input availability. Check native/reference rollout agreement when selecting a native backend. Source import and synthetic tests alone do not validate a new vehicle or prove equivalence of platform-specific deadlines.

Native hardware kernels target Linux with g++. Set `FINSHYBRID_CACHE_DIR` to a writable directory outside the checkout. Set `FINSSIM_LOCAL_LOG_ROOT` to the intended recording volume. Calibration export requires explicit layout, geometry and coefficient files.
