# Predictor and online adaptation

Fossen dynamics advance body velocity. A neural encoder maps measured depth, attitude and body velocity into ten learned features. Fixed A/B matrices advance the learned state; C/D matrices read out six velocity corrections. Corrected velocity enters pose integration in the same step, including center-of-mass/body-origin geometry.

Model() defaults to historical DE83 package assets. Use DE101Model for simulation native exports and Model(path) for the hardware operator. Checkpoints include nominal physics and normalization statistics.

MultiStepBiasUpdater fits completed historical windows with velocity, depth-increment and attitude residuals. The bias is constant throughout each candidate rollout. The target depth increments come from measured completed transitions. Magnitude/change penalties, norm limits, backtracking and acceptance guards constrain updates.

The configured history is 40 completed intervals, warmup 20, at most eight windows, eight steps, supervision at steps 1/5/8 and forgetting 0.98. Velocity/depth/attitude weights are 1/0.25/0.2. See hardware/config.json and implementation docstrings for exact scales and guards.

## Metrics

Tracking RMSE compares actual position with reference during closed-loop operation. Prediction RMSE compares a model rollout with future recorded state on a common Fossen-controlled trajectory, using recorded future inputs for offline conditional prediction. Bias uses only prior completed intervals. Horizontal errors use squared Euclidean distance in world XZ. Keep failures and invalid counts in reported denominators.
