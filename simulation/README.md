# Simulation experiments

See [REPRODUCE.md](REPRODUCE.md) for configurable paths, preflight checks, and Windows execution commands.

## Data and model pairing

datasets/g01 through g10 are independent training groups. Each group has three random current realizations, collected with four fixed MPC prior configurations: matched, inertia mismatch, damping mismatch, and combined inertia/damping mismatch. The simulated plant parameters remain fixed during collection.

Each 25.6 s joint figure-eight rollout records physics at 50 Hz. Completed 100 ms intervals give 255 transitions per rollout, 12 rollouts and 3,060 transitions per group. Files include simulator pose, body velocity, mean applied wrench estimates and segment identity. Eight-step windows stay within segment boundaries.

The horizontal Gauss-Markov current has component standard deviation 0.02 m/s, time constant 8 s and speed cap 0.15 m/s. Four policies share each current seed. Training seeds span 3117001 through 3117030. The common independent validation dataset has 1,020 transitions in ../data/validation.npz.

models/gNN/{F,Kx,H} contains native model exports. Kx is DKUC in the paper. H is the residual predictor; online variants share H weights. F is physics-only in the same export format. Load these exports with finshybrid.de101.DE101Model.

Each group trains Kx and H with initialization 281, 1,200 Adam updates, batch size 128 and validation-based checkpoint selection. The 300 g payload is added only at evaluation. No payload-specific retraining is performed.

## Running and reproduction

- Run python reproduction/de101/verify.py --full-objectives for the historical 15-model numerical bundle.
- Run python -m finshybrid.train --help for portable residual training.
- See reproduction/de101/run.py --help for the historical Unity coordinator. It requires the external frozen Windows Player specified by its runtime manifest.
- historical/ retains the original ten-group collection/training/forecast orchestration. These scripts use PI_UV_CONFIG for paths and require the external runtime/audit artifacts listed in REPRODUCE.md.
- historical/multistep/ retains the later Python/C# Unity bias bridge.
- results/ contains aggregate tracking and prediction tables for the later ten-seed study.

The historical 75-episode bundle and the later ten-group experiment are separate protocols. Timing depends on host, CPU affinity, concurrency and deadline handling.
