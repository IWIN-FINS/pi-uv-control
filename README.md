# PI-UV Control

Source-only implementation of the Fossen-based residual predictor, Koopman latent propagation, multi-step online velocity-bias adaptation, and deployment interfaces.

## Contents

| Path | Purpose |
| --- | --- |
| `src/finshybrid/` | Encoder, physics, training/export, prediction, MPC objective and online adaptation |
| `src/finshybrid/native/` | Native rollout kernels |
| `deployment/hardware/` | Hardware G1 plant, causal bias wrapper, MPC runtime, thrust reconstruction and log utilities |
| `docs/method.md` | Model and adaptation equations |
| `docs/deployment.md` | Integration contract and handoff instructions |
| `tests/` | Source-only synthetic checks |
| `provenance/` | Source provenance and hashes |

This release excludes datasets, trained weights, recorded trajectories, results, binaries, virtual environments, baseline controllers and ablation experiment queues. Historical Git commits are retained; use the current tree or release ZIP for the source-only distribution.

## Install

Python 3.10 or later is required. From this directory:

```bash
python -m pip install -e ".[train,mpc,plot,hardware,dev]"
python -m pytest -q
```

The native backend additionally requires a C++ compiler. The hardware adapters are supplied as source modules; add `deployment/hardware` to the application module search path. ROS, vehicle communication and simulator engines remain host-side dependencies.

## Train and export

Supply separate training and validation NPZ files containing normalized legacy features `x`, `y` of shape `(N,42)`, normalized wrench `u` of shape `(N,6)`, and contiguous episode identifiers `seg`. Windows may not cross episode boundaries. Dataset collection and splitting must be performed before invoking this trainer.

```bash
python -m finshybrid.train --template src/finshybrid/config/reference_metadata.json --train /path/to/train.npz --validation /path/to/validation.npz --output /path/to/new-run --seed 281 --steps 1200 --device cpu
```

The included metadata is an untrained historical reference configuration, not a vehicle calibration. Supply a matching nominal-physics template for a different vehicle. The command writes new operator weights and training diagnostics outside the repository. `--device cuda --cuda-graph` enables the inherited CUDA training implementation. Set the training budget explicitly for the intended protocol; the CLI default is not a claim about every published experiment.

## Predict and adapt

Pass a trained checkpoint explicitly to `finshybrid.predictor.Model(path)`, or set `FINSHYBRID_MODEL`. No default trained checkpoint is shipped.

```bash
python -m finshybrid.multistep_demo --model /path/to/operator.json --backend numpy --output /path/to/interface-check.json
```

This command generates an interface demonstration from the supplied model. It is not a closed-loop experiment. A recorded completed-interval replay can be supplied using `--replay`. See [deployment instructions](docs/deployment.md) before integrating with a vehicle.

The active multi-step updater is `MultiStepBiasUpdater`. The earlier `BiasUpdater`, `OnlinePredictor` and `ShadowCycle` are retained as low-level compatibility interfaces; they do not implicitly select the multi-step update. Select the multi-step method explicitly in the hardware wrapper.

## Scope of deployment

The package contains numerical model and controller integration source. It does not contain Unity scenes, ROS transport bindings, motor firmware, calibrated thrust tables or a ready-to-arm vehicle application. The host application supplies state estimates, reference trajectories, calibration, actuator allocation, timing and motor dispatch. See the handoff contract for each input and the required timestamp ordering.
