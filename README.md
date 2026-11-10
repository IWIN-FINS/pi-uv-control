# PI-UV-Control

Physics-informed underwater vehicle prediction and model predictive control with a learned residual representation and causal multi-step online velocity bias adaptation.

This English-language distribution brings together the simulation and hardware algorithm code from IWIN-FINS/Underwater-Adaptive-Dynamics. The Python import name remains finshybrid. Frozen models retain their original numerical conventions.

## Install and validate

Python 3.10+ is required; Python 3.11 or 3.12 is recommended.

    git clone https://github.com/IWIN-FINS/pi-uv-control.git
    cd pi-uv-control
    python -m venv .venv
    # Linux/macOS: source .venv/bin/activate
    # Windows PowerShell: .venv/Scripts/Activate.ps1
    python -m pip install -e ".[dev,mpc,plot]"
    python -m pytest -q
    python reproduction/de101/verify.py --full-objectives
    python tools/verify_distribution.py

Install the train extra for PyTorch-based offline training. NumPy inference and bias updates run without PyTorch.

## Contents

| Directory | Purpose |
|---|---|
| src/finshybrid/ | Fossen dynamics, geometry, predictor, training, MPC, online bias, native C++ kernels |
| simulation/ | Ten training datasets and paired Fossen/DKUC/residual models, original experiment source and ten-seed aggregate results |
| hardware/ | Actual hardware checkpoint, bias configuration, 30 recorded completed intervals and replay instructions |
| reproduction/de101/ | Historical 75-episode model bundle, Unity C# source, configuration and objective fixtures |
| reproduction/multistep_depth_bias/ | Hardware-derived multi-step bias implementation and source excerpts |
| tests/, examples/, reference/ | Numerical checks, interface examples and reference kernels |
| docs/, provenance/ | Algorithm and deployment contracts, source identity and file hashes |

## Hardware replay

    python -m finshybrid.multistep_demo --model hardware/hardware_operator.json --replay hardware/hardware_intervals.json --output runs/hardware_replay.json

On Linux with g++, add --backend native --cache-dir runs/native_cache. This replays recorded intervals offline. Live execution requires the vehicle's ROS/state-estimation, actuator transport, calibrated thruster maps and supervisory integration.

## Models and scope

The simulation family preserves the DE101 deliberately mismatched Fossen parameters. Each of ten datasets trains DKUC and a residual predictor; online variants reuse the corresponding residual weights. A left-rear 300 g load changes the simulated plant at evaluation time. Simulation and hardware models are stored separately.

The ten-seed simulation tables cover Fossen, DKUC, the frozen residual and the multi-step online variant, with and without payload. The historical DE101 bundle also retains earlier methods for provenance.

A complete frozen Unity Player, third-party engine assets and live ROS/MCU transport are external dependencies. Portable prediction, replay and numerical tests run from this repository. Historical experiment scripts retain original machine paths and require environment reconstruction.

## License

Apache-2.0 for the included FinsSim-authored portions. Preserve LICENSE, NOTICE and THIRD_PARTY_NOTICES.md. Original source comments and historical names are retained. New documentation and usage instructions are in English.
