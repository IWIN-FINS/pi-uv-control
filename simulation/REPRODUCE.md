# Reproduce on another computer

## 1. Install and configure

Clone this repository and install the dev, mpc, plot extras. Add train for historical training and orchestration (PyTorch). Add psutil for process supervision.

    python -m pip install -e ".[dev,mpc,plot,train]"
    python -m pip install psutil
    cp simulation/config.example.json simulation/config.local.json

On PowerShell use Copy-Item instead of cp. Edit the paths in config.local.json. All relative paths are resolved relative to that JSON file, irrespective of your shell's working directory. Set repository to the checkout root. Set player to your external frozen FinsROVHybridBasis.exe. Spaces in paths are supported. Python subprocesses use the currently running Python interpreter.

The example places external experiment artifacts under runs/external. You may instead point each path at another existing directory. Keep the original filenames and directory layout inside each artifact root.

## 2. Portable historical DE101 reproduction (75 episodes)

The published 75-job bundle has all model exports, current tables and canonical settings. It can be checked and materialized on Windows or Linux without starting Unity:

    python -m finshybrid.simulation --config simulation/config.local.json check
    python -m finshybrid.simulation --config simulation/config.local.json prepare --output runs/de101-reproduction

Review runs/de101-reproduction/execution_plan.json. No simulator starts during prepare. To run on Windows with the verified external Player:

    python -m finshybrid.simulation --config simulation/config.local.json run --output runs/de101-reproduction --resume

The runner checks the frozen runtime manifest, runs serially with the historical CPU-affinity policy, preserves failures, and refuses automatic retries of partial output. --resume is required to use a previously prepared output directory. Changed model/runtime/plan hashes stop execution.

This is the original 75-episode protocol. It is distinct from the later ten-training-group protocol and its expanded ten-seed results.

## 3. Ten-training-group historical pipeline

The larger scripts also require the original FinsSim audit tools and historical runtime artifacts. A Git clone supplies the ten datasets, model exports and source coordinators; these external dependencies must be obtained separately:

| Config key | Required content |
|---|---|
| finssim_tools | FinsSim tools directory containing de62_runtime, de65_common, de90_flows, de84/de83 audits and their dependencies |
| collection_pool | DE65 collection queue, data audit, shared validation, build audit and exact Player |
| de101 | Original DE101 build/runtime, physical profile, repaired logging protocol |
| de100 | Fixed runtime build and numerical/physical preflight evidence |
| de92, de92_formal, offline | Earlier source contracts referenced by the audit/export scripts |
| diagnostics, unity_artifacts | Search roots for collision checks and supporting evidence |
| geometry_acceptance | Original geometry acceptance JSON |
| unity_editor_data, vcvars64 | Unity Editor Data and MSVC environment script, needed when rebuilding the multi-step adapter |
| training_run, loaded_run, pilot_run, multistep_run, paired_conditions | New output and paired-input locations |

Check the configured environment before launching collection:

    python -m finshybrid.simulation --config simulation/config.local.json check --stage historical

A nonzero exit lists missing files/modules. Do not equate this dependency check with numerical or physical acceptance. The archived pipeline performs its own frozen-source, input, objective, causal and physical audits.

Set PI_UV_CONFIG to the absolute configuration filename before invoking archived scripts directly:

    # PowerShell
    $env:PI_UV_CONFIG = (Resolve-Path simulation/config.local.json).Path
    python simulation/historical/collect.py
    python simulation/historical/pipeline.py

Collection writes to training_run. The pipeline trains and evaluates the original 400 attempts. It preserves the historical seed and method definitions, including the older bias arm. The later ten-seed aggregate tables already in results are a separate expanded evaluation. This command does not schedule those extra seeds or the added-load/new-bias experiments.

The loaded and multi-step historical branches require the corresponding original prepared input artifacts and build acceptance. They are retained for environment reconstruction. Follow their original numerical/build gates before successor.py. No command here implicitly launches a new load study.

Artifact path aliases relocate old paths inside JSON records when reading them; original files remain byte-exact. Aliases use longest-prefix matching, including dictionary keys in freeze manifests. Source hashes still apply to the relocated files. Use a fresh output root after changing source or configuration; do not reuse a frozen completed study or bypass an old hash mismatch.

## 4. Build settings

Historical Python scripts read paths through finshybrid.experiment_paths. The standalone compile_kernel.cmd accepts PI_UV_VCVARS64 and PI_UV_KERNEL_SOURCE. build.py uses vcvars64 and unity_editor_data from the JSON.

The original Unity Player and affinity implementation target Windows. Linux supports offline checks, prediction, replay and plan preparation. Running Linux closed-loop episodes requires an independently validated Linux runtime and execution adapter.

## 5. Reproducibility boundaries

All 30 group-specific exports and 10 datasets are bundled. The model coefficients, seeds, solver budgets, tolerances and causal update rules are retained. External Unity engine assets and full historical artifact trees are not included in this repository. A source-only clone supports offline verification and plan preparation; full historical collection needs the listed runtime and evidence files.
