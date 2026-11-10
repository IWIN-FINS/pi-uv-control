# Historical DE101 numerical and Unity bundle

Includes 15 models (five methods, three initializations), encoder/step fixtures, 18 native MPC objective/gradient fixtures, the 75-job plan, five current tables and FinsSim-authored Unity C# sources.

    python reproduction/de101/verify.py --full-objectives
    python reproduction/de101/run.py --help

publication_manifest.json preserves original bundle hashes. Unity execution requires the external frozen Windows Player verified by config/runtime_manifest.json. Historical scripts retain original machine paths. See simulation/README.md for the later ten-group study.
