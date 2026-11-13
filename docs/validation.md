# Distribution validation

The source-only package was checked with six synthetic tests covering scalar/batched prediction parity, normalized bias units, causal interval rejection, hardware import closure, missing trajectory samples and plotting adapters. All six passed.

Every Python file was parsed and all published text was scanned for non-English CJK content. The distribution verifier checks file hashes and excludes common dataset, weight, runtime and binary formats.

No model was retrained and no experiment or vehicle was started for packaging. A training smoke run was not performed because the available packaging Python environments lacked the optional PyTorch dependency. Install the `train` extra before using the trainer. Native hardware compilation and live vehicle execution require a Linux deployment environment and are not established by the Python import checks.

Packaging changes affect external artifact locations, documentation, diagnostics and the residual-only historical loader. Numerical source kernels are retained from the recorded origins; provenance identifies their original hashes.
