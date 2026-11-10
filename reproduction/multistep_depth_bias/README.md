# Multi-step depth-increment online velocity bias

The portable implementation is src/finshybrid/multistep_bias.py. Its native backend builds C++ kernels in src/finshybrid/native. This folder retains the hardware checkpoint, recorded intervals, configuration, source excerpts and earlier numerical verification.

See hardware/README.md for replay commands and coordinate/time contracts, and docs/method.md for the method description. Original source excerpts require the hardware host environment; the portable package supports standalone replay.
