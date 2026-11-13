# Main method

The predictor combines a frozen nominal Fossen velocity step with a learned residual. An encoder initializes a latent state once at the start of a prediction window. Latent dynamics then propagate linearly:

\[
z_{j+1}=Az_j+Bu_j,\qquad
\hat\nu_{j+1}=F_\nu(\hat x_j,u_j)+S_\nu(Cz_j+Du_j+b_k).
\]

The G1 encoder uses 10 input features, fully connected layers of widths 32, 16 and 10, ReLU activations, a scaled input skip connection and layer normalization. The skip connection inside the encoder is distinct from adding the learned velocity residual to nominal physics.

Pose is propagated using the geometry kernels and the body-origin / center-of-mass offset. State and wrench normalization are defined in `scales.py`. The exported 46-element vector is a legacy storage layout, not 46 learned latent states.

## Online update

`MultiStepBiasUpdater` adjusts six normalized velocity-bias components from completed historical intervals. It evaluates candidate biases through multi-step rollouts, using velocity, depth-increment and regularization terms. A candidate remains fixed throughout each prediction window. The physical parameters, encoder and Koopman matrices remain frozen.

The implementation retains a bounded history and supervises selected rollout horizons. Read the constructor defaults and deployment configuration together: changing the update history, weights or interval timing defines a different experiment. The hardware wrapper reconstructs interval-mean wrench from final motor output, thrust calibration and thruster geometry.

No future measured state or uncompleted motor interval may enter an update. An offline forecast may replay recorded future inputs only as conditional prediction inputs; that does not authorize using future observations in online adaptation.

## Implementations

`predictor.py` implements the portable G1/A export contract. `de101.py` supplies an explicit historical export adapter. `mpc.py` contains the objective and a SciPy reference solve. The reference solve is a diagnostic implementation and is not a claim of timing equivalence to a native simulator solver. `deployment/hardware` contains the hardware plant and controller runtime used for integration.
