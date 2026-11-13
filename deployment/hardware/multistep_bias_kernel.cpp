// Batch independent states AND biases using the unchanged hardware predictor.
// This library is compiled only by the offline multi-step bias experiment.
#include "fossen_mpc_kernel.cpp"

extern "C" int multistep_bias_rollout(std::size_t count, std::size_t horizon,
    const double* initial, const double* wrenches, const double* params,
    const double* initial_latent, const double* A, const double* B,
    const double* C, const double* D, const double* biases,
    const double* us, const double* vs, double* result) {
    for (std::size_t i=0; i<count; ++i) {
        int status=g1_rollout(1, horizon, initial+12*i,
            wrenches+6*horizon*i, params, initial_latent+10*i,
            A, B, C, D, biases+6*i, us, vs, result+12*horizon*i);
        if(status) return status;
    }
    return 0;
}
