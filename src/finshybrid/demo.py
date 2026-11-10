"""A causal synthetic API demonstration, NOT an experimental tracking result."""
import json
import numpy as np
from .predictor import Model
from .bias import BiasUpdater
from .scales import PC, VS


def main():
    model, online = Model(), BiasUpdater()
    state = np.r_[PC, np.zeros(6)]
    wrench = np.zeros(6)
    # Generate completed observations from a toy plant, with a known injected error.
    # The updater receives only observations and completed inputs, not this error.
    for _ in range(60):
        observed = model.predict(state, wrench[None])[0]
        observed[6:] += VS * np.array([.03, 0, 0, 0, 0, 0])
        online.update(model.innovation(state, wrench, observed))
        state = observed
    predicted = model.predict(state, np.zeros((8, 6)), online.b)
    print(json.dumps(dict(kind="synthetic API smoke, not a research RMSE",
                          completed_samples=online.tick, bias=online.b.tolist(),
                          next_state=predicted[0].tolist(), horizon=len(predicted)), indent=2))


if __name__ == "__main__":
    main()
