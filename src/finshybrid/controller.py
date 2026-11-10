"""Causal observation bookkeeping; no hardware or simulator side effects."""
import numpy as np
from .predictor import Model
from .bias import BiasUpdater


class OnlinePredictor:
    def __init__(self, model=None):
        self.model = model if model is not None else Model()
        self.reset()

    def reset(self):
        self.previous = None
        self.online = BiasUpdater()

    def observe(self, state, completed_mean_wrench=None):
        """At t_k, input is the ESTIMATED mean wrench over [t_(k-1),t_k].

        Observations must be consecutive 100ms samples. The integration layer
        must reject/handle missing timestamps; this class does not interpolate.
        The input is NOT the command about to be optimized for [t_k,t_(k+1)].
        """
        state = self.model._state(state)
        if state.shape != (1,12): raise ValueError('Expected one observation')
        report = None
        if self.previous is not None:
            if completed_mean_wrench is None:
                raise ValueError('Completed-interval mean wrench required after first observation')
            report = self.online.update(self.model.innovation(self.previous, completed_mean_wrench, state[0]))
        self.previous = state[0].copy()
        return report

    def predict(self, future_wrenches):
        if self.previous is None: raise RuntimeError('Observe the initial state first')
        # Fix this snapshot throughout one optimizer invocation, including probes.
        return self.model.predict(self.previous, future_wrenches, self.online.b.copy())
