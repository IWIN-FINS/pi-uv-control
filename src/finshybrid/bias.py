"""Original 40-sample / .98 forgetting / 6-parameter velocity bias.

No online A/B/C/D update. The caller supplies causal completed-interval residuals.
"""
import numpy as np


class BiasUpdater:
    def __init__(self):
        self.reset()

    def reset(self):
        """Call at episode start, not when a disturbance switches on/off."""
        self.history = []
        self.tick = 0
        self.b = np.zeros(6)

    def update(self, innovation):
        y = np.asarray(innovation, dtype=np.float64)
        if y.shape != (6,) or not np.isfinite(y).all():
            raise ValueError("innovation must be a finite normalized velocity vector (6,)")
        self.tick += 1
        self.history = (self.history + [y.copy()])[-40:]
        values = np.asarray(self.history)
        weights = .98 ** np.arange(len(values) - 1, -1, -1)
        # Preserve DE92 normalization, even though it is mathematically unity here.
        weights *= np.sum(.98 ** np.arange(min(self.tick, 40))) / weights.sum()
        before = float(np.sum(weights[:, None] * (values - self.b) ** 2))
        after, accepted = before, False
        if self.tick >= 20:
            target = (weights @ values + .5 * self.b) / (weights.sum() + 1.5)
            candidate = self.b + .2 * (target - self.b)
            candidate *= min(1., .25 / max(np.linalg.norm(candidate), 1e-30))
            delta = candidate - self.b
            candidate = self.b + delta * min(1., .01 / max(np.linalg.norm(delta), 1e-30))
            value = float(np.sum(weights[:, None] * (values - candidate) ** 2))
            if value <= before + 1e-14:
                self.b = candidate.copy()
                accepted, after = True, value
        return dict(b=self.b.copy(), accepted=accepted, before=before, after=after,
                    samples=len(values), tick=self.tick)
