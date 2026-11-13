"""Portable API around the frozen G1/A model. All external states use SI units."""
from pathlib import Path
import json
import os
import numpy as np
from . import encoder
from . import physics
from .geometry import rotation, start_pose_average, euler
from .scales import PC, PS, VS, US, CENTER


class Model:
    """Physical state = [world origin pose6, body COM velocity6]; latent = 10.

    The 46-element legacy adapter is an export layout, not 46 learned states.
    Loading and inference do not require PyTorch, Unity, ROS, or local drive paths.
    """

    def __init__(self, path=None, seed=281):
        path = path or os.environ.get("FINSHYBRID_MODEL")
        if not path:
            raise ValueError("Supply a model path or set FINSHYBRID_MODEL; weights are distributed separately")
        path = Path(path)
        self.raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if self.raw.get('experiment') == 'DE101' and type(self) is Model:
            raise ValueError('DE101 checkpoints require finshybrid.de101.DE101Model; do not use the DE83 loader')
        if self.raw.get("de83Mode") != "A" or self.raw.get("latent") != 10:
            raise ValueError("Expected a DE83 G1/A (10 latent) checkpoint")
        if self.raw.get("inputColumns") != encoder.BASIC.tolist():
            raise ValueError("G1 input columns do not match the frozen contract")
        self.meta, self.weights, self.operator, self.a, self.b = encoder.unpack(self.raw)
        self.A = self.a[36:, 36:]
        self.B = self.b[36:]
        self.C = self.a[7:13, 36:]
        self.D = self.b[7:13]
        for array in [self.a, self.b, self.operator, *self.meta.values(), *self.weights]:
            array.setflags(write=False)
        for array in [self.A, self.B, self.C, self.D]:
            array.setflags(write=False)

    @staticmethod
    def _state(state):
        x = np.asarray(state, dtype=np.float64)
        if x.ndim not in (1, 2) or x.shape[-1] != 12 or not np.isfinite(x).all():
            raise ValueError("state must be finite (...,12): pose6 followed by body COM velocity6")
        return np.atleast_2d(x)

    def lift(self, state):
        s = self._state(state)
        z = np.zeros((len(s), 46), dtype=np.float64)
        z[:, 0] = 1
        z[:, 1:7] = (s[:, :6] - PC) / PS
        z[:, 7:13] = s[:, 6:] / VS
        z[:, 36:] = encoder.encode(z, self.meta, self.weights, "G1")
        return z

    @staticmethod
    def physical(z):
        return np.concatenate((z[..., 1:7] * PS + PC, z[..., 7:13] * VS), axis=-1)

    def step(self, z, normalized_wrench, bias=None):
        """Advance once, using C*z_current, not C*z_next. Inputs are normalized."""
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        u = np.atleast_2d(np.asarray(normalized_wrench, dtype=np.float64))
        bias = np.zeros(6) if bias is None else np.asarray(bias, dtype=np.float64)
        if z.shape[1] != 46 or u.shape != (len(z), 6) or bias.shape != (6,):
            raise ValueError("Expected z=(batch,46), u=(batch,6), bias=(6,)")
        if not all(np.isfinite(v).all() for v in (z, u, bias)):
            raise ValueError("Nonfinite model input")
        out = z @ self.a.T + u @ self.b.T
        out[:, 7:13] += physics.nominal(z, u, self.raw["nominal"]) + bias
        p = z[:, 1:7] * PS + PC
        pf = start_pose_average(p, z[:, 7:13] * VS, out[:, 7:13] * VS)
        pos = pf[:, :3] - np.einsum("nij,j->ni", rotation(pf) - rotation(p), CENTER)
        pose = np.concatenate((pos, euler(rotation(pf))), axis=-1)
        out[:, 1:7] = (pose - PC) / PS
        if not np.isfinite(out).all():
            raise FloatingPointError("Nonfinite prediction; do not dispatch to actuators")
        return out

    def predict(self, state, wrench_sequence, bias=None):
        """One physical initial state, SI body wrench sequence (H,6), constant bias.

        Returns physical SI states at steps 1..H, WITHOUT re-encoding predictions.
        Bias is in normalized velocity coordinates, not force/current coordinates.
        """
        if self._state(state).shape != (1, 12):
            raise ValueError("predict expects one initial state")
        w = np.asarray(wrench_sequence, dtype=np.float64)
        if w.ndim != 2 or w.shape[1] != 6 or not len(w) or not np.isfinite(w).all():
            raise ValueError("Expected finite (H,6) wrench sequence in N and N*m")
        z = self.lift(state)
        result = []
        for wrench in w:
            z = self.step(z, wrench[None] / US, bias)
            result.append(self.physical(z)[0])
        return np.array(result)

    def innovation(self, previous_observation, mean_completed_wrench, observation):
        """Completed-interval innovation relative to ZERO online correction."""
        baseline = self.predict(previous_observation, np.asarray(mean_completed_wrench)[None])[0]
        current = self._state(observation)
        if current.shape != (1, 12):
            raise ValueError("innovation expects one observation")
        return (current[0, 6:] - baseline[6:]) / VS
