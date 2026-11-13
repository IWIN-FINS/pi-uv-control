"""Multi-step depth-increment-constrained six-velocity bias adaptation.

Derived from the 2026-11-08 FinsSim hardware updater; see provenance/source_manifest.json.
Only bias changes. Fossen parameters and the G1 encoder/operators remain frozen.
"""
from __future__ import annotations

from collections import deque
import ctypes
import hashlib
from pathlib import Path
import subprocess

import numpy as np

from .predictor import Model
from .scales import VS, US
from .geometry import rotation

PERIOD_NS = 100_000_000


def wrapped(value):
    return np.arctan2(np.sin(value), np.cos(value))


class WindowPredictor:
    """Portable NumPy reference; encode once per window, then retain latent memory."""

    def __init__(self, model: Model):
        if abs(model.raw.get("sampleDt", -1) - .1) > 1e-12:
            raise ValueError("Multi-step adaptation requires a 100 ms model")
        self.model = model

    def prepare(self, initial, wrenches):
        self.initial = np.ascontiguousarray(initial, dtype=np.float64)
        self.wrenches = np.ascontiguousarray(wrenches, dtype=np.float64)
        if (self.initial.ndim != 2 or self.initial.shape[1] != 12
                or self.wrenches.ndim != 3 or self.wrenches.shape[0] != len(self.initial)
                or self.wrenches.shape[2] != 6 or not len(self.initial)
                or not self.wrenches.shape[1] or not np.isfinite(self.initial).all()
                or not np.isfinite(self.wrenches).all()):
            raise ValueError("Expected initial=(windows,12), wrenches=(windows,horizon,6)")
        self.initial_lifted = self.model.lift(self.initial)

    def predict(self, biases):
        biases = np.atleast_2d(np.asarray(biases, dtype=np.float64))
        if biases.ndim != 2 or biases.shape[1] != 6 or not np.isfinite(biases).all():
            raise ValueError("Expected finite (candidates,6) biases")
        n, horizon = self.wrenches.shape[:2]
        result = np.empty((len(biases), n, horizon, 12), dtype=np.float64)
        for candidate, bias in enumerate(biases):
            z = self.initial_lifted.copy()
            for step in range(horizon):
                z = self.model.step(z, self.wrenches[:, step] / US, bias)
                physical = self.model.physical(z)
                if not np.isfinite(physical).all() or np.any(abs(physical[:, [3, 5]]) >= 1.):
                    raise ValueError("Window prediction outside original model domain")
                result[candidate, :, step] = physical
        return result


class NativeWindowPredictor(WindowPredictor):
    """Original Linux C++ rollout; compile into an explicitly supplied cache directory."""

    def __init__(self, model, cache_dir):
        super().__init__(model)
        nominal = model.raw["nominal"]
        if (nominal.get("velocityIntegrator") != "rk4_coupled"
                or nominal.get("velocitySubsteps") != 4):
            raise ValueError("Native hardware kernel requires coupled RK4 / four substeps")
        if cache_dir is None:
            raise ValueError("An explicit writable cache_dir is required for the native backend")
        source = Path(__file__).with_name("native") / "multistep_bias_kernel.cpp"
        physics = source.with_name("fossen_mpc_kernel.cpp")
        signature = hashlib.sha256(source.read_bytes() + physics.read_bytes()
                                   + b"g++ -O3 -shared -fPIC -std=c++17").hexdigest()[:16]
        cache = Path(cache_dir).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        library_path = cache / f"multistep_{signature}.so"
        if not library_path.exists():
            subprocess.run(["g++", "-O3", "-shared", "-fPIC", "-std=c++17",
                            str(source), "-o", str(library_path)], check=True)
        self.library = ctypes.CDLL(str(library_path))
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.call = self.library.multistep_bias_rollout
        self.call.argtypes = [ctypes.c_size_t, ctypes.c_size_t] + [pointer] * 12
        self.call.restype = ctypes.c_int
        self.parameters = np.ascontiguousarray(np.concatenate(
            [nominal[k] for k in ("mass", "linearDrag", "quadraticDrag", "spring", "bias")]))
        self.operators = [np.ascontiguousarray(a, dtype=np.float64)
                          for a in (model.A, model.B, model.C, model.D)]

    def predict(self, biases):
        b = np.atleast_2d(np.asarray(biases, dtype=np.float64))
        if b.ndim != 2 or b.shape[1] != 6 or not np.isfinite(b).all():
            raise ValueError("Expected finite (candidates,6) biases")
        n, horizon = len(self.initial), self.wrenches.shape[1]
        result = np.empty((len(b), n, horizon, 12), dtype=np.float64)
        arrays = [np.tile(a, (len(b),) + (1,) * (a.ndim - 1))
                  for a in (self.initial, self.wrenches, self.initial_lifted[:, 36:])]
        repeated = np.ascontiguousarray(np.repeat(b, n, axis=0))
        status = self.call(len(b) * n, horizon, arrays[0], arrays[1], self.parameters,
                           arrays[2], *self.operators, repeated, US, VS, result)
        if status or not np.isfinite(result).all():
            raise ValueError("Window prediction outside original model domain")
        return result


class MultiStepBiasUpdater:
    """Damped local least squares using already completed historical rollouts.

    No target depth is accepted: depth supervision is observed displacement.
    Guard tolerances apply to the same past fit window, not future accuracy.
    """
    def __init__(self, model, depth_weight=.25, *, depth_scale_m=.02,
                 guard_relative=.05, anchors=8, backend='numpy', cache_dir=None):
        if (not np.isfinite([depth_weight, depth_scale_m, guard_relative]).all()
                or depth_weight < 0 or depth_scale_m <= 0 or not 0 <= guard_relative <= .1
                or anchors != int(anchors) or not 1 <= anchors <= 8):
            raise ValueError('Invalid multi-step bias configuration')
        if backend not in ('numpy', 'native'):
            raise ValueError('backend must be numpy or native')
        self.predictor = (WindowPredictor(model) if backend == 'numpy'
                          else NativeWindowPredictor(model, cache_dir))
        self.depth_weight = float(depth_weight)
        self.depth_scale_m = float(depth_scale_m)
        self.guard_relative = float(guard_relative)
        self.anchors = int(anchors)
        self.reset()

    def reset(self):
        self.b = np.zeros(6)
        self.history = deque(maxlen=40)
        self.tick = self.accepted = self.rejected = 0
        self.last_knowledge_end_ns = 0

    def pause(self):
        self.history.clear()
        self.tick = 0

    def _prepare(self):
        history = list(self.history)
        # Eight completed 0.8 s windows, ending at ages 0,.4,...,2.8 sec.
        ends = list(range(len(history)-1, 6, -4))[:self.anchors]
        initial = np.array([history[i-7]['previous'] for i in ends])
        wrenches = np.array([[history[j]['wrench'] for j in range(i-7, i+1)] for i in ends])
        self.truth = np.array([[history[j]['current'] for j in range(i-7, i+1)] for i in ends])
        self.ages = np.array([len(history)-1-i for i in ends])
        weights = .98**self.ages
        self.window_weights = weights/weights.sum()
        self.predictor.prepare(initial, wrenches)
        self.initial = initial

    def residuals(self, predictions):
        h = np.array([0, 4, 7])
        pred = predictions[:, :, h]
        truth = self.truth[:, h]
        velocity = (pred[..., 6:]-truth[None, ..., 6:])/VS
        depth_pred = -(pred[..., 1]-self.initial[None, :, None, 1])
        depth_true = -(truth[..., 1]-self.initial[:, None, 1])
        depth = (depth_pred-depth_true[None])/self.depth_scale_m
        angle = wrapped(pred[..., 3:6]-truth[None, ..., 3:6])/.1
        root_weights = np.sqrt(self.window_weights)[None, :, None, None]/np.sqrt(3)
        # Six velocity dimensions stay summed, consistent with original loss.
        rv = (velocity*root_weights).reshape(len(pred), -1)
        rd = (depth*root_weights[..., 0]).reshape(len(pred), -1)
        ra = (angle*root_weights).reshape(len(pred), -1)
        residual = np.c_[rv, np.sqrt(self.depth_weight)*rd, np.sqrt(.2)*ra]
        # Body-up chordal direction error guard; not a geodesic angle.
        predicted_up = rotation(pred[..., :6])[..., :, 1]
        true_up = rotation(truth[..., :6])[..., :, 1]
        gravity_difference = (predicted_up-true_up[None])/.1
        tilt_loss = np.sum((gravity_difference*root_weights)**2, axis=(1, 2, 3))
        parts = dict(velocity=np.sum(rv**2, axis=1), depth=np.sum(rd**2, axis=1),
                     angle=np.sum(ra**2, axis=1), tilt=tilt_loss)
        return residual, parts

    def _objective(self, residual, candidates, prior):
        return np.sum(residual**2, axis=1)+.05*np.sum(candidates**2, axis=1)+.5*np.sum((candidates-prior)**2, axis=1)

    def update(self, start_ns, end_ns, previous, mean_wrench_si, current, *, available_ns):
        start_ns, end_ns, available_ns = int(start_ns), int(end_ns), int(available_ns)
        previous, current, wrench = [np.array(x, dtype=float, copy=True) for x in (previous, current, mean_wrench_si)]
        if (end_ns-start_ns != PERIOD_NS or available_ns < end_ns or previous.shape != (12,)
                or current.shape != (12,) or wrench.shape != (6,)
                or not all(np.isfinite(a).all() for a in (previous, current, wrench))):
            raise ValueError('Require valid completed 100 ms state/input interval')
        if self.history and start_ns < self.history[-1]['end_ns']:
            raise ValueError('Nonmonotonic completed interval')
        if self.history and (start_ns != self.history[-1]['end_ns']
                             or not np.allclose(previous, self.history[-1]['current'], atol=1e-10, rtol=0)):
            self.pause()
        self.history.append(dict(previous=previous, current=current, wrench=wrench, end_ns=end_ns))
        self.tick += 1
        self.last_knowledge_end_ns = end_ns
        info = dict(accepted=False, tick=self.tick, samples=len(self.history), end_ns=end_ns,
                    available_ns=available_ns, before=None, after=None, reason='warmup')
        if self.tick < 20 or len(self.history) < 8:
            return info
        self._prepare()
        old = self.b.copy()
        eps = 1e-4
        probes = np.r_[old[None], old[None]+np.eye(6)*eps, old[None]-np.eye(6)*eps]
        try:
            residuals, parts = self.residuals(self.predictor.predict(probes))
        except ValueError:
            self.rejected += 1
            return dict(info, reason='prediction_domain')
        residual = residuals[0]
        jac = ((residuals[1:7]-residuals[7:13])/(2*eps)).T
        hessian = jac.T@jac+.55*np.eye(6)
        gradient = jac.T@residual+.05*old
        target = old-np.linalg.solve(hessian, gradient)
        candidate = old+.2*(target-old)
        candidate *= min(1., .25/max(np.linalg.norm(candidate), 1e-30))
        delta = candidate-old
        candidate = old+delta*min(1., .01/max(np.linalg.norm(delta), 1e-30))
        fractions = .5**np.arange(6)
        candidates = old+(candidate-old)*fractions[:, None]
        try:
            proposed_residual, after = self.residuals(self.predictor.predict(candidates))
        except ValueError:
            self.rejected += 1
            return dict(info, reason='proposal_domain')
        baseline = float(self._objective(residuals[:1], old[None], old)[0])
        objectives = self._objective(proposed_residual, candidates, old)
        good = objectives <= baseline+1e-14
        # For depth_weight=0 this is a multi-step velocity/attitude control arm.
        # Depth-constrained arms additionally require no worse historical depth.
        if self.depth_weight > 0:
            good &= after['depth'] <= parts['depth'][0]+1e-12
        for key in ('velocity', 'tilt', 'angle'):
            good &= after[key] <= (1+self.guard_relative)*parts[key][0]+1e-10
        choices = np.flatnonzero(good)
        info.update(before=baseline, depth_before=float(parts['depth'][0]),
                    velocity_before=float(parts['velocity'][0]), tilt_before=float(parts['tilt'][0]))
        if len(choices):
            j = int(choices[0])
            self.b = candidates[j].copy()
            self.accepted += 1
            info.update(accepted=True, after=float(objectives[j]),
                        depth_after=float(after['depth'][j]), velocity_after=float(after['velocity'][j]),
                        tilt_after=float(after['tilt'][j]), step_fraction=float(fractions[j]), reason='accepted')
        else:
            self.rejected += 1
            info.update(after=baseline, reason='loss_or_guard_rejected')
        return info
