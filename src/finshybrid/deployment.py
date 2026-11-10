"""Timestamped shadow-cycle adapter. No actuator transport or automatic arming."""
import time
import numpy as np
from .controller import OnlinePredictor
from .scales import US


class ShadowCycle:
    """Host supplies synchronized SI observations, completed inputs and a solver.

    A solver receives a fresh historical-objective record and returns
    {success: bool, sequence: (8,8)} in normalized thruster coordinates.
    Returning a candidate never dispatches it. Timeouts require host policy.
    """
    def __init__(self, model=None, timestamp_tolerance=.002):
        if not np.isfinite(timestamp_tolerance) or not 0 <= timestamp_tolerance < .1:
            raise ValueError('timestamp_tolerance must be in [0,.1) seconds')
        self.online = OnlinePredictor(model)
        self.tolerance = timestamp_tolerance
        self.timestamp = None

    def reset(self):
        self.online.reset()
        self.timestamp = None

    def observe(self, timestamp_s, state, completed_mean_wrench=None):
        t = float(timestamp_s)
        if not np.isfinite(t):
            raise ValueError('Nonfinite timestamp')
        if self.timestamp is not None and abs(t-self.timestamp-.1) > self.tolerance:
            raise ValueError('Expected consecutive 100ms samples; host must handle gaps/reset')
        if self.timestamp is None and completed_mean_wrench is not None:
            raise ValueError('First sample has no completed interval')
        report = self.online.observe(state, completed_mean_wrench)
        self.timestamp = t
        return report

    def prepare_objective(self, config, reference_pose, reference_rates, start=None):
        """Reference rates are WORLD pose derivatives, not body COM velocities.

        config defines measured allocation, force limits, objective weights.
        Pose/rates use the same Unity axes/Euler convention as the model.
        """
        if self.timestamp is None:
            raise RuntimeError('Observe first')
        def array(value, shape, name):
            a = np.asarray(value, dtype=float)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f'{name} must be finite {shape}')
            return a.copy()
        lower = array(config['force_min_n'], (8,), 'force_min_n')
        upper = array(config['force_max_n'], (8,), 'force_max_n')
        if np.any(upper <= lower):
            raise ValueError('Each force maximum must exceed its minimum')
        basis = array(config['allocation_basis'], (6,8), 'allocation_basis')
        warm = np.zeros((8,8)) if start is None else array(start, (8,8), 'start')
        if np.any(abs(warm)>1):
            raise ValueError('start exceeds normalized thruster bounds')
        weights = {}
        for external, internal in [('position_weight','positionWeight'),
                                   ('velocity_weight','velocityWeight'),
                                   ('effort_weight','effortWeight')]:
            weights[internal] = array(config[external], (6,), external)
            if np.any(weights[internal]<0):
                raise ValueError('Objective weights must be nonnegative')
        record = dict(z=self.online.model.lift(self.online.previous)[0],
                      de83VelocityBias=self.online.online.b.copy(),
                      referencePose=array(reference_pose,(8,6),'reference_pose'),
                      referenceVelocity=array(reference_rates,(8,6),'reference_rates'),
                      forceCenter=(upper+lower)/2, forceHalf=(upper-lower)/2,
                      allocationBasis=basis, wrenchLimit=US.copy(), start=warm.ravel(),
                      **weights)
        return record

    def solve_shadow(self, record, solver, deadline_s=.1):
        """Solve using one immutable bias snapshot; report failures to the host.

        deadline_s is a diagnostic threshold, not a preemptive timeout.
        No returned command is automatically safe/authorized for a real robot.
        """
        if not np.isfinite(deadline_s) or deadline_s<=0:
            raise ValueError('deadline_s must be positive')
        started = time.perf_counter()
        # Caller/successive cycles cannot mutate this invocation's snapshot.
        frozen = {k:np.array(v,copy=True) for k,v in record.items()}
        for value in frozen.values():value.setflags(write=False)
        answer = solver(self.online.model, frozen)
        elapsed = time.perf_counter()-started
        sequence = np.asarray(answer['sequence'],dtype=float)
        valid = sequence.shape==(8,8) and np.isfinite(sequence).all() and np.all(abs(sequence)<=1)
        accepted = bool(answer['success']) and bool(valid) and elapsed<=deadline_s
        force = None
        if accepted:
            force = (frozen['forceCenter']+sequence[0]*frozen['forceHalf']).tolist()
        return dict(status='candidate_only' if accepted else 'solver_rejected',
                    dispatch_permitted=False, candidate_force_n=force,
                    solve_seconds=elapsed, deadline_exceeded=elapsed>deadline_s,
                    bias_snapshot=frozen['de83VelocityBias'].tolist())
