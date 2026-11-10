"""Original G1/A rollout and causal online velocity bias for the common MPC.

Reuses finshybrid (Underwater-Adaptive-Dynamics, see its LICENSE/NOTICE) without
changing frozen models. Native horizons preserve predictor.Model.step's
kinematics, current-latent readout and recursive latent memory.
"""
import ctypes
from collections import deque
from pathlib import Path
import time

import numpy as np

from finshybrid.predictor import Model
from finshybrid.bias import BiasUpdater
from finshybrid.scales import VS, US
from fossen_mpc import FossenPlant, DT, H, wrapped
from export_dynamics_training import interval_mean


class G1Plant:
    def __init__(self, model_path, nominal, matrix):
        self.model_path = Path(model_path).resolve()
        self.model = Model(self.model_path)
        if self.model.raw['nominal'] != nominal:
            raise ValueError('G1 checkpoint nominal differs from live MPC; retrain rather than edit the checkpoint')
        if abs(self.model.raw.get('sampleDt', -1)-DT) > 1e-12:
            raise ValueError('G1 checkpoint period must be 100 ms')
        self.nominal = nominal
        self.matrix = np.asarray(matrix)
        self.physics = FossenPlant(nominal, matrix)
        self.bias = np.zeros(6)
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        self.native = self.physics.library.g1_rollout
        self.native.argtypes = [ctypes.c_size_t, ctypes.c_size_t] + [pointer]*12
        self.native.restype = ctypes.c_int
        self.operators = [np.ascontiguousarray(a, dtype=np.float64)
                          for a in (self.model.A,self.model.B,self.model.C,self.model.D)]

    def rollout_batch(self, state, forces, bias=None):
        forces = np.asarray(forces, dtype=float)
        if (forces.ndim != 3 or forces.shape[2] != 8 or min(forces.shape[:2]) == 0
                or not np.isfinite(forces).all()):
            raise ValueError('Expected finite (batch,horizon,8) channel forces')
        b = self.bias.copy() if bias is None else np.asarray(bias, dtype=float).copy()
        if b.shape != (6,) or not np.isfinite(b).all() or np.linalg.norm(b) > .250000001:
            raise ValueError('Invalid online velocity bias')
        s = np.ascontiguousarray(self.model._state(state))
        if s.shape != (1,12):raise ValueError('Expected one initial state')
        # Every probe starts at the same observed state. Encode it only once.
        latent = np.ascontiguousarray(self.model.lift(s)[:,36:])
        wrenches = np.ascontiguousarray(forces @ self.matrix.T)
        result = np.empty((*forces.shape[:2],12), dtype=np.float64)
        if self.native(len(forces),forces.shape[1],s,wrenches,self.physics.parameters,
                       latent,*self.operators,np.ascontiguousarray(b),US,VS,result):
            raise ValueError('G1 prediction outside feasible domain')
        return result

    def rollout(self, state, forces):
        return self.rollout_batch(state, np.asarray(forces)[None])[0]

    def trajectory_sensitivity(self, state, forces, move):
        # Differentiate the WHOLE recursive trajectory, including latent memory.
        # A physical-state local Jacobian would silently discard that memory.
        eps = 1e-4
        delta = np.asarray(move).T.reshape(-1,H,8)*eps
        probes = np.concatenate([forces[None]+delta, forces[None]-delta])
        values = self.rollout_batch(state, probes)
        n = len(delta)
        diff = values[:n]-values[n:]
        diff[:,:,3:6] = wrapped(diff[:,:,3:6])
        return (diff/(2*eps)).transpose(1,2,0).reshape(H*12,n)


class CausalG1Bias:
    """Fixed 100 ms observations and actual completed MCU-output integrals.

    Only past/current samples enter interpolation. Vision/synchronization/input
    gaps clear residual history and grid; retain the bounded bias until disarm.
    The default reuses the original BiasUpdater. Multi-step depth fitting is an
    explicit experiment option; frozen G1 and nominal dynamics are unchanged.
    """
    period_ns = 100_000_000

    def __init__(self, plant, estimator, *, method='single_step', depth_weight=.25):
        if method not in ('single_step', 'multistep_depth'):
            raise ValueError('Unknown causal bias update method')
        if not np.isfinite(depth_weight) or depth_weight < 0:
            raise ValueError('Invalid multi-step depth weight')
        self.method, self.depth_weight = method, float(depth_weight)
        self.plant, self.estimator = plant, estimator
        self.outputs = deque(maxlen=180)
        self.reset()

    def reset(self):
        if self.method == 'multistep_depth':
            from multistep_bias import MultiStepBiasUpdater
            self.online = MultiStepBiasUpdater(self.plant, self.depth_weight)
        else:
            self.online = BiasUpdater()
        self.states = deque(maxlen=180)
        self.grid = None
        self.last = None
        self.last_reason = 'episode_reset'
        self.accepted = 0
        self.window_resets = 0
        self.waiting_for_output = False
        self.completed_intervals = 0
        self.last_update = None
        self.plant.bias[:] = 0

    def pause(self, reason):
        if self.states or self.online.history or self.last is not None:
            self.window_resets += 1
        self.states.clear(); self.grid = None; self.last = None
        self.online.history.clear(); self.online.tick = 0
        self.waiting_for_output = False
        self.last_reason = reason

    def put_output(self, packet):
        # Copy callback data; keep every telemetry snapshot, not control-rate decimation.
        stamp = int(packet.get('mapped_snapshot_time_ns',0))
        if stamp <= 0:return
        if self.outputs and stamp <= self.outputs[-1][0]:
            self.outputs.clear(); self.pause('nonmonotonic_output_time')
        estimate = self.estimator.estimate(packet['motor_throttle'], packet['motor_enabled_mask'])
        self.outputs.append((stamp, dict(packet), estimate))
        self._drain()

    def state_at(self, stamp):
        times = np.array([p[0] for p in self.states], dtype=np.int64)
        i = int(np.searchsorted(times,stamp,side='right'))-1
        if i < 0:raise ValueError('state_outside_grid')
        if times[i] == stamp:return self.states[i][1].copy()
        if i+1 >= len(times):raise ValueError('state_not_completed')
        a,b = self.states[i], self.states[i+1]
        gap_limit = self.period_ns if self.method == 'multistep_depth' else 180_000_000
        if b[0]-a[0] > gap_limit:raise ValueError('state_grid_gap')
        d = b[1]-a[1]; d[3:6] = wrapped(d[3:6])
        out = a[1]+d*((stamp-a[0])/(b[0]-a[0]))
        out[3:6] = wrapped(out[3:6])
        return out

    def observe(self, stamp, state):
        stamp = int(stamp)
        state = np.asarray(state,dtype=float)
        if state.shape != (12,) or not np.isfinite(state).all():raise ValueError('Invalid online state')
        if self.states and stamp <= self.states[-1][0]:return self.diagnostics()
        if self.states and stamp-self.states[-1][0] > 180_000_000:self.pause('state_grid_gap')
        self.states.append((stamp,state.copy()))
        if self.grid is None:
            self.grid = ((stamp+self.period_ns-1)//self.period_ns)*self.period_ns
        return self._drain()

    def _drain(self):
        """A late output may complete an interval without another state callback."""
        if self.grid is None or not self.states:return self.diagnostics()
        self.waiting_for_output=not self.outputs or self.outputs[-1][0]<self.grid<=self.states[-1][0]
        if not self.outputs:return self.diagnostics()
        completed = min(self.states[-1][0],self.outputs[-1][0])
        packets = [p[1] for p in self.outputs]
        estimates = [p[2] for p in self.outputs]
        times = np.array([p[0] for p in self.outputs],dtype=np.int64)
        while self.grid <= completed:
            end = self.grid
            try:
                current = self.state_at(end)
                if self.last is not None:
                    start, previous = self.last
                    mean = interval_mean(start,end,packets,estimates,times)
                    # Zero-bias baseline; correction is NOT multiplied by DT.
                    # Model.innovation accepts the integrated COM wrench directly.
                    innovation = self.plant.model.innovation(previous,mean,current)
                    update = (self.online.update(start,end,previous,mean,current,available_ns=time.time_ns())
                              if self.method == 'multistep_depth' else self.online.update(innovation))
                    self.completed_intervals += 1
                    self.last_update = dict(start_ns=start,end_ns=end,innovation=innovation.tolist(),
                        mean_wrench_si=mean.tolist(),accepted=bool(update['accepted']),
                        fit_before=None if update['before'] is None else float(update['before']),
                        fit_after=None if update['after'] is None else float(update['after']))
                    if self.method == 'multistep_depth':
                        self.last_update.update(update_method=self.method,optimizer=update,
                            previous_state_unity=previous.tolist(),current_state_unity=current.tolist())
                    self.accepted += int(update['accepted'])
                    self.plant.bias[:] = self.online.b
                    self.last_reason = 'completed_interval'
                self.last = (end,current)
            except ValueError as exc:
                self.window_resets += 1
                self.online.history.clear(); self.online.tick = 0; self.last = None
                self.last_reason = str(exc)
            self.grid += self.period_ns
        self.waiting_for_output=self.outputs[-1][0]<self.grid<=self.states[-1][0]
        return self.diagnostics()

    def diagnostics(self):
        return dict(update_method=self.method,depth_weight=self.depth_weight if self.method=='multistep_depth' else None,
            bias=self.plant.bias.tolist(), samples=len(self.online.history),
            accepted_updates=self.accepted, last_completed_ns=None if self.last is None else self.last[0],
            status=self.last_reason,waiting_for_output=self.waiting_for_output,
            window_resets=self.window_resets,window_tick=self.online.tick,
            completed_intervals=self.completed_intervals,bias_norm=float(np.linalg.norm(self.plant.bias)),
            physical_velocity_increment_per_model_step=(self.plant.bias*VS).tolist(),last_update=self.last_update)
