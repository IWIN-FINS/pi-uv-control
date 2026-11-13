"""Fossen MPC and bounded pressure/IMU fallback; no ROS or hardware access.

One real-time iteration: nonlinear nominal rollout, finite-difference discrete
linearization, condensed box QP solved by projected accelerated gradient, then
nonlinear rollout/cost check. Four two-step input blocks, eight 100 ms steps.
This is a new solver, not the archived native SQP implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import ctypes
from functools import lru_cache
import hashlib
import itertools
import json
import os
from pathlib import Path
import struct
import subprocess
import time

import numpy as np

from dynamics_thrust import C, ThrustEstimator
from mpc_runtime import control_time_limits
from export_dynamics_training import quaternion_matrix, unity_euler
from finshybrid import physics
from finshybrid.geometry import rotation, start_pose_average, euler, rate_matrix
from finshybrid.scales import PC, PS, VS, US, CENTER

DT = .1
H = 8
LINEAR_SPEED_LIMIT_MPS = .5


def check_linear_speed(state):
    """Shared per-body-axis overspeed guard; independent of reference speed."""
    velocity=np.asarray(state,dtype=float)[6:9]
    if velocity.shape!=(3,) or not np.isfinite(velocity).all():
        raise ValueError('Invalid trajectory linear velocity')
    peak=float(np.max(abs(velocity)))
    if peak>LINEAR_SPEED_LIMIT_MPS:
        raise ValueError(f'Trajectory speed {peak:.4f} m/s exceeds limit {LINEAR_SPEED_LIMIT_MPS:.2f} m/s; control disabled')


def control_period_seconds(value):
    """Control may replan faster than the frozen 100 ms prediction model."""
    value=float(value)
    if not np.isfinite(value) or not 0<value<=DT:
        raise ValueError('Control period must be positive and no larger than prediction DT')
    return value


@lru_cache(maxsize=1)
def native_kernel():
    """Load one source-hashed double-precision dynamics/QP library per process."""
    source=Path(__file__).with_name('fossen_mpc_kernel.cpp')
    cache_root = os.environ.get('FINSHYBRID_CACHE_DIR')
    if not cache_root:
        raise ValueError('Set FINSHYBRID_CACHE_DIR to an external writable cache directory')
    cache = Path(cache_root)/'fossen'
    cache.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(source.read_bytes()+b'g++ -O3 -shared -fPIC -std=c++17').hexdigest()[:16]
    path=cache/f'fossen_{signature}.so'
    if not path.exists():
        subprocess.run(['g++','-O3','-shared','-fPIC','-std=c++17',str(source),'-o',str(path)],
                       check=True,env=dict(os.environ,TMPDIR=str(cache)))
    library=ctypes.CDLL(str(path))
    pointer=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    library.box_qp_scaled.argtypes=[ctypes.c_size_t,ctypes.c_int,ctypes.c_double,ctypes.c_double]+[pointer]*7
    library.box_qp_scaled.restype=ctypes.c_int
    return library


def control_nominal(document, *, configured_restoring=False):
    """Build the live predictor without treating archived stiffness as measured.

    The offline experiment document explicitly retains an unverified roll/pitch
    spring. Overestimating it makes an optimal controller oppose the assumed
    passive righting torque. Omit those two terms for commissioning; keep the
    source document and all identified mass/drag/buoyancy parameters unchanged.
    """
    nominal=copy.deepcopy(document['nominal'])
    if not configured_restoring:
        nominal['spring'][3]=nominal['spring'][5]=0.
    return nominal


def wrapped(value):
    return np.arctan2(np.sin(value), np.cos(value))


def convert_state(position_ros, quaternion_ros, velocity_origin_ros, omega_ros, com_ros, surface,
                  *, tilt_limit_rad=.5):
    if not np.isfinite(tilt_limit_rad) or not 0<tilt_limit_rad<1.:
        raise ValueError('Tilt limit must be positive and below the prediction domain of 1 rad')
    q = np.asarray(quaternion_ros, dtype=float)
    if not .9 < np.linalg.norm(q) < 1.1:
        raise ValueError('invalid_orientation')
    q = q/np.linalg.norm(q)
    r = C @ quaternion_matrix(q) @ C.T
    pose = np.r_[C@(np.asarray(position_ros)-[0,0,surface]), unity_euler(r)]
    velocity = np.asarray(velocity_origin_ros)+np.cross(omega_ros,com_ros)
    state = np.r_[pose,C@velocity,-C@np.asarray(omega_ros)]
    if not np.isfinite(state).all():raise ValueError('invalid_state')
    if np.max(np.abs(state[[3,5]])) >= tilt_limit_rad:
        raise ValueError(f'tilt_limit_exceeded: roll={np.degrees(state[3]):.1f}, '
                         f'pitch={np.degrees(state[5]):.1f}, limit={np.degrees(tilt_limit_rad):.1f} deg')
    return state


class ForceMap:
    """Per-channel branch-aware inverse of the collection's static force curves."""
    def __init__(self, document, throttle_limit=.25):
        if not 0 < throttle_limit <= .30:
            raise ValueError('throttle_limit must be in (0,.30]')
        self.estimator = ThrustEstimator(document)
        self.matrix = np.r_[C@self.estimator.matrix[:3],-C@self.estimator.matrix[3:]]
        self.limit = float(throttle_limit)
        self.branches = []
        lower,upper = [],[]
        for channel in document['channels']:
            branch = {}
            for sign in (-1,1):
                side = 'positive' if sign*channel['positive_throttle_rotation_sign']>0 else 'negative'
                curve = document['curves'][channel['propeller_type']][side]
                throttle = np.asarray(curve['command_abs'])*document['historical_final_throttle_scale']
                factor=curve['c_omega']*(2*np.pi/60)**2
                rpm=np.asarray(curve['rpm_abs'])
                force = factor*rpm**2
                # Retain flat RPM bins: they encode the measured deadband.
                # np.interp uses the right edge for an interior flat; zero
                # force is handled explicitly so it always maps to zero.
                if len(force)<2 or np.any(np.diff(force)<0) or force[-1]<=0 or throttle[-1]<self.limit:
                    raise ValueError('Insufficient invertible calibration support')
                branch[sign] = (rpm,throttle,factor)
            self.branches.append(branch)
            lower.append(-branch[-1][2]*np.interp(self.limit,branch[-1][1],branch[-1][0])**2)
            upper.append(branch[1][2]*np.interp(self.limit,branch[1][1],branch[1][0])**2)
        self.lower,self.upper=np.asarray(lower),np.asarray(upper)

    def throttle(self, forces):
        f=np.asarray(forces,dtype=float)
        if f.shape!=(8,) or not np.isfinite(f).all() or np.any(f<self.lower-1e-8) or np.any(f>self.upper+1e-8):
            raise ValueError('Force outside calibrated control limits')
        out=np.zeros(8)
        for i,value in enumerate(f):
            if value:
                rpm,throttle,factor=self.branches[i][1 if value>0 else -1]
                out[i]=np.sign(value)*np.interp(np.sqrt(abs(value)/factor),rpm,throttle)
        return np.clip(out,-self.limit,self.limit)


class FossenPlant:
    def __init__(self, nominal_config, matrix, *, native=True):
        self.nominal=nominal_config
        self.matrix=np.asarray(matrix)
        self.m,self.ld,self.qd,self.spring,self.bias=[np.asarray(nominal_config[k],float)
            for k in ('mass','linearDrag','quadraticDrag','spring','bias')]
        if self.nominal.get('velocityIntegrator')!='rk4_coupled' or self.nominal.get('velocitySubsteps')!=4:
            raise ValueError('Expected the tested coupled RK4/four-substep configuration')
        self.native=None
        if native:
            self.library=native_kernel()
            pointer=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
            self.native=self.library.fossen_step
            self.native.argtypes=[ctypes.c_size_t,pointer,pointer,pointer,pointer]
            self.native.restype=ctypes.c_int
            self.parameters=np.ascontiguousarray(np.r_[self.m,self.ld,self.qd,self.spring,self.bias])

    def step(self, states, forces):
        if self.native is None:return self.numpy_step(states,forces)
        s=np.ascontiguousarray(np.atleast_2d(states),dtype=np.float64)
        f=np.atleast_2d(forces)
        if s.shape[1]!=12 or f.shape!=(len(s),8) or not np.isfinite(s).all() or not np.isfinite(f).all():
            raise ValueError('Expected finite batched states and forces')
        wrench=np.ascontiguousarray(f@self.matrix.T)
        result=np.empty_like(s)
        if self.native(len(s),s,wrench,self.parameters,result) or np.max(abs(result[:,[3,5]]))>=1.:
            raise ValueError('Fossen prediction outside feasible domain')
        return result

    def numpy_step(self, states, forces):
        s=np.atleast_2d(states).astype(float)
        f=np.atleast_2d(forces).astype(float)
        wrench=f@self.matrix.T
        def rhs(x):
            p,v=x[:,:6],x[:,6:]
            r=rotation(p)
            mom,ang=v[:,:3]*self.m[:3],v[:,3:]*self.m[3:]
            cor=np.c_[np.cross(v[:,3:],mom),np.cross(v[:,:3],mom)+np.cross(v[:,3:],ang)]
            restoring=np.c_[self.bias[1]*r[:,1,:],self.spring[3]*r[:,1,2],
                            np.zeros(len(x)),-self.spring[5]*r[:,1,0]]
            acc=(wrench-self.ld*v-self.qd*v*abs(v)+restoring-cor)/self.m
            return np.c_[np.einsum('nij,nj->ni',rate_matrix(p),v),acc]
        x=s.copy()
        for _ in range(4):
            k1=rhs(x);k2=rhs(x+.0125*k1);k3=rhs(x+.0125*k2);k4=rhs(x+.025*k3)
            x+=.025*(k1+2*k2+2*k3+k4)/6
        velocity=x[:,6:]
        p=s[:,:6]
        pf=start_pose_average(p,s[:,6:],velocity)
        rr=rotation(pf)
        pos=pf[:,:3]-np.einsum('nij,j->ni',rr-rotation(p),CENTER)
        result=np.c_[pos,euler(rr),velocity]
        if not np.isfinite(result).all() or np.max(abs(result[:,[3,5]]))>=1.:
            raise ValueError('Fossen prediction outside feasible attitude domain')
        return result

    def rollout(self, state, forces):
        s=np.asarray(state); values=[]
        for f in forces:
            s=self.step(s,f)[0];values.append(s)
        return np.asarray(values)

    def linearize(self, states, forces):
        # Batch all 8 horizon stages and all centered perturbations.
        combined=np.c_[states,forces]
        eps=np.r_[np.full(12,1e-5),np.full(8,1e-4)]
        delta=np.diag(eps)
        probes=np.concatenate([combined[:,None,:]+delta,combined[:,None,:]-delta],axis=1)
        result=self.step(probes[:,:,:12].reshape(-1,12),probes[:,:,12:].reshape(-1,8)).reshape(H,40,12)
        diff=result[:,:20]-result[:,20:]
        diff[:,:,3:6]=wrapped(diff[:,:,3:6])
        jac=(diff/(2*eps)[None,:,None]).transpose(0,2,1)
        return jac[:,:,:12],jac[:,:,12:]


@dataclass
class PilotInput:
    forward: float=0.
    left: float=0.
    yaw_ccw: float=0.
    down: float=0.
    up: float=0.
    start: bool=False

    def neutral(self):
        return not any((self.forward,self.left,self.yaw_ccw,self.down,self.up))


class PilotReference:
    def __init__(self,forward_speed=.12,lateral_speed=.10,yaw_rate=.4,depth_rate=.10,depth_min=.15,depth_max=1.5,
                 descent_rate=.04,*,control_period=DT):
        self.forward_speed,self.lateral_speed,self.yaw_rate=forward_speed,lateral_speed,yaw_rate
        self.depth_rate,self.depth_min,self.depth_max=depth_rate,depth_min,depth_max
        self.descent_rate=descent_rate
        self.control_period=control_period_seconds(control_period)
        self.yaw=self.depth=None

    def reset(self,state):
        self.depth=float(-state[1]); self.yaw=float(state[4])
        if not self.depth_min<=self.depth<=self.depth_max:
            raise ValueError('Current depth outside configured operating range')

    def build(self,state,command):
        if self.depth is None:self.reset(state)
        input_rate=command.down-command.up
        rate=input_rate*(self.descent_rate if input_rate>0 else self.depth_rate)
        self.depth=float(np.clip(self.depth+rate*self.control_period,self.depth_min,self.depth_max))
        # ROS positive CCW -> Unity negative rotation about y (axial transform).
        yaw_rate=-command.yaw_ccw*self.yaw_rate
        self.yaw=float(wrapped(self.yaw+yaw_rate*self.control_period))
        refs=np.repeat(np.asarray(state)[None],H,axis=0)
        refs[:,3:6]=0.; refs[:,6:]=0.
        for j in range(H):
            offset=j*DT+(DT-self.control_period)
            refs[j,1]=-np.clip(self.depth+rate*offset,self.depth_min,self.depth_max)
            refs[j,4]=wrapped(self.yaw+yaw_rate*offset)
        refs[:,6]=command.forward*self.forward_speed
        refs[:,8]=command.left*self.lateral_speed
        refs[:,7]=-rate
        refs[:,10]=yaw_rate
        return refs


class LineTrajectory:
    """Fixed-world, level, constant-depth line out/back with smooth reversals.

    Plan at a stationary observed pose before arming. The reference clock only
    advances when build() runs in the normal MPC branch. Pausing never moves
    the endpoints or silently re-anchors the experiment to the observed robot.
    """
    def __init__(self,distance=.30,speed=.03,cycles=2,depth_min=.20,depth_max=.60,*,control_period=DT):
        if (not np.isfinite([distance,speed,cycles,depth_min,depth_max]).all() or
                not .05<=distance<=1.00 or not .005<=speed<=.10 or
                int(cycles)!=cycles or not 1<=cycles<=5 or not 0<depth_min<depth_max<=.70):
            raise ValueError('Invalid commissioning trajectory limits')
        self.distance,self.speed,self.cycles=distance,speed,int(cycles)
        self.depth_min,self.depth_max=depth_min,depth_max
        self.control_period=control_period_seconds(control_period)
        self.half_period=np.pi*distance/(2*speed)
        self.duration=4.+2*self.cycles*self.half_period
        self.anchor=None;self.elapsed=0.;self.depth=None;self.yaw=None
        self.segment=0

    def plan(self,state,*,from_preparation=False):
        state=np.asarray(state,dtype=float)
        if state.shape!=(12,) or not np.isfinite(state).all():raise ValueError('Invalid trajectory anchor')
        if not self.depth_min<=-state[1]<=self.depth_max:
            raise ValueError(f'Stabilize depth at {self.depth_min:.2f}–{self.depth_max:.2f} mbefore replanning')
        tilt_limit=20 if from_preparation else 15
        if max(abs(state[[3,5]]))>np.deg2rad(tilt_limit):
            raise ValueError(f'Planned tilt exceeds {tilt_limit}°')
        self.anchor=state[:6].copy();self.anchor[[3,5]]=0.
        self.yaw=float(state[4]);self.depth=float(-state[1]);self.elapsed=0.;self.segment=0
        self.direction=rotation(self.anchor[None])[0][:,0]
        self.endpoint=self.anchor[:3]+self.distance*self.direction
        return self.description()

    def description(self):
        if self.anchor is None:return None
        return dict(kind='constant_depth_smooth_line_out_and_back',anchor_unity=self.anchor.tolist(),
                    endpoint_unity=self.endpoint.tolist(),distance_m=self.distance,
                    maximum_reference_speed_mps=self.speed,cycles=self.cycles,duration_sec=self.duration,
                    endpoint_dwell_sec=0.,start_dwell_sec=2.,finish_dwell_sec=2.,
                    position_tolerance_for_arm_m=.08,depth_tolerance_for_arm_m=.04,
                    tracking_depth_error_stop_m=None,control_period_sec=self.control_period,
                    prediction_dt_sec=DT)

    def reset(self,state,*,require_stationary=True):
        if self.anchor is None:raise ValueError('While disarmed, press R to plan and inspect the trajectory')
        self.check(state)
        if require_stationary:
            if (np.linalg.norm(np.asarray(state)[[0,2]]-self.anchor[[0,2]])>.08 or
                    abs(state[1]-self.anchor[1])>.04 or abs(wrapped(state[4]-self.yaw))>.15):
                raise ValueError('Vehicle has left the preview start; disarm and press R to replan')
            if max(abs(np.asarray(state)[6:9]))>.05 or max(abs(np.asarray(state)[9:]))>.15:
                raise ValueError('Hold still before starting the trajectory')
        self.elapsed=0.;self.segment=0

    def check(self,state):
        if self.anchor is None:raise ValueError('Trajectory has not been planned')
        state=np.asarray(state)
        if max(abs(state[[3,5]]))>np.deg2rad(20):raise ValueError('Trajectory tilt exceeds 20°; control disabled')
        if abs(wrapped(state[4]-self.yaw))>.45:raise ValueError('Excessive trajectory heading error; control disabled')
        offset=state[[0,2]]-self.anchor[[0,2]];direction=self.direction[[0,2]]
        along=float(offset@direction);cross=offset-along*direction
        if np.linalg.norm(cross)>.15 or not -.15<=along<=self.distance+.15:
            raise ValueError('Vehicle left the trajectory corridor; control disabled')
        check_linear_speed(state)

    def point(self,t):
        active=np.clip(t-2.,0.,2*self.cycles*self.half_period)
        phase=np.pi*active/self.half_period
        distance=.5*self.distance*(1.-np.cos(phase))
        speed=.5*self.distance*np.pi/self.half_period*np.sin(phase)
        if t<=2. or t>=self.duration-2.:speed=0.
        return self.anchor[:3]+distance*self.direction,speed*self.direction

    def build(self,state,command):
        if not command.neutral():raise ValueError('Joystick motion detected; trajectory disabled. Start always disarms')
        self.check(state)
        if self.elapsed>=self.duration:raise RuntimeError('Trajectory complete; control disabled')
        current,_=self.point(self.elapsed)
        if np.linalg.norm((np.asarray(state)[:3]-current)[[0,2]])>.20:
            raise ValueError('Horizontal tracking error exceeds 0.20 m; control disabled')
        refs=np.zeros((H,12));rr=rotation(self.anchor[None])[0]
        for j in range(H):
            position,velocity=self.point(self.elapsed+(j+1)*DT)
            refs[j,:3]=position;refs[j,3:6]=self.anchor[3:6]
            refs[j,6:9]=rr.T@velocity
        self.elapsed+=self.control_period
        return refs


class DivePreparation:
    """Bounded pressure/IMU depth acquisition before the horizontal anchor.

    A fresh surface preview is still reviewed before Start. Horizontal output
    stays zero until target depth and a fresh visual EKF persist for a window.
    The surface preview is advisory. Acquire the fixed final anchor at depth;
    horizontal drift during preparation is not a stability or corridor gate.
    """
    def __init__(self,line,hold,target_depth=.40,stable_sec=2.,timeout=60.,*,require_stationary=True,depth_tolerance=.02):
        if (not np.isfinite([target_depth,stable_sec,timeout,depth_tolerance]).all() or
                not line.depth_min+.025<=target_depth<=line.depth_max-.025 or
                not hold.depth_min<=target_depth<=hold.depth_max or
                not .005<=depth_tolerance<=.10 or
                not 1.<=stable_sec<=5. or not 10.<=timeout<=60.):
            raise ValueError('Invalid dive preparation limits')
        self.line,self.hold=line,hold
        if line.control_period!=hold.control_period:
            raise ValueError('Preparation and hold control periods must match')
        self.target_depth=target_depth
        self.depth_tolerance=float(depth_tolerance)
        self.acquisition_depth_max=hold.depth_max
        self.require_stationary=require_stationary
        self.stable_sec,self.timeout=stable_sec,timeout
        self.preview_state=None;self.phase='unplanned';self.stable_since=None
        self.started=None;self.last_update=None
        self.blocking_reasons=[]
        self.message='While disarmed, press R to inspect the trajectory after descent'

    @property
    def active(self):return self.phase in ('diving','waiting_vision')

    def prepare(self,state):
        state=np.asarray(state,dtype=float)
        if state.shape!=(12,) or not np.isfinite(state).all():raise ValueError('Invalid dive preview state')
        if -state[1]<self.hold.depth_min:
            raise ValueError('Starting depth is below the configured minimum')
        projected=state.copy();projected[1]=-self.target_depth
        plan=self.line.plan(projected)
        self.preview_state=projected;self.stop()
        return dict(plan,stage='surface_preview',dive_target_depth_m=self.target_depth,
                    depth_target_policy='direct_setpoint',preparation_timeout_sec=self.timeout,
                    required_stable_sec=self.stable_sec,source_state_unity=state.tolist(),
                    preparation_depth_tolerance_m=self.depth_tolerance,
                    require_stationary_before_dive=self.require_stationary,
                    anchor_policy='fresh_pose_and_heading_at_target_depth',
                    surface_preview_enforced=False,blind_preparation_horizontal_output='zero')

    def stop(self):
        self.phase='prepared' if self.preview_state is not None else 'unplanned'
        self.stable_since=None
        self.blocking_reasons=[]

    def start(self,state,observation,now):
        if self.preview_state is None:raise ValueError('While disarmed, press R to preview and inspect the trajectory')
        state=np.asarray(state,dtype=float)
        if state.shape!=(12,) or not np.isfinite(state).all():raise ValueError('Invalid preparation start state')
        if self.require_stationary and (max(abs(state[6:9]))>.05 or max(abs(state[9:]))>.15):
            raise ValueError('Hold still before preparing to descend')
        if max(abs(state[[3,5]]))>np.deg2rad(15):raise ValueError('Level the vehicle before preparing to descend')
        if abs(-state[1]-observation['depth_m'])>.04:raise ValueError('EKF and pressure depths disagree; descent blocked')
        if not np.isfinite(list(observation.values())).all() or observation['depth_m']<self.hold.depth_min:
            raise ValueError('Invalid depth or preparation sensors')
        self.validate_start(state)
        # A vehicle already below the setpoint must be allowed to ascend.
        # Only acquisition may cover the observed initial depth; this envelope
        # never expands while active and is removed on return to normal depth.
        self.acquisition_depth_max=max(self.hold.depth_max,observation['depth_m']+.02)
        self.hold.enter(observation,target_depth=self.target_depth)
        # Command the final depth immediately. Force, attitude and output slew
        # remain bounded by the pressure/IMU controller, independent of vision.
        self.hold.target_depth=self.target_depth
        self.started=self.last_update=now;self.phase='diving'
        self.stable_since=None
        self.blocking_reasons=[]
        self.line.elapsed=0.;self.message=f'Moving to {self.target_depth:.2f} m: vertical thrusters only; horizontal output is zero'

    def validate_start(self,state):
        """Specialized fixed-world preparations can revalidate their preview."""

    def solve(self,observation,state=None):
        return self.hold.solve(observation,observation_depth_max=self.acquisition_depth_max)

    def validate_update(self,observation,now):
        if not self.active:raise ValueError('Dive preparation is not active')
        if now-self.started>=self.timeout:raise TimeoutError('Descent/vision stabilization timed out; control disabled')
        if not np.isfinite(list(observation.values())).all():raise ValueError('Invalid preparation sensors')
        if not self.hold.depth_min<=observation['depth_m']<=self.acquisition_depth_max:
            raise ValueError('Preparation depth outside the permitted range; control disabled')
        if observation['depth_m']<=self.hold.depth_max:
            self.acquisition_depth_max=self.hold.depth_max
        if max(abs(observation[k]) for k in ('roll_rad','pitch_rad'))>np.deg2rad(20):
            raise ValueError('Preparation tilt exceeds 20°; control disabled')
        elapsed=now-self.last_update
        if not 0<=elapsed<=control_time_limits(self.line.control_period)[1]:raise TimeoutError('Preparation control cycle timed out')
        self.last_update=now

    def ready_to_track(self,state,now):
        return self.finish_tracking(state)

    def finish_tracking(self,state):
        final=np.asarray(state,dtype=float).copy();final[1]=-self.target_depth
        self.line.plan(final,from_preparation=True)
        self.line.reset(state,require_stationary=False)
        self.phase='tracking';self.message='Target depth reached; vision and EKF ready; starting planar tracking'
        return True

    def update(self,observation,state,now):
        """state is None whenever a fully fresh visual EKF snapshot is absent."""
        self.validate_update(observation,now)
        if self.phase=='diving':
            if abs(observation['depth_m']-self.target_depth)<=self.depth_tolerance:
                self.phase='waiting_vision'
        blockers=[]
        def require(condition,message):
            if not condition:blockers.append(message)
        require(self.phase=='waiting_vision','Depth adjustment in progress')
        error=abs(observation['depth_m']-self.target_depth)
        require(error<=self.depth_tolerance,
                f'Pressure depth error {error*100:.1f}>{self.depth_tolerance*100:g} cm')
        if state is not None:
            state=np.asarray(state,dtype=float)
            if state.shape!=(12,) or not np.isfinite(state).all():raise ValueError('Invalid fresh preparation state')
        else:blockers.append('Waiting for a fresh vision EKF state')
        self.blocking_reasons=blockers
        if not blockers:
            if self.stable_since is None:self.stable_since=now
            if now-self.stable_since>=self.stable_sec:
                # Acquire the origin once at depth, independently of the
                # surface preview. Tracking checks use this fixed new anchor.
                return self.ready_to_track(state,now)
        else:self.stable_since=None
        if self.phase=='waiting_vision':
            progress=0. if self.stable_since is None else now-self.stable_since
            detail=blockers[0] if blockers else 'Depth reached; vision and EKF ready'
            self.message=f'Preparing to track {progress:.1f}/{self.stable_sec:.1f}s：{detail}'
        return False


def box_qp(hessian,gradient,lower,upper,initial,iterations=400,tolerance=2e-5):
    """Convex box QP, with projected KKT residual reported explicitly."""
    hessian,gradient,lower,upper,initial=[np.asarray(a,dtype=np.float64)
                                       for a in (hessian,gradient,lower,upper,initial)]
    n=len(gradient)
    if (n<1 or hessian.shape!=(n,n) or any(a.shape!=(n,) for a in (gradient,lower,upper,initial))
            or not all(np.isfinite(a).all() for a in (hessian,gradient,lower,upper,initial))
            or np.any(lower>upper) or iterations<1 or iterations!=int(iterations)
            or not np.isfinite(tolerance) or tolerance<=0):
        raise ValueError('Expected finite consistent QP arrays and valid iteration limits')
    diagonal=np.sqrt(np.maximum(np.diag(hessian),1e-10))
    h=hessian/diagonal[:,None]/diagonal[None,:]
    g=gradient/diagonal
    lo,hi=lower*diagonal,upper*diagonal
    lipschitz=float(np.linalg.eigvalsh(h)[-1])
    x=np.empty(n);residual_out=np.empty(1)
    tick=native_kernel().box_qp_scaled(n,int(iterations),tolerance,lipschitz,
        *[np.ascontiguousarray(a) for a in (h,g,lo,hi,initial*diagonal)],x,residual_out)
    if tick<1 or not np.isfinite(x).all():raise RuntimeError('Native QP received invalid scaling or output')
    residual=float(residual_out[0])
    return x/diagonal,dict(iterations=tick,kkt_residual=residual,converged=residual<tolerance)


class FossenMPC:
    def __init__(self,plant,force_map,budget_sec=None,*,control_period=DT):
        self.control_period=control_period_seconds(control_period)
        self.plant,self.force_map=plant,force_map
        self.budget=control_time_limits(self.control_period)[0] if budget_sec is None else budget_sec
        self.force_slew=.7*(self.control_period/DT)
        self.weights=np.array([0.,300.,0.,80.,60.,80.,120.,160.,120.,12.,20.,12.])
        self.sequence=np.zeros((H,8));self.previous=np.zeros(8)
        self.move=np.zeros((H*8,32))
        for j in range(H):self.move[j*8:(j+1)*8,(j//2)*8:(j//2+1)*8]=np.eye(8)

    def reset(self,forces=None):
        values=np.zeros(8) if forces is None else np.asarray(forces,dtype=float)
        if (values.shape!=(8,) or not np.isfinite(values).all() or
                np.any(values<self.force_map.lower-1e-8) or np.any(values>self.force_map.upper+1e-8)):
            raise ValueError('Invalid MPC warm-start force')
        self.previous=values.copy();self.sequence=np.repeat(values[None],H,axis=0)

    def reset_applied(self,throttle,enabled_mask):
        """Seed from MCU output, allowing only the known 3D DShot rounding.

        Firmware maps each sign to 999 intervals and reports the decoded
        float32 throttle. A saturated .18 command therefore reports 180/999.
        Keep that actual force for slew/cost; project only the optimizer seed
        into its unchanged continuous command bounds. Do not clip telemetry.
        """
        values=np.asarray(throttle,dtype=float)
        quantized_limit=float(np.floor(np.float32(np.float32(self.force_map.limit)*999)+.5)/999)
        if (values.shape!=(8,) or not np.isfinite(values).all()
                or np.any(abs(values)>quantized_limit+2e-8)):
            raise ValueError('Applied MCU throttle exceeds DShot-quantized control limit')
        over=abs(values)>self.force_map.limit+2e-8
        if np.any(abs(abs(values[over])*999-np.rint(abs(values[over])*999))>3e-5):
            raise ValueError('Applied MCU throttle is not a valid DShot rounding level')
        applied=self.force_map.estimator.estimate(values,enabled_mask)
        if not applied['valid']:raise ValueError('Cannot reconstruct MCU forces at method transition')
        actual=np.asarray(applied['force_along_positive_throttle_axis_n'])
        seed=np.clip(actual,self.force_map.lower,self.force_map.upper)
        self.reset(seed)
        self.previous=actual.copy()
        return dict(source='actual_MCU_output',actual_force_n=actual.tolist(),
            feasible_seed_force_n=seed.tolist(),projection_n=(seed-actual).tolist(),
            throttle_limit=self.force_map.limit,quantized_throttle_limit=quantized_limit)

    def cost(self,prediction,references,forces):
        error=prediction-references;error[:,3:6]=wrapped(error[:,3:6])
        weights=np.r_[np.ones(7),3.]
        differences=forces-np.vstack([self.previous,forces[:-1]])
        return float(np.sum(weights[:,None]*self.weights*error**2)+.025*np.sum(forces**2)+.06*np.sum(differences**2))

    def shifted_sequence(self):
        """Time-average continuation on the unchanged 100 ms prediction grid.

        With 50 ms replanning, each future grid cell overlaps half of its old
        cell and half of the next. Do not shift a full model step every tick.
        """
        shifted=np.vstack([self.sequence[1:],self.sequence[-1]])
        if self.control_period==DT:return shifted
        fraction=self.control_period/DT
        return (1-fraction)*self.sequence+fraction*shifted

    def solve(self,state,references):
        started=time.perf_counter()
        cpu_started=time.process_time()
        # Warm start projects the shifted trajectory onto two-step move blocks.
        shifted=self.shifted_sequence()
        guess=shifted.reshape(4,2,8).mean(1).ravel()
        lo=np.tile(self.force_map.lower,4);hi=np.tile(self.force_map.upper,4)
        # Hard first-move force slew bound, relative to command actually sent.
        lo[:8]=np.maximum(lo[:8],self.previous-self.force_slew)
        hi[:8]=np.minimum(hi[:8],self.previous+self.force_slew)
        guess=np.clip(guess,lo,hi)
        forces=(self.move@guess).reshape(H,8)
        nominal=self.plant.rollout(state,forces)
        if hasattr(self.plant,'trajectory_sensitivity'):
            jmat=self.plant.trajectory_sensitivity(state,forces,self.move)
        else:
            a,b=self.plant.linearize(np.vstack([state,nominal[:-1]]),forces)
            sensitivities=[];s=np.zeros((12,32))
            for j in range(H):
                s=a[j]@s+b[j]@self.move[8*j:8*(j+1)]
                sensitivities.append(s)
            jmat=np.asarray(sensitivities).reshape(96,32)
        error=nominal-references;error[:,3:6]=wrapped(error[:,3:6])
        w=(np.r_[np.ones(7),3.][:,None]*self.weights).ravel()
        diff=np.eye(H*8)
        for j in range(1,H):diff[j*8:(j+1)*8,(j-1)*8:j*8]=-np.eye(8)
        dm=diff@self.move
        prev=np.r_[self.previous,np.zeros(56)]
        regularizer=.025*self.move.T@self.move+.06*dm.T@dm
        hessian=2*(jmat.T@(w[:,None]*jmat)+regularizer)
        gradient=2*(jmat.T@(w*error.ravel())-jmat.T@(w[:,None]*jmat)@guess-.06*dm.T@prev)
        candidate,info=box_qp(hessian,gradient,lo,hi,guess)
        candidate_forces=(self.move@candidate).reshape(H,8)
        prediction=self.plant.rollout(state,candidate_forces)
        baseline=self.cost(nominal,references,forces)
        value=self.cost(prediction,references,candidate_forces)
        # Backtrack the nonlinear objective, never blindly send the QP result.
        fraction=1.
        while value>baseline+1e-8 and fraction>.0625:
            fraction*=.5
            candidate_forces=(self.move@(guess+fraction*(candidate-guess))).reshape(H,8)
            prediction=self.plant.rollout(state,candidate_forces)
            value=self.cost(prediction,references,candidate_forces)
        elapsed=time.perf_counter()-started
        cpu_elapsed=time.process_time()-cpu_started
        if not info['converged']:raise RuntimeError('QP failed convergence check')
        if value>baseline+1e-8:raise RuntimeError('Nonlinear MPC cost check failed')
        if elapsed>self.budget:raise TimeoutError(f'MPC deadline exceeded: {elapsed:.3f}s (CPU {cpu_elapsed:.3f}s)')
        throttle=self.force_map.throttle(candidate_forces[0])
        self.sequence=candidate_forces.copy();self.previous=candidate_forces[0].copy()
        return throttle,dict(**info,elapsed_sec=elapsed,cost=value,warm_start_cost=baseline,
                             cpu_elapsed_sec=cpu_elapsed,
                             control_period_sec=self.control_period,prediction_dt_sec=DT,
                             line_search_fraction=fraction,force_n=candidate_forces[0].tolist(),
                             throttle=throttle.tolist(),prediction=prediction.tolist())


class PilotGate:
    def __init__(self):
        self.armed=False;self.last_start=True;self.neutral_seen=False

    def tick(self,command,healthy,can_arm):
        edge=command.start and not self.last_start
        self.last_start=command.start
        if not command.start and command.neutral():self.neutral_seen=True
        if not healthy:self.armed=False;self.neutral_seen=False
        if edge:
            if self.armed:self.armed=False
            elif healthy and can_arm and self.neutral_seen and command.neutral():self.armed=True
        return self.armed

    def stop(self):
        self.armed=False;self.neutral_seen=False


class VisionFallback:
    """Bound blind hold; recover an armed session only after vision and neutral."""
    def __init__(self,timeout=3.,recovery_sec=.5):
        if not np.isfinite([timeout,recovery_sec]).all() or not .5<=timeout<=15. or not 0<recovery_sec<=timeout:
            raise ValueError('Invalid vision fallback timing')
        self.timeout,self.recovery_sec=timeout,recovery_sec
        self.reset()

    def reset(self):
        self.active=False;self.deadline=None;self.recovered_since=None;self.waiting_neutral=False

    def update(self,armed,vision_fresh,vision_age,now,*,neutral=False):
        if not armed:
            self.reset();return False
        if not self.active and vision_fresh:return False
        if not np.isfinite([vision_age,now]).all() or vision_age<0:
            raise ValueError('Invalid vision timestamp; control disabled')
        if self.waiting_neutral and vision_fresh:
            if neutral:self.reset();return False
            return True
        if not self.active or self.waiting_neutral:
            self.active=True;self.deadline=now+max(0.,self.timeout-vision_age)
            self.waiting_neutral=False;self.recovered_since=None
        # The first loss fixes the deadline. Intermittent detections must not
        # extend it, even when the reported vision age returns to zero.
        if now>=self.deadline:
            raise TimeoutError('Vision degradation timed out; control disabled; restore localization before pressing Start again')
        if vision_fresh:
            if self.recovered_since is None:self.recovered_since=now
            if now-self.recovered_since>=self.recovery_sec:
                if neutral:self.reset();return False
                self.waiting_neutral=True;self.deadline=None
        else:self.recovered_since=None
        return True


class DepthAttitudeHold:
    """Pressure/IMU PD hold in ROS FLU; only four vertical channels may run.

    No horizontal state, absolute yaw, archived spring or joystick command is
    used. Depth is body-origin depth corrected for the pressure sensor offset.
    Gains are conservative commissioning values, not identified parameters.
    """
    def __init__(self,force_map,pressure_offset_z=-.115,net_upward_force=.084366,depth_min=.05,depth_max=.70,
                 *,control_period=DT,enforce_depth_limits=True):
        self.force_map=force_map;self.offset=float(pressure_offset_z);self.buoyancy=float(net_upward_force)
        self.depth_min,self.depth_max=depth_min,depth_max
        self.enforce_depth_limits=bool(enforce_depth_limits)
        self.control_period=control_period_seconds(control_period)
        axes=force_map.estimator.axes
        self.channels=np.flatnonzero(np.abs(axes[:,2])>.99)
        if len(self.channels)!=4 or np.any(abs(axes[self.channels,:2])>1e-8):
            raise ValueError('Expected four confirmed vertical thrusters')
        self.matrix=force_map.estimator.matrix[np.ix_([2,3,4],self.channels)]
        if np.linalg.matrix_rank(self.matrix)!=3:raise ValueError('Vertical allocation has insufficient rank')
        self.target_depth=None;self.previous=np.zeros(8)
        self.last_tick=None;self.last_depth=None;self.depth_rate=0.
        self.settings=dict(depth_kp_n_per_m=30.,depth_kd_ns_per_m=22.,
                           attitude_kp_nm_per_rad=3.,attitude_kd_nms_per_rad=.8,
                           force_slew_n_per_cycle=.35*(self.control_period/DT),
                           control_period_sec=self.control_period,
                           max_body_vertical_force_n=3.,max_torque_nm=.35,
                           throttle_limit=force_map.limit,pressure_offset_z_body_m=self.offset,
                           net_upward_force_n=self.buoyancy,depth_min_m=depth_min,depth_max_m=depth_max,
                           depth_range_stop_enabled=self.enforce_depth_limits)

    def observe(self,packet):
        depth,roll,pitch,p,q=[float(packet[k]) for k in
                            ('depth_m','roll_rad','pitch_rad','roll_rate_rps','pitch_rate_rps')]
        if not np.isfinite([depth,roll,pitch,p,q]).all():raise ValueError('Invalid hold sensors')
        depth+=self.offset*np.cos(roll)*np.cos(pitch)
        tick=int(packet['mcu_time_ms'])
        # Use the older pressure cache stamp, with MCU wraparound arithmetic.
        sample=(tick-max((tick-int(t))&0xffffffff for t in packet['pressure_time_ms']))&0xffffffff
        if self.last_tick is not None:
            elapsed=((sample-self.last_tick)&0xffffffff)*.001
            if elapsed>=.5:
                self.last_tick,self.last_depth,self.depth_rate=sample,depth,0.
                raise ValueError('Pressure derivative history interrupted')
            if elapsed>0:
                measured=np.clip((depth-self.last_depth)/elapsed,-.5,.5)
                self.depth_rate+=(-np.expm1(-elapsed/.25))*(measured-self.depth_rate)
        if sample!=self.last_tick:self.last_tick,self.last_depth=sample,depth
        return dict(depth_m=float(depth),depth_rate_mps=float(self.depth_rate),roll_rad=roll,pitch_rad=pitch,
                    roll_rate_rps=p,pitch_rate_rps=q)

    def enter(self,observation,*,target_depth=None):
        self.target_depth=float(observation['depth_m'] if target_depth is None else target_depth)
        if not np.isfinite(self.target_depth):
            raise ValueError('Invalid hold target depth')
        if self.enforce_depth_limits and not self.depth_min<=self.target_depth<=self.depth_max:
            raise ValueError('Degraded-mode depth outside configured range; control disabled')
        # Withdraw the old MPC output immediately. Ramping down its saturated
        # descent command would keep diving while the new controller asks up.
        # Build the new hold output from zero under its own slew limit.
        self.previous.fill(0.)

    def solve(self,observation,*,observation_depth_max=None):
        if self.target_depth is None:raise ValueError('Depth hold has not been entered')
        started=time.perf_counter()
        d=observation
        maximum=self.depth_max if observation_depth_max is None else float(observation_depth_max)
        if not np.isfinite(maximum) or maximum<self.depth_max:
            raise ValueError('Invalid depth acquisition envelope')
        if self.enforce_depth_limits and not self.depth_min<=d['depth_m']<=maximum:
            raise ValueError('Degraded-mode depth outside configured range; control disabled')
        roll,pitch=d['roll_rad'],d['pitch_rad']
        projection=np.cos(roll)*np.cos(pitch)
        if not np.isfinite(list(d.values())).all() or projection<=.49:
            raise ValueError('Invalid hold observation or attitude')
        # Positive body z is upward. Positive depth error/rate needs upward force.
        gains=self.settings
        vertical=(gains['depth_kp_n_per_m']*(d['depth_m']-self.target_depth)
                  +gains['depth_kd_ns_per_m']*d['depth_rate_mps']-self.buoyancy)/projection
        kp,kd=gains['attitude_kp_nm_per_rad'],gains['attitude_kd_nms_per_rad']
        limits=np.array([gains['max_body_vertical_force_n'],gains['max_torque_nm'],gains['max_torque_nm']])
        desired=np.clip([vertical,-kp*roll-kd*d['roll_rate_rps'],
                         -kp*pitch-kd*d['pitch_rate_rps']],-limits,limits)
        weighted=np.array([1.,8.,8.])[:,None]*self.matrix
        target=desired*np.array([1.,8.,8.])
        ids=self.channels
        lo=np.maximum(self.force_map.lower[ids],self.previous[ids]-gains['force_slew_n_per_cycle'])
        hi=np.minimum(self.force_map.upper[ids],self.previous[ids]+gains['force_slew_n_per_cycle'])
        force,qp=vertical_box_qp(2*(weighted.T@weighted+1e-4*np.eye(4)),
                                 -2*weighted.T@target,lo,hi)
        if not qp['converged']:raise RuntimeError('Depth/attitude allocation did not converge')
        forces=np.zeros(8);forces[ids]=force
        throttle=self.force_map.throttle(forces);self.previous=forces.copy()
        if time.perf_counter()-started>control_time_limits(self.control_period)[0]:raise TimeoutError('Depth/attitude control deadline exceeded')
        return throttle,dict(**qp,controller='pressure_IMU_depth_attitude_PD',
                             elapsed_sec=time.perf_counter()-started,force_n=forces.tolist(),
                             requested_wrench_z_roll_pitch_ros=desired.tolist(),
                             allocated_wrench_z_roll_pitch_ros=(self.matrix@force).tolist(),
                             target_depth_m=self.target_depth,observation=d,
                             throttle=throttle.tolist())


def vertical_box_qp(hessian,gradient,lower,upper):
    """Exact four-channel allocation: at most 81 free/lower/upper active sets.

    Unlike an iterative tolerance budget, this covers the nearly singular
    redundant vertical actuator geometry even while several slew bounds bind.
    """
    for count,active in enumerate(itertools.product((0,-1,1),repeat=4),1):
        active=np.asarray(active);free=np.flatnonzero(active==0);fixed=np.flatnonzero(active!=0)
        force=np.where(active<0,lower,upper).copy()
        if len(free):
            force[free]=np.linalg.solve(hessian[np.ix_(free,free)],
                -gradient[free]-hessian[np.ix_(free,fixed)]@force[fixed])
        if np.any(force<lower-1e-9) or np.any(force>upper+1e-9):continue
        force=np.clip(force,lower,upper)
        derivative=hessian@force+gradient
        residual=float(np.max(abs(force-np.clip(force-derivative,lower,upper))))
        if residual<1e-6:return force,dict(iterations=count,kkt_residual=residual,converged=True)
    raise RuntimeError('No feasible converged vertical allocation')


def external_frame(throttle,enabled,sequence):
    values=np.asarray(throttle,dtype=float)
    if values.shape!=(8,) or not np.isfinite(values).all() or np.any(abs(values)>.300001):
        raise ValueError('Expected eight physical-channel throttles within .30')
    if not enabled and np.any(values):raise ValueError('Disabled command must be all zero')
    body=struct.pack('<B8f',0x80|((int(sequence)&63)<<1)|int(bool(enabled)),*values)
    crc=0xffff
    for byte in body:
        crc^=byte
        for _ in range(8):crc=(crc>>1)^(0xa001 if crc&1 else 0)
    return b'\xaa'+body+struct.pack('<H',crc)+b'\xbb'
