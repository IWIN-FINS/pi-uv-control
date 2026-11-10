"""Portable DE101 output checks; distinct from the archived full historical audit."""
import csv,json,math,hashlib
from pathlib import Path
import numpy as np
from finshybrid.de101 import DE101Model
from finshybrid.bias import BiasUpdater
from finshybrid.geometry import rotation
from finshybrid.scales import VS

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return [r for r in csv.DictReader(f) if r.get('phase','KoopmanMpc')=='KoopmanMpc']
def audit(folder,model_path,flow_path,complete):
    folder=Path(folder);model=DE101Model(model_path);online=lines(folder/'online_adaptation.jsonl');aligned=lines(folder/'observation_alignment.jsonl')
    assert len(aligned)>=len(online)+1
    if complete:assert len(aligned)==800 and len(online)==799
    z=np.array([r['previous'] for r in online]);inputs=np.array([r['input'] for r in online])
    assert len(z)>0
    full=np.c_[z[:,:13],np.zeros((len(z),23)),z[:,13:]]
    pred=model.step(full,inputs);compact=np.c_[pred[:,:13],pred[:,36:]]
    saved=np.array([r['baseNext'] for r in online]);base_error=float(abs(compact-saved).max());assert base_error<2e-8
    innovation=np.array([r['observed'][7:13] for r in online])-pred[:,7:13]
    updater=BiasUpdater();bias_error=0.;lift_error=0.
    for k,r in enumerate(online):
        assert abs(r['t']-(k+1)*.1)<.003 and abs(r['previousTime']-k*.1)<.003
        for name,index in [('previous',k),('observed',k+1)]:
            a=aligned[index];p=np.asarray(a['afterPose'],np.float32).astype(float);wv=np.asarray(a['afterWorldVelocity'],np.float32).astype(float)
            rr=rotation(p);v=np.r_[rr.T@wv[:3],rr.T@wv[3:]].astype(np.float32).astype(float)
            actual=model.lift(np.r_[p,v])[0];small=np.r_[actual[:13],actual[36:]]
            lift_error=max(lift_error,float(abs(small-r[name]).max()))
        assert lift_error<3e-7
        assert not r['acceptedP'] and not np.any(r['poseBias'])
        if model.arm=='H_bias':
            update=updater.update(innovation[k]);assert bool(update['accepted'])==r['acceptedV']
            expected=update['b']
        else:
            assert not r['acceptedV'];expected=np.zeros(6)
        bias_error=max(bias_error,float(abs(expected-r['velocityBias']).max()));assert bias_error<2e-7
    flow=rows(folder/'current_velocity_trace.csv');physical=rows(folder/'current_physics.csv')
    actual=np.array([[float(r[f'flow_{a}_world_mps']) for a in 'xyz'] for r in flow]);expected=np.loadtxt(flow_path,delimiter=',')
    assert len(actual)<=4000 and len(actual)>0
    flow_error=float(abs(actual-expected[:len(actual)]).max());assert flow_error<3e-6
    physical_complete=len(physical)==len(actual)
    if complete:assert physical_complete and len(actual)==4000
    profile=read(folder/'physical_profile_fixed.json');snap=read(folder/'model_parameters_baseline_start.json')
    assert all(json.loads(snap[k])==profile for k in ('truthBefore','truthAfter','truthEnd'))
    assert snap['predictor']==model.raw['nominal']
    motors=rows(folder/'motor_actions.csv');assert len(motors)>0
    sq=np.array([sum((float(r[f'pose_{i}'])-float(r[f'reference_pose_{i}']))**2 for i in (0,2)) for r in motors])
    result=dict(audit_scope='portable model/causal-bias/observed-lift/flow/fixed-profile checks; not the complete historical audit',
                base_error=base_error,bias_error=bias_error,lift_error=lift_error,flow_error=flow_error,
                full_physical_trace_available=physical_complete,completed=bool(complete),
                path_rmse_m=None,loaded_path_rmse_m=None)
    if complete:
        assert len(motors)==4000
        t=np.array([float(r['phase_time_s']) for r in motors]);mask=(t>=20)&(t<50)
        result.update(path_rmse_m=math.sqrt(float(sq.mean())),loaded_path_rmse_m=math.sqrt(float(sq[mask].mean())))
    return result
