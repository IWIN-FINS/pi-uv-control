"""Reconstruct DE65-format training transitions on fresh whole trajectories."""
import sys
import hashlib
import numpy as np
from collect import ROOT, TOOLS, TRUTH, POLICIES, trainseeds, once, sha, read
sys.path.append(str(TOOLS))
from de65_common import rows, base, US
from body_frame_math import to_body

def extract(folder, flows, seed):
    marker=read(folder/'completed.json')
    assert marker['baseline_samples']==256 and marker['compensated_samples']==0
    truth=read(TRUTH)
    import json
    for phase in ('baseline_start','baseline_end'):
        snap=read(folder/f'model_parameters_{phase}.json')
        assert all(json.loads(snap[k])==truth for k in ('truthBefore','truthAfter','truthEnd'))
    flow=rows(folder/'current_velocity_trace.csv')
    assert len(flow)==1280 and all(r['phase']=='BaselineMpc' for r in flow)
    ff=np.array([[float(r[f'flow_{a}_world_mps']) for a in 'xyz'] for r in flow])
    assert 0<np.linalg.norm(ff,axis=1).max()<=.150001 and np.max(abs(ff[:,1]))==0
    if seed in flows:np.testing.assert_array_equal(ff,flows[seed])
    flows[seed]=ff
    motor=rows(folder/'motor_actions.csv')
    assert len(motor)==1280 and all(r['phase']=='BaselineMpc' for r in motor)
    assert max(abs(float(r['phase_time_s'])-i*.02) for i,r in enumerate(motor))<.001
    def values(key):
        return np.array([[float(r[f'{key}_{i}']) for i in range(6)] for r in motor])
    pose=values('pose'); vel=to_body(pose,values('velocity'))
    inp=np.full((1280,6),np.nan); actual=inp.copy()
    for r in rows(folder/'body_input_steps.csv'):
        i=round(float(r['phase_time_s'])/.02);j=int(r['axis'])
        assert r['phase']=='BaselineMpc' and np.isnan(inp[i,j])
        inp[i,j]=float(r['estimated_body_input']);actual[i,j]=float(r['true_body_input'])
    assert np.max(abs(inp-actual))<1e-4
    u=inp.reshape(256,5,6).mean(1)[:255]/US
    prev=np.zeros_like(u);prev[1:]=u[:-1];ids=np.arange(0,1275,5)
    d=dict(x=base.physical_features(pose[ids],vel[ids],prev),y=base.physical_features(pose[ids+5],vel[ids+5],u),
           u=u,pose=pose[ids],velocity=vel[ids],previous=prev*US)
    assert np.max(abs(d['x'][1:,1:36]-d['y'][:-1,1:36]))<1e-12
    return d

def prepare(g):
    group=ROOT/f'g{g+1:02}'; group.mkdir(exist_ok=True)
    train=group/'train.npz'; audit=group/'dataset.json'
    if audit.exists():
        record=read(audit);assert record['sha256']==sha(train)
        return record
    parts=[];records=[];flows={}
    for seed in trainseeds(g):
        for policy in POLICIES:
            folder=ROOT/'collection'/group.name/policy/f'seed{seed}'/f'invariant-seed{seed}'
            d=extract(folder,flows,seed)
            digest=hashlib.sha256(b''.join(d[k].tobytes() for k in ('x','y','u'))).hexdigest()
            records.append(dict(seed=seed,policy=policy,digest=digest,source={f.name:sha(f) for f in folder.iterdir() if f.name in ('motor_actions.csv','body_input_steps.csv','current_velocity_trace.csv','completed.json','current_model_config.json','model_parameters_baseline_start.json','model_parameters_baseline_end.json')}))
            d['seg']=np.full(255,len(parts),dtype=np.int64);parts.append(d)
    data={k:np.concatenate([d[k] for d in parts]) for k in parts[0]}
    assert len(data['x'])==3060 and len(base.windows(data,8))==2976
    np.savez_compressed(train,**data)
    record=dict(records=records,sha256=sha(train),transitions=3060,windows8=2976)
    once(audit,record)
    return record
