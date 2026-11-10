"""Forecast all four methods on one common, recorded F trajectory."""
import csv
import json
import numpy as np
from collect import ARMS, once, sha
from finshybrid.de101 import DE101Model
from finshybrid.encoder import encode
from finshybrid.bias import BiasUpdater
from finshybrid.geometry import rotation
from finshybrid.scales import PC, PS, VS
from models import nstep
from run_offline import metric

def compare(folder, model_dir, dest, complete):
    with (folder/'online_adaptation.jsonl').open(encoding='utf-8-sig') as f:
        history=[json.loads(line) for line in f if line.strip()]
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not history:
        assert not complete
        once(dest.with_suffix('.json'),dict(source=str(folder),full=False,origins=0,summary=[]))
        return []
    small=np.vstack((history[0]['previous'],[r['observed'] for r in history]))
    n=len(small);u=np.asarray([r['input'] for r in history])
    assert np.max(abs(np.asarray([r['previous'] for r in history])-small[:-1]))<1e-12
    assert np.max(abs(np.asarray([r['t'] for r in history])-.1*np.arange(1,n)))<.003
    if complete:assert n==800
    origins=np.arange(max(0,n-8))
    obs=np.c_[small[:,:13],np.zeros((n,29))]
    forecasts=[];invalid=[]
    for arm in ARMS:
        model=DE101Model(model_dir/arm/'operator.json')
        a,b=model.a,model.b
        k=np.c_[np.r_[a[7:13,36:],a[36:,36:],a[1:7,36:]],np.r_[b[7:13],b[36:],b[1:7]]]
        kind='H' if arm=='H_bias' else arm
        z0=np.c_[obs[:,:13],np.zeros((n,23)),encode(obs,model.meta,model.weights,'G1')]
        bias=np.zeros((n,6))
        if arm in ('F','H_bias'):
            expected=nstep(z0[:-1],u,k,kind,model.delta_scale)
            if arm=='F':
                assert np.max(abs(expected[:,:13]-np.asarray([r['baseNext'] for r in history])[:,:13]))<3e-7
            else:
                updater=BiasUpdater()
                for i in range(1,n):
                    bias[i]=updater.update(obs[i,7:13]-expected[i-1,7:13])['b']
        z=z0[origins].copy();alive=np.ones(len(origins),bool);seq=[];bad=[]
        for h in range(8):
            ix=np.flatnonzero(alive)
            if len(ix):
                with np.errstate(all='ignore'):
                    z[ix]=nstep(z[ix],u[origins[ix]+h],k,kind,model.delta_scale,bias[origins[ix]])
                alive[ix]=np.isfinite(z[ix]).all(1)&(abs(z[ix,1:]).max(1)<=100)&(abs(z[ix,4]*PS[3]+PC[3])<=1.3)
            z[~alive]=np.nan;seq.append(z[:,:13].copy());bad.append(~alive.copy())
        forecasts.append(np.stack(seq,axis=1));invalid.append(np.stack(bad,axis=1))
    # Scoring starts after generating all predictions on the observational stream.
    with (folder/'motor_actions.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['phase']=='KoopmanMpc'][::5][:n]
    assert len(rows)==n and max(abs(float(r['phase_time_s'])-.1*i) for i,r in enumerate(rows))<.003
    def values(key):return np.array([[float(r[f'{key}_{j}']) for j in range(6)] for r in rows])
    pose=values('pose');wv=values('velocity');rot=rotation(pose).transpose(0,2,1)
    vel=np.c_[np.einsum('nij,nj->ni',rot,wv[:,:3]),np.einsum('nij,nj->ni',rot,wv[:,3:])]
    errors=np.full((4,len(origins),8,5),np.nan);summary=[]
    for m,arm in enumerate(ARMS):
        for h in range(1,9):
            ok=~invalid[m][:,h-1];ids=origins[ok]+h
            if len(ids):errors[m,ok,h-1]=metric(forecasts[m][ok,h-1],pose[ids],vel[ids])
            if h not in (1,2,4,8):continue
            for phase,sel in [('all',origins>=0),('loaded',(origins>=200)&(origins<=491))]:
                ix=sel&ok
                rms=np.sqrt(errors[m,ix,h-1].mean(0)).tolist() if ix.any() else None
                summary.append(dict(method=arm,horizon=h,phase=phase,common_F_completed=complete,
                    origins=int(sel.sum()),valid=int(ix.sum()),invalid=int((sel&~ok).sum()),
                    full_window_rmse=rms if complete and ix.any() and ix.sum()==sel.sum() else None,
                    valid_prefix_diagnostic_rmse=rms))
    np.savez_compressed(dest,methods=ARMS,origins=origins,invalid=np.array(invalid),squared_errors=errors)
    once(dest.with_suffix('.json'),dict(source=str(folder),full=complete,sha256=sha(dest),
        metric_order=['horizontal_m','depth_m','SO3_rad','linear_velocity_mps','angular_velocity_radps'],summary=summary))
    return summary
