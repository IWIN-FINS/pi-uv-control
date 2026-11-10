"""Freeze the experimental design and collect thirty new simulation seeds."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import msvcrt

ROOT=Path(__file__).resolve().parent
REPO=Path('D:/code/Underwater-Adaptive-Dynamics')
TOOLS=Path('D:/code/FinsSim/tools')
OLD=ROOT.parent/'de101-mainline-failure-inclusive-20261027'
POOL=ROOT.parent/'de65-current-data-scaling-20261015'
BUNDLE=REPO/'reproduction/de101'
VAL=REPO/'data/validation.npz'
TRUTH=OLD/'formal/still/train281/F/seed1101221/invariant-seed1101221/physical_profile_fixed.json'
ARMS=('F','Kx','H','H_bias')
CONDS=('still','steady','slow','fast','nonperiodic')
POLICIES=('matched','inertia','damping','inertia_damping')
def trainseeds(g):return [3117001+3*g+j for j in range(3)]
def testseed(e,r):return 3117221+2*e+r
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(exist_ok=True,parents=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def once(p,x):
    if p.exists():assert read(p)==json.loads(json.dumps(x)),str(p)
    else:write(p,x)
def state(stage,**kw):write(ROOT/'execution/status.json',dict(stage=stage,unix=time.time(),pid=os.getpid(),**kw))
def lock(name):
    p=ROOT/'execution'/name;p.parent.mkdir(exist_ok=True,parents=True)
    f=p.open('a+b')
    if not f.tell():f.write(b'0');f.flush()
    f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);return f
def collect():
    handle=lock('collection.lock')
    assert int(subprocess.check_output(['powershell','-NoProfile','-Command',
        "@(Get-Process FinsROVHybridBasis -ErrorAction SilentlyContinue).Count"]).strip())==0
    source=read(POOL/'collection/queue.json')['jobs']
    templates={j['id'].split('/')[1]:j for j in source if j['id'].startswith('train/')}
    plan=[]; groups=[]
    for g in range(10):
        group=f'g{g+1:02}'; groups.append(dict(group=group,training_seeds=trainseeds(g)))
        for seed in trainseeds(g):
            for j in range(4):
                policy=POLICIES[(j+seed-3117001)%4]
                env={v['key']:v['value'] for v in templates[policy]['environment']}
                jid=f'{group}/{policy}/seed{seed}'
                env.update(FINSSIM_EDMD_MPC_BATCH_SEEDS=str(seed),FINSSIM_EDMD_MPC_DIR=str(ROOT/'collection'/jid))
                assert env['FINSSIM_DE62_COLLECTION']=='1' and env['FINSSIM_HYBRID_CURRENT_MODE']=='gauss_markov'
                plan.append(dict(id=jid,environment=[dict(key=k,value=v) for k,v in env.items()]))
    nominal=read(BUNDLE/'models/train281/F/operator.json')['nominal']
    assert nominal==read(REPO/'experiments/mainline_comparison/mainline_nominal.json')
    true_profile=read(TRUTH)
    assert true_profile==json.loads(read(POOL/'data/audit.json')['training_plant_profile'])
    # Ensure the shared validation *arrays* and segment identities are the original ones.
    import numpy as np
    with np.load(VAL) as a,np.load(POOL/'data/n3060/validation.npz') as b:
        assert all(np.array_equal(a[k],b[k]) for k in ('x','y','u','pose','velocity','previous','seg'))
    rounds=[dict(condition=c,seed=testseed(e,r)) for e,c in enumerate(CONDS) for r in range(2)]
    assert len({s for g in groups for s in g['training_seeds']}|{r['seed'] for r in rounds})==40
    protocol=dict(groups=groups,training=dict(trajectories_per_group=12,transitions_per_group=3060,
        validation_common=1020,validation_sha256=sha(VAL),initialization=281,updates=1200,batch=128,
        training_methods=['Kx','H'],online='H_bias shares all weights with H. Only causal velocity bias adapts.',
        Kx='fit full velocity and pose increment from scratch using DE101 initialization and training recipe'),
        frozen_fossen_baseline=str(BUNDLE/'models/train281/F/operator.json'),plant_profile=str(TRUTH),
        nominal=nominal,truth_sha256=sha(TRUTH),rounds=rounds,methods=list(ARMS),closed_loop_attempts=400,
        pairing='Same ten test conditions across datasets. Each four-arm round uses common initial pose, noise draws, flow table and joint8 reference. Rotate controller order by dataset and round.',
        forecast='Common recorded F trajectory from the corresponding round. Only completed past transitions for bias. Fixed origin bias and recorded future inputs for h=1,2,4,8. Score against held-out simulator truth. Keep invalid and missing counts.',
        reporting='All attempts, no performance retries. Full and loaded RMSE missing on incomplete episodes. Known model domain exits retained. Unknown errors stop with logs.')
    once(ROOT/'design_protocol.json',protocol)
    once(ROOT/'collection_plan.json',plan)
    sys.path.insert(0,str(TOOLS))
    import de62_runtime as rt
    binary=POOL/'player/FinsROVHybridBasis_Data/Managed/Assembly-CSharp.dll'
    assert sha(binary)==read(POOL/'build_audit.json')['assembly_sha256']
    once(ROOT/'build_audit.json',read(POOL/'build_audit.json'))
    once(ROOT/'collection_freeze.json',{str(p):sha(p) for p in [Path(__file__),ROOT/'collection_plan.json',ROOT/'design_protocol.json',binary,TRUTH,VAL,POOL/'collection/queue.json',TOOLS/'de62_runtime.py']})
    state('collecting',episodes_planned=120)
    rt.ROOT=ROOT;rt.PLAYER=POOL/'player/FinsROVHybridBasis.exe';rt.run_queue('collection',plan)
    assert read(ROOT/'collection/queue.json.completed.json')['jobs']==120
    write(ROOT/'collection_finished.json',dict(episodes=120,unix=time.time()))
    state('collection_complete',episodes=120)
if __name__=='__main__':
    try:collect()
    except Exception:
        import traceback
        state('collection_error',traceback=traceback.format_exc());raise
