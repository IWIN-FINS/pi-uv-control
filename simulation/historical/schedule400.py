"""Bind legacy DE101 settings and verify the common paired environment."""
import csv
import subprocess
import numpy as np
from collect import ROOT, REPO, OLD, BUNDLE, ARMS, CONDS, testseed, read, once, sha
from de90_flows import table

def junction(target, source):
    if not target.exists():
        cmd=f"New-Item -ItemType Junction -Path '{target}' -Target '{source.resolve()}' | Out-Null"
        subprocess.run(['powershell','-NoProfile','-Command',cmd],check=True)
    assert target.resolve()==source.resolve()

def freeze_plan():
    template={tuple(j['id'].split('/')[:3]):j['environment'] for j in read(BUNDLE/'config/plan.json')}
    plan=[];inputs={};flowdir=ROOT/'flows';flowdir.mkdir(exist_ok=True)
    for g in range(10):
        group=ROOT/f'g{g+1:02}';group.mkdir(exist_ok=True)
        for name in ('player','unity_source'):junction(group/name,OLD/name)
        once(group/'design_protocol.json',read(ROOT/'design_protocol.json'))
        once(group/'build_audit.json',read(OLD/'build_audit.json'))
        for e,c in enumerate(CONDS):
            for r in range(2):
                seed=testseed(e,r);path=flowdir/f'{seed}-{c}.csv';flow=table(seed,c)
                if not path.exists():np.savetxt(path,flow,delimiter=',',fmt='%.17g')
                assert flow.shape==(4000,3) and np.array_equal(flow,np.loadtxt(path,delimiter=','))
                inputs[str(path)]=sha(path);paired=[]
                for j in range(4):
                    arm=ARMS[(g+e*2+r+j)%4];jid=f'{c}/train281/{arm}/seed{seed}'
                    env=dict(template[c,'train281',arm])
                    env.update(FINSSIM_DE90_FLOW_FILE=str(path),FINSSIM_EDMD_MPC_BATCH_SEEDS=str(seed),
                        FINSSIM_EDMD_MPC_DIR=str(group/'formal'/jid),
                        FINSSIM_EDMD_MPC_MODEL_PATH=str(group/'models/train281'/arm/'operator.json'),
                        FINSSIM_EDMD_THRUSTER_CURVE_PATH=str(BUNDLE/'config/thruster_curve_config.json'))
                    assert env['FINSSIM_BODY_TASK']=='joint8' and env['FINSSIM_BODY_EVAL_SECONDS']=='80'
                    assert env['FINSSIM_DE56_PRIOR']=='mismatched'
                    paired.append({k:v for k,v in env.items() if k not in ('FINSSIM_EDMD_MPC_DIR','FINSSIM_EDMD_MPC_MODEL_PATH','FINSSIM_DE83_ONLINE','FINSSIM_DE92_MODE')})
                    plan.append(dict(group=group.name,id=jid,environment=[dict(key=k,value=v) for k,v in env.items()]))
                assert all(v==paired[0] for v in paired)
    assert len(set((j['group'],j['id']) for j in plan))==400
    once(ROOT/'plan400.json',plan);once(ROOT/'flows_freeze.json',inputs)
    for p,h in read(BUNDLE/'config/runtime_manifest.json').items():assert sha(OLD/'player'/p)==h,p
    return plan

def signature(folder):
    def columns(name,keys):
        with (folder/name).open(encoding='utf-8-sig',newline='') as f:
            return [tuple(r[k] for k in keys) for r in csv.DictReader(f) if r.get('phase','KoopmanMpc')=='KoopmanMpc']
    m=columns('motor_actions.csv',[f'{name}_{i}' for name in ('reference_pose','pose','velocity') for i in range(6)])
    n=columns('observation_noise_audit.csv',['index','standard_gaussian_draw'])
    flow=columns('current_velocity_trace.csv',[f'flow_{a}_world_mps' for a in 'xyz'])
    return np.array([r[:6] for r in m],float),np.array(m[0][6:],float),n,flow

def pair(a,b):
    r=min(len(a[0]),len(b[0]));n=min(len(a[2]),len(b[2]));f=min(len(a[3]),len(b[3]))
    assert min(r,n,f)>0 and np.max(abs(a[0][:r]-b[0][:r]))<1e-5 and np.max(abs(a[1]-b[1]))<1e-7
    assert a[2][:n]==b[2][:n] and a[3][:f]==b[3][:f]
    return dict(reference=r,noise=n,flow=f)
