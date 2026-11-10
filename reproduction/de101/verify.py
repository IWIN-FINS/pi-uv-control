"""No simulator or hardware actions: validate DE101 publication and native targets."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from finshybrid.de101 import DE101Model
from finshybrid.mpc import objective_batch,central_gradient

BUNDLE=Path(__file__).resolve().parent
REPO=BUNDLE.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(full_objectives=False):
    manifest=read(BUNDLE/'publication_manifest.json')
    for r in manifest['copied']+manifest['generated']:
        assert sha(REPO/r['path'])==r['sha256'],r['path']
    count=0;max_encoder=0.;max_step=0.;h_shared=[]
    for seed in (281,282,283):
        for arm in ('F','Knu','Kx','H','H_bias'):
            folder=BUNDLE/f'models/train{seed}/{arm}';model=DE101Model(folder/'operator.json')
            for c in read(folder/'encoder_cases.json')['cases']:
                actual=model.lift(np.r_[c['p'],c['v']])[0]
                max_encoder=max(max_encoder,float(abs(actual-c['z']).max()));assert max_encoder<3e-13
            for c in read(folder/'step_cases.json')['cases']:
                actual=model.step(c['z'],c['u'])[0]
                max_step=max(max_step,float(abs(actual-c['next']).max()));assert max_step<3e-13
            count+=1
        h=read(BUNDLE/f'models/train{seed}/H/operator.json');hb=read(BUNDLE/f'models/train{seed}/H_bias/operator.json')
        h.pop('de99Arm');hb.pop('de99Arm');assert h==hb;h_shared.append(seed)
    objectives=[]
    if full_objectives:
        for entry in read(BUNDLE/'objectives/index.json'):
            r=read(BUNDLE/entry['path']);model=DE101Model(BUNDLE/f"models/train{entry['init']}/{entry['arm']}/operator.json")
            value=float(objective_batch(model,r,r['start'])[0]);gradient=central_gradient(model,r,r['start'])
            ce=abs(value-r['startCost'])/max(1,abs(value));g=np.array(r['startGradient'])
            ge=float(np.max(abs(gradient-g)/np.maximum(1,np.maximum(abs(g),abs(gradient)))))
            assert ce<3e-6 and ge<5e-5,(entry,ce,ge)
            objectives.append(dict(path=entry['path'],cost_error=ce,gradient_error=ge))
    plan=read(BUNDLE/'config/plan.json');assert len(plan)==75 and len({r['id'] for r in plan})==75
    original=read(BUNDLE/'historical/frozen_plan.json')
    path_keys={'FINSSIM_EDMD_MPC_DIR','FINSSIM_EDMD_MPC_MODEL_PATH','FINSSIM_EDMD_THRUSTER_CURVE_PATH','FINSSIM_DE90_FLOW_FILE'}
    for old,new in zip(original,plan):
        assert old['id']==new['id'];env={r['key']:r['value'] for r in old['environment']}
        assert {k:v for k,v in env.items() if k not in path_keys}=={k:v for k,v in new['environment'].items() if k not in path_keys}
    return dict(status='passed',models=count,max_encoder_error=max_encoder,max_step_error=max_step,
                shared_H_Hbias=h_shared,canonical_jobs=len(plan),full_objectives=objectives)
def main():
    p=argparse.ArgumentParser();p.add_argument('--full-objectives',action='store_true');args=p.parse_args()
    print(json.dumps(verify(args.full_objectives),indent=2))
if __name__=='__main__':main()
