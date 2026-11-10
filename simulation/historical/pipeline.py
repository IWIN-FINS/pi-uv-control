"""Sequential data conversion, training, held-out closed loops, prediction and reporting."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import time
import sys
import types
import shutil
import subprocess
import traceback
from pathlib import Path
from collect import ROOT,REPO,TOOLS,BUNDLE,OLD,VAL,TRUTH,ARMS,CONDS,read,once,write,sha,state,lock,testseed
LEGACY=REPO/'experiments/mainline_closed_loop_de101'
COMP=REPO/'experiments/mainline_comparison'
sys.path[:0]=[str(LEGACY),str(COMP),str(REPO/'src'),str(TOOLS)]
import common
import native
import numpy as np
import torch
import run_offline as trainlib
import models
from finshybrid.torch_model import nominal,score
from finshybrid.train import load,windows
from finshybrid.de101 import DE101Model
from finshybrid.scales import PC,PS,VS
from dataset import prepare
from schedule400 import freeze_plan,signature,pair
from forecast import compare as evaluate

def verify(file):
    for path,digest in read(file).items():assert sha(path)==digest,path

def freeze_sources():
    paths=[*ROOT.glob('*.py'),*LEGACY.glob('*.py'),*COMP.glob('*.py'),*COMP.glob('*.json'),
        *(REPO/'src/finshybrid').glob('*.py'),*(REPO/'src/finshybrid/assets').glob('train*/operator.json'),
        *TOOLS.glob('*.py'),ROOT/'design_protocol.json',ROOT/'plan400.json',ROOT/'flows_freeze.json',
        ROOT/'collection_freeze.json',BUNDLE/'config/runtime_manifest.json',BUNDLE/'config/plan.json',
        BUNDLE/'config/thruster_curve_config.json',OLD/'build_audit.json',VAL,TRUTH,
        OLD/'execution/truncated_log_repair/repair_protocol.json',
        common.PARENT/'n3060_u1200/execution/preflight/audit_template.json']
    paths.extend((BUNDLE/'models/train281/F').glob('*.json'))
    once(ROOT/'source_freeze.json',{str(p):sha(p) for p in paths})
    verify(ROOT/'collection_freeze.json');verify(ROOT/'flows_freeze.json')

def train_group(group,valid):
    training=group/'training';training.mkdir(exist_ok=True)
    data=load(group/'train.npz')
    state('training',group=group.name,methods=['Kx','H'],updates_each=1200)
    trainlib.ARMS=('Kx','H');trainlib.SEEDS=(281,)
    def local_status(root,stage,**kw):
        if 'total_models' in kw:kw['total_models']=2
        write(root/'status.json',dict(stage=stage,**kw))
    trainlib.status=local_status
    if not (training/'acceptance.json').exists():trainlib.acceptance(training,data,valid)
    assert read(training/'acceptance.json')['status']=='passed'
    trainlib.train(training,data,valid)
    for arm in trainlib.ARMS:
        folder=training/'models/train281'/arm
        rec=read(folder/'training.json')
        raw,meta,arrays=models.load_model(folder/'model.json')
        check=models.Comparison(meta,arrays[1:],arrays[0],arm,np.asarray(raw['delta_scale']),'cuda')
        measured=score(check,valid,windows(valid)[::8])
        assert abs(float(measured)-rec['validation8'])<1e-9*max(1,abs(measured))
        assert rec['steps']==1200 and rec['model_sha256']==sha(folder/'model.json')
    native.ROOT=group;native.OFFLINE=training;native.INITS=(281,);native.ARMS=('Kx','H','H_bias')
    native.export()
    modeldir=group/'models/train281'
    for path in (BUNDLE/'models/train281/F').glob('*.json'):
        target=modeldir/'F'/path.name;target.parent.mkdir(exist_ok=True)
        if not target.exists():shutil.copy2(path,target)
        assert sha(path)==sha(target)
    for arm in ARMS:
        folder=modeldir/arm;model=DE101Model(folder/'operator.json')
        assert model.raw['nominal']==models.NOMINAL
        for case in read(folder/'encoder_cases.json')['cases']:
            z=model.lift(np.r_[case['p'],case['v']])
            np.testing.assert_allclose(z[0],case['z'],rtol=0,atol=3e-7)
        for case in read(folder/'step_cases.json')['cases']:
            z=model.step(case['z'],case['u'])[0]
            np.testing.assert_allclose(z,case['next'],rtol=1e-10,atol=2e-8)
    h=read(modeldir/'H/operator.json');hb=read(modeldir/'H_bias/operator.json')
    assert {k:v for k,v in h.items() if k!='de99Arm'}=={k:v for k,v in hb.items() if k!='de99Arm'}
    once(group/'model_freeze.json',{str(p):sha(p) for p in sorted(modeldir.glob('*/*.json'))})
    print('TRAINED',group.name,flush=True)

def runner():
    repair=read(OLD/'execution/truncated_log_repair/repair_protocol.json')
    v=types.ModuleType('validation');v.__file__=str(LEGACY/'validation.py')
    exec(compile(repair['source'],v.__file__,'exec'),v.__dict__);sys.modules['validation']=v
    import run as legacy
    assert Path(legacy.__file__).resolve()==LEGACY/'run.py'
    return v,legacy

def formal(plan):
    v,legacy=runner()
    import de84_pcore_resume as affinity
    affinity.ROOT=ROOT;affinity.install_affinity()
    records=[]
    for g in range(10):
        group=ROOT/f'g{g+1:02}';verify(group/'model_freeze.json')
        common.ROOT=group;common.SEEDS=tuple(testseed(e,r) for e in range(5) for r in range(2))
        native.ROOT=group;legacy.ROOT=group;v.ROOT=group;v.observation_module()
        if hasattr(native.numpy_step,'lookup'):del native.numpy_step.lookup
        jobs=[j for j in plan if j['group']==group.name]
        for i,item in enumerate(jobs):
            job={k:item[k] for k in ('id','environment')}
            def tick(stage,**kw):
                kw.pop('total',None);kw.pop('attempted',None)
                state(stage,group=group.name,recorded=len(records),planned=400,**kw)
            legacy.status=tick;tick('closed_loop',current=job['id'])
            if i%4==0:sigs=[];round_records={}
            saved=group/f'execution/episodes/episode_{i:02}.json'
            if saved.exists():
                rec=read(saved);assert rec['id']==job['id']
                for p,h in rec['output_hashes'].items():assert sha(p)==h
            else:
                q=group/f'execution/queues/job{i:02}';out=q/'outcome.json'
                if out.exists():result=read(out)
                else:
                    assert not q.exists(),f'Unresolved launch evidence {q}'
                    result=legacy.run_one(i,job)
                tick('auditing',current=job['id'])
                rec=legacy.finish_episode(job,result)
            folder=Path(rec['folder'])
            assert read(folder/'physical_profile_fixed.json')==read(TRUTH)
            sig=signature(folder);rec['paired_prefixes']=[pair(s,sig) for s in sigs];sigs.append(sig)
            once(saved,rec);records.append(rec['completed'])
            c,_,arm,seed=job['id'].split('/');round_records[arm]=rec
            write(ROOT/'execution/progress.json',dict(recorded=len(records),planned=400,
              complete=sum(records),terminal_failures=len(records)-sum(records),last=group.name+'/'+rec['id']))
            if i%4==3:
                f=round_records['F'];dest=group/f'predictions/{c}-{seed}.npz'
                tick('paired_forecasts',current=c+'/'+seed)
                if not dest.with_suffix('.json').exists():evaluate(Path(f['folder']),group/'models/train281',dest,f['completed'])
                else:
                    summary=read(dest.with_suffix('.json'))
                    if 'sha256' in summary:assert sha(dest)==summary['sha256']
    assert len(records)==400
    return records

def main():
    handle=lock('pipeline.lock')
    write(ROOT/'execution/supervisor.json',dict(pid=os.getpid(),stage='preparing_frozen_queue'))
    plan=freeze_plan();freeze_sources()
    assert models.NOMINAL==nominal()==read(BUNDLE/'models/train281/F/operator.json')['nominal']
    write(ROOT/'execution/supervisor.json',dict(pid=os.getpid(),stage='waiting_for_collection',closed_loop_jobs=len(plan)))
    while not (ROOT/'collection_finished.json').exists():
        progress=read(ROOT/'execution/status.json')
        if progress['stage']=='collection_error':raise RuntimeError(progress)
        time.sleep(15)
    assert read(ROOT/'collection_finished.json')['episodes']==120
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    state('preparing_datasets');digests=set()
    for g in range(10):
        record=prepare(g);ids={r['digest'] for r in record['records']}
        assert len(ids)==12 and not ids&digests;digests.update(ids)
    assert len(digests)==120
    write(ROOT/'execution/supervisor.json',dict(pid=os.getpid(),stage='active',status='status.json'))
    valid=load(VAL)
    for g in range(10):train_group(ROOT/f'g{g+1:02}',valid)
    verify(ROOT/'source_freeze.json');verify(ROOT/'flows_freeze.json')
    active=subprocess.check_output(['powershell','-NoProfile','-Command',"@(Get-Process FinsROVHybridBasis -ErrorAction SilentlyContinue).Count"])
    assert int(active.strip())==0,'Another Unity player is running'
    records=formal(plan)
    state('reporting',recorded=400)
    subprocess.run(['D:/ProgramData/anaconda3/python.exe',str(ROOT/'report400.py')],check=True)
    verify(ROOT/'source_freeze.json');state('complete',recorded=400,completed=sum(records),failed=400-sum(records))
if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/f'execution/pipeline_error_{os.getpid()}.json',dict(traceback=traceback.format_exc()))
        write(ROOT/'execution/supervisor.json',dict(pid=os.getpid(),stage='engineering_error'))
        raise
