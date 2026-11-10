from finshybrid.experiment_paths import config_path, load_record
"""Serial, checkpointed offline comparison. Never launches Unity/closed-loop."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse, csv, datetime, hashlib, json, time, traceback, gc, msvcrt
from pathlib import Path
import numpy as np
import torch
from models import Comparison, initial, nstep, save_model, load_model
from finshybrid.train import load, windows, EagerTrainer
from finshybrid.torch_model import Predictor, GraphTrainer, fixture, score
from finshybrid.encoder import encode, unpack
from finshybrid.predictor import Model
from finshybrid.bias import BiasUpdater
from finshybrid.geometry import rotation
from finshybrid.scales import PC,PS,VS

REPO=config_path("repository")
SOURCE=config_path("de92_formal")
ARMS=('Knu','Kx','H')
METHODS=('F','Knu','Kx','H','H_bias')
CONDITIONS=('still','steady','slow','fast','nonperiodic')
SEEDS=(281,282,283)
TRACES=(1092211,1092212,1092213)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(p)
def status(root,stage,**kw):dump(root/'status.json',dict(stage=stage,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
def folder(c,i,s):return SOURCE/f'{c}/train{i}/fixed_legacy_n40_f98/seed{s}/invariant-seed{s}'
def verify(root):
    for p,h in json.loads((root/'freeze.json').read_text()).items():
        if sha(p)!=h:raise RuntimeError(f'Frozen source/input changed: {p}')

def acceptance(root,data,valid):
    status(root,'numerical_acceptance')
    hashes=[]
    for d in (data,valid):
        hashes.append({hashlib.sha256(np.c_[d['x'][d['seg']==s],d['u'][d['seg']==s],d['y'][d['seg']==s]].tobytes()).hexdigest() for s in np.unique(d['seg'])})
    assert not hashes[0]&hashes[1], 'Identical trajectories across train/validation'
    # Independent target geometry check, including body-origin (not COM) positions.
    from scipy.spatial.transform import Rotation
    pose0=data['x'][:,1:7]*PS+PC;pose1=data['y'][:,1:7]*PS+PC
    r0=rotation(pose0);r1=rotation(pose1)
    dr=Rotation.from_matrix(r0.transpose(0,2,1)@r1).as_rotvec()
    assert np.max(abs(r0@Rotation.from_rotvec(dr).as_matrix()-r1))<1e-12
    batch=fixture(valid,windows(valid)[:3]); records=[]
    for arm in ARMS:
        meta,w,k,ls,ds=initial(data,281,arm)
        p=Comparison(meta,w,k,arm,ds)
        x,u,y=batch;tx,tu,ty=map(p.t,batch)
        z=np.c_[x[:,:13],np.zeros((len(x),23)),encode(x,meta,w,'G1')];ref=[]
        for ui in u:z=nstep(z,ui,k,arm,ds);ref.append(z)
        actual=p.rollout(tx,tu).detach().numpy()
        error=float(np.max(abs(actual-np.stack(ref))));assert error<1e-8,(arm,error)
        pars=[p.k,*p.w];anchor=[v.detach().clone() for v in pars]
        loss=p.objective(tx,tu,ty,p.t(ls),anchor);loss.backward();errors=[]
        for par in pars:
            grad=par.grad.detach().numpy().copy();ix=np.unravel_index(abs(grad).argmax(),grad.shape);old=par[ix].item()
            with torch.no_grad():
                par[ix]=old+1e-6;hi=p.objective(tx,tu,ty,p.t(ls),anchor).item()
                par[ix]=old-1e-6;lo=p.objective(tx,tu,ty,p.t(ls),anchor).item();par[ix]=old
            errors.append(abs((hi-lo)/2e-6-grad[ix])/max(1,abs(grad[ix])))
        assert max(errors)<1e-4,(arm,errors)
        ut=p.t(u).requires_grad_();val=p.rollout(tx,ut)[-1,:,1:13].square().mean();g=torch.autograd.grad(val,ut)[0].numpy()
        ix=np.unravel_index(abs(g).argmax(),g.shape);up=u.copy();um=u.copy();up[ix]+=1e-6;um[ix]-=1e-6
        fd=(p.rollout(tx,p.t(up))[-1,:,1:13].square().mean()-p.rollout(tx,p.t(um))[-1,:,1:13].square().mean()).item()/2e-6
        control=abs(fd-g[ix])/max(1,abs(g[ix]));assert control<1e-4
        if arm!='H':
            def forbidden(*a,**kw):raise AssertionError('Black box invoked Fossen')
            p.fossen=forbidden
            assert np.array_equal(actual,p.rollout(tx,tu).detach().numpy())
        else:
            old=Predictor(meta,w,k,'A');diff=float(abs(old.rollout(tx,tu).detach().numpy()-actual).max());assert diff<1e-12
            assert abs(old.objective(tx,tu,ty,p.t(ls),anchor).item()-loss.item())<1e-12
        gpu=Comparison(meta,w,k,arm,ds,'cuda');gb=tuple(gpu.t(a) for a in batch);ga=[v.detach().clone() for v in [gpu.k,*gpu.w]]
        gpu_error=float(abs(gpu.rollout(*gb[:2]).detach().cpu().numpy()-actual).max());assert gpu_error<1e-8
        gpu.objective(*gb,gpu.t(ls),ga).backward()
        ge=max(float(abs(a.grad.cpu().numpy()-b.grad.numpy()).max()) for a,b in zip([gpu.k,*gpu.w],pars));assert ge<1e-7
        driver=GraphTrainer(gpu,batch,ls);driver.graph.replay();torch.cuda.synchronize()
        gl=driver.value.item();gg=[a.grad.clone() for a in driver.pars];el=driver.gradient().item()
        graph_error=max(float(abs(a.grad-b).max()) for a,b in zip(driver.pars,gg));assert abs(gl-el)<1e-9 and graph_error<1e-7
        eager=Comparison(meta,w,k,arm,ds,'cuda');ed=EagerTrainer(eager,batch,ls)
        _,ok1=driver.update(batch,1);_,ok2=ed.update(batch,1)
        update_error=max(float(abs(a-b).max()) for a,b in zip(driver.pars,ed.pars));assert ok1==ok2 and update_error<1e-9
        records.append(dict(arm=arm,numpy8=error,parameter_gradient=max(errors),control_gradient=control,cuda8=gpu_error,cuda_gradient=ge,graph_gradient=graph_error,update_error=update_error))
        print(records[-1],flush=True)
        del driver,ed,eager,gpu;gc.collect();torch.cuda.empty_cache()
    # Whole segments must be contiguous; labels must form actual adjacent sequences.
    for d in (data,valid):
        for s in np.unique(d['seg']):
            ids=np.flatnonzero(d['seg']==s);assert np.all(np.diff(ids)==1)
            assert np.max(abs(d['y'][ids[:-1],1:13]-d['x'][ids[1:],1:13]))<1e-8
    # Same encoder/batches and Knu/Kx shared velocity/latent ridge initialization.
    aa=initial(data,281,'Knu');bb=initial(data,281,'Kx')
    assert np.array_equal(aa[2][:16],bb[2][:16])
    for a,b in zip(aa[1],bb[1]):assert np.array_equal(a,b)
    frozen=Model();meta,w,k,_,_=unpack(frozen.raw);kk=np.r_[k,np.zeros((6,16))]
    z=frozen.lift(np.r_[PC,np.zeros(6)]);uu=np.ones((1,6))*.01
    assert abs(nstep(z,uu,kk,'H',np.ones(6),np.zeros(6))-frozen.step(z,uu)).max()<1e-10
    zero=np.zeros_like(kk);h=nstep(z,uu,zero,'H',np.ones(6));f=nstep(z,uu,zero,'F',np.ones(6));assert np.array_equal(h,f)
    # Online bias at an origin only consumes completed prefix transitions.
    rng=np.random.default_rng(51);residuals=rng.normal(size=(80,6))*.01
    def sequence(values):
        b=BiasUpdater();return np.array([b.update(v)['b'] for v in values])
    changed=residuals.copy();changed[40:]+=100
    assert np.array_equal(sequence(residuals)[:40],sequence(changed)[:40])
    target=residuals[:20];weights=.98**np.arange(19,-1,-1)
    candidate=.2*(weights@target)/(weights.sum()+1.5)
    np.testing.assert_allclose(sequence(target)[-1],candidate,atol=1e-14,rtol=0)
    dump(root/'acceptance.json',dict(status='passed',records=records,segment_continuity=True,identical_train_validation_trajectories=0,mainline_equivalence=True,blackbox_no_fossen=True,rotation_labels=True,bias_prefix_causality=True,bias_independent_first_update=True))

def train(root,data,valid):
    ids=windows(data);vi=windows(valid)[::8]
    for seed in SEEDS:
      for arm in ARMS:
        dest=root/f'models/train{seed}/{arm}';dest.mkdir(parents=True,exist_ok=True)
        done=dest/'training.json'
        if done.exists():
            assert sha(dest/'model.json')==json.loads(done.read_text())['model_sha256'];continue
        meta,w,k,ls,ds=initial(data,seed,arm);p=Comparison(meta,w,k,arm,ds,'cuda')
        initial_value=score(p,valid,vi)
        # Do not silently weaken pure black-box targets to make initialization pass.
        if not np.isfinite(initial_value):raise RuntimeError(f'Invalid initial rollout {seed}/{arm}; preserve and diagnose')
        rng=np.random.default_rng(seed+510);driver=GraphTrainer(p,fixture(data,rng.choice(ids,128,False)),ls)
        best=(initial_value,p.arrays(),0);trace=[];accepted=0;start_tick=1
        checkpoint=dest/'checkpoint.pt'
        if checkpoint.exists():
            state=torch.load(checkpoint,weights_only=False)
            for a,b in zip(driver.pars,state['parameters']):a.data.copy_(b)
            driver.first=state['first'];driver.second=state['second'];driver.anchor=state['anchor']
            # Captured objective anchors refer to original initial values, unchanged by resume.
            rng.bit_generator.state=state['rng'];best=state['best'];trace=state['trace'];accepted=state['accepted'];start_tick=state['tick']+1
        started=time.monotonic()
        for tick in range(start_tick,1201):
            loss,ok=driver.update(fixture(data,rng.choice(ids,128,False)),tick);accepted+=int(ok)
            if tick%10==0:status(root,'training',init=seed,arm=arm,step=tick,total_steps=1200,complete_models=len(list((root/'models').glob('train*/*/training.json'))),total_models=9)
            if tick%100==0:
                value=score(p,valid,vi)
                trace.append(dict(step=tick,validation8=value if np.isfinite(value) else None,loss=loss if np.isfinite(loss) else None,accepted=accepted,segment_elapsed_s=time.monotonic()-started))
                if value<best[0]:best=(value,p.arrays(),tick)
                print(seed,arm,trace[-1],flush=True)
                temp=dest/'checkpoint.tmp'
                torch.save(dict(tick=tick,parameters=[v.detach().clone() for v in driver.pars],first=driver.first,second=driver.second,anchor=driver.anchor,rng=rng.bit_generator.state,best=best,trace=trace,accepted=accepted),temp);temp.replace(checkpoint)
        for a,b in zip(driver.pars,best[1]):a.data.copy_(p.t(b))
        save_model(dest/'model.json',p,meta,ds,arm)
        dump(done,dict(seed=seed,arm=arm,steps=1200,accepted=accepted,best_step=best[2],validation8=best[0],trace=trace,model_sha256=sha(dest/'model.json')))
        del driver,p;gc.collect();torch.cuda.empty_cache()

def metric(z,pose,velocity):
    pp=z[:,1:7]*PS+PC;vv=z[:,7:13]*VS;dp=pp-pose;dv=vv-velocity
    angle=np.arccos(np.clip((np.sum(rotation(pp)*rotation(pose),axis=(1,2))-1)/2,-1,1))
    return np.c_[np.sum(dp[:,[0,2]]**2,1),dp[:,1]**2,angle**2,np.sum(dv[:,:3]**2,1),np.sum(dv[:,3:]**2,1)]

def evaluate(root):
    summaries=[]
    for cond in CONDITIONS:
      for seed in SEEDS:
       for ts in TRACES:
        status(root,'retrospective_prediction',condition=cond,init=seed,trace_seed=ts)
        dest=root/f'predictions/{cond}-{seed}-{ts}.npz';done=dest.with_suffix('.json')
        if done.exists():
            r=json.loads(done.read_text());assert sha(dest)==r['sha256'];summaries.extend(r['summary']);continue
        base=folder(cond,seed,ts);online=[json.loads(s) for s in (base/'online_adaptation.jsonl').read_text().splitlines() if s.strip()]
        small=np.vstack((online[0]['previous'],[r['observed'] for r in online]));u=np.array([r['input'] for r in online])
        assert len(small)==800 and len(u)==799
        assert abs(np.array([r['previous'] for r in online])-small[:-1]).max()<1e-12
        assert abs(np.array([r['t'] for r in online])-.1*np.arange(1,800)).max()<.003
        x=np.c_[small[:,:13],np.zeros((800,29))]
        # Truth is scoring-only. Inputs above are the recorded observations/estimated wrench.
        with (base/'motor_actions.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=[r for r in csv.DictReader(f) if r['phase']=='KoopmanMpc'][::5]
        pose=np.array([[float(r[f'pose_{j}']) for j in range(6)] for r in rows]);wv=np.array([[float(r[f'velocity_{j}']) for j in range(6)] for r in rows])
        assert len(rows)==800
        assert max(abs(float(r['phase_time_s'])-.1*i) for i,r in enumerate(rows))<.003
        rr=rotation(pose).transpose(0,2,1);velocity=np.c_[np.einsum('nij,nj->ni',rr,wv[:,:3]),np.einsum('nij,nj->ni',rr,wv[:,3:])]
        origins=np.arange(792);all_errors=[];all_bad=[];local=[]
        for method in METHODS:
            raw,meta,arrays=load_model(root/f'models/train{seed}/{method if method in ARMS else "H"}/model.json')
            k=arrays[0];arm=raw['arm'];ds=np.array(raw['delta_scale'])
            if method=='F':k=np.zeros_like(k);arm='F'
            latent=encode(x,meta,arrays[1:],'G1');bias=np.zeros((800,6))
            if method=='H_bias':
                previous=np.c_[x[:-1,:13],np.zeros((799,23)),latent[:-1]]
                forecast=nstep(previous,u,k,arm,ds)
                updater=BiasUpdater()
                for i in range(1,800):bias[i]=updater.update(x[i,7:13]-forecast[i-1,7:13])['b']
            z=np.c_[x[origins,:13],np.zeros((792,23)),latent[origins]];errors=[];bad=[];dead=np.zeros(792,bool)
            for h in range(1,9):
                with np.errstate(all='ignore'):z=nstep(z,u[origins+h-1],k,arm,ds,bias[origins])
                dead|=~np.isfinite(z).all(1)|(abs(z[:,1:]).max(1)>100)|(abs(z[:,4]*PS[3])>1.3)
                errors.append(metric(z,pose[origins+h],velocity[origins+h]));bad.append(dead.copy())
            error=np.stack(errors,1);invalid=np.stack(bad,1);all_errors.append(error);all_bad.append(invalid)
            for label,mask in [('loaded',(origins>=200)&(origins<=491)),('all',origins>=0)]:
                rms=np.sqrt(np.mean(error[mask],axis=0))
                local.append(dict(condition=cond,init=seed,trace_seed=ts,method=method,interval=label,rmse_h1_to_h8=rms.tolist() if np.isfinite(rms).all() else None,invalid=int(invalid[mask].sum())))
        dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,errors=np.array(all_errors),invalid=np.array(all_bad),origins=origins,methods=METHODS)
        dump(done,dict(sha256=sha(dest),summary=local));summaries.extend(local)
    dump(root/'prediction_summary.json',dict(scope='retrospective common-trajectory forecasts, NOT closed-loop tracking or independent confirmation',metric_order=['horizontal_m','depth_m','SO3_rad','linear_velocity_mps','angular_velocity_radps'],records=summaries))
    status(root,'offline_computed_pending_independent_audit',complete_models=9,complete_traces=45,closed_loop_started=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--acceptance-only',action='store_true');args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    try:
        if not (root/'freeze.json').exists():
            files=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.json'))+list((REPO/'src/finshybrid').glob('*.py'))+list((REPO/'data').glob('*.npz'))
            files+=list((REPO/'src/finshybrid/assets').glob('train*/operator.json'))
            for c in CONDITIONS:
             for i in SEEDS:
              for s in TRACES:
               files += [folder(c,i,s)/n for n in ('online_adaptation.jsonl','motor_actions.csv','observation_alignment.jsonl')]
            dump(root/'freeze.json',{str(p):sha(p) for p in files})
        verify(root)
        data=load(REPO/'data/train.npz');valid=load(REPO/'data/validation.npz')
        if not (root/'acceptance.json').exists():acceptance(root,data,valid)
        assert json.loads((root/'acceptance.json').read_text())['status']=='passed'
        if args.acceptance_only:
            status(root,'acceptance_passed_training_not_started');return
        train(root,data,valid);evaluate(root);verify(root)
        from report_offline import report
        report(root)
    except Exception:
        dump(root/'failure.json',dict(traceback=traceback.format_exc()));status(root,'failed');raise

if __name__=='__main__':main()
