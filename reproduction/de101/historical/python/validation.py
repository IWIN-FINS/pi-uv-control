"""Independent replay using unchanged tolerances and the reused repaired binary."""
from common import *
import types
from native import numpy_step
from de84_control import folder

def observation_module():
 import de84_metrics as m
 source=(TOOLS/'de84_observation.py').read_text()
 old="np.asarray(r['measuredPosition'], np.float32)+age32*np.asarray(r['measuredWorldVelocity'], np.float32)"
 new="(np.asarray(r['measuredPosition'], np.float32).astype(float)+float(age32)*np.asarray(r['measuredWorldVelocity'], np.float32).astype(float)).astype(np.float32)"
 assert source.count(old)==1;source=source.replace(old,new)
 mod=types.ModuleType('de101_cv');mod.__file__=str(TOOLS/'de84_observation.py');exec(compile(source,mod.__file__,'exec'),mod.__dict__)
 m.replay_alignment=mod.replay
 once(ROOT/'execution/observation_replay_source.json',dict(source=source,tolerance=3e-7,parent_sha256=sha(TOOLS/'de84_observation.py')))
 return mod

def namespace():
 source=read(PARENT/'n3060_u1200/execution/preflight/audit_template.json')['source']
 ns={'__name__':'de101_independent_audit'};exec(compile(source,'de101_independent_audit','exec'),ns)
 return ns

def acceptance():
 observation_module();ns=namespace();records=[];nonzero=0
 import de84_metrics as m
 m.ROOT=ROOT;m.source_contract()
 engineering=Path('D:/FinsSim-deps/diagnostics/de100-data-scaling-20261027/engineering/geometry_acceptance.json')
 assert read(engineering)['status']=='passed'
 for init in INITS:
  for arm in ARMS:
   raw=read(ROOT/f'models/train{init}/{arm}/operator.json')
   p=PARENT/f'n3060_u1200/preflight/fast/train{init}/{arm}/seed1100000/invariant-seed1100000'
   assert read(p/'compact_equivalence.json')['passed']
   candidates=sorted((p/'preflight_objectives').glob('*.json'))
   selected=[f for f in candidates if abs(float(read(f)['t']))<.01 or (arm=='H_bias' and abs(float(read(f)['t'])-8)<.01)]
   assert len(selected)==(2 if arm=='H_bias' else 1)
   for f in selected:
    r=read(f);x=np.array(r['start']);probes=np.repeat(x[None],129,axis=0)
    probes[1:65]+=np.eye(64)*1e-5;probes[65:]-=np.eye(64)*1e-5
    v=ns['objective_batch'](r,raw,probes,ns['cost_weights']('legacy'))
    g=(v[1:65]-v[65:])/2e-5;actual=np.array(r['startGradient'])
    ce=float(abs(v[0]-r['startCost'])/max(1,abs(v[0])))
    ge=float(np.max(abs(g-actual)/np.maximum(1,np.maximum(abs(g),abs(actual)))))
    assert ce<3e-6 and ge<5e-5,(f,ce,ge)
    nonzero+=int(np.linalg.norm(r['de83VelocityBias'])>1e-10)
    records.append(dict(file=str(f),sha256=sha(f),cost_error=ce,gradient_error=ge))
   if arm=='H_bias':
    alignment=m.jsonlines(p/'observation_alignment.jsonl');online=m.jsonlines(p/'online_adaptation.jsonl')
    ns['replay'](p,raw,alignment,online,CONFIG[arm],320);m.replay_alignment(p,raw,320)
 assert len(records)==18 and nonzero==3
 # The simulation uses one paired environment seed: settings differ only by
 # arm/model/online mode, not horizon, budget, trajectory, or flow file.
 plan=read(ROOT/'frozen_plan.json');groups={}
 excluded={'FINSSIM_EDMD_MPC_DIR','FINSSIM_HYBRID_MODEL_DIR','FINSSIM_DE83_ONLINE','FINSSIM_DE92_MODE'}
 # Compare exact environment values after replacing model/output paths and
 # the intentionally different online switch, retain all other keys.
 for job in plan:
  env={e['key']:e['value'] for e in job['environment']}
  canonical={k:v for k,v in env.items() if k not in excluded and '/models/' not in v.replace('\\','/')}
  key=job['id'].split('/')[0]
  if key in groups:assert groups[key]==canonical,(job['id'],set(groups[key].items())^set(canonical.items()))
  else:groups[key]=canonical
 result=dict(status='passed',objectives=records,nonzero_bias_objectives=nonzero,geometry_evidence=dict(path=str(engineering),sha256=sha(engineering)),native_binary_sha256=read(ROOT/'build_audit.json')['assembly_sha256'],same_model_coefficients=True,causal_bias_replays=3,native_equivalence_reused=15,no_new_performance_gate=True)
 dump(ROOT/'acceptance.json',result);print('Numerical acceptance passed18objectives/3nonzero plus causal replay',flush=True)

def audit_complete(job):
 ns=namespace();ns['jobs']=lambda stage:[job]
 ns['dump']=lambda path,value:dump(folder(job)/('de101_'+Path(path).name),value)
 return ns['audit']('formal')

def audit_partial(job):
 """Audit all available physical/causal prefixes; never fabricate completion."""
 import de84_metrics as m
 from de83_metrics import load_rows
 from de90_flows import flow_at
 p=folder(job);raw=read(ROOT/f"models/{job['id'].split('/')[1]}/{job['id'].split('/')[2]}/operator.json")
 env={r['key']:r['value'] for r in job['environment']};term=read(p/'de100_terminal_failure.json')
 assert term['site']=='De83Observe' and term['reason']=='Body-frame Euler chart near singularity'
 assert not (p/'completed.json').exists()
 alignment=m.jsonlines(p/'observation_alignment.jsonl');online=m.jsonlines(p/'online_adaptation.jsonl')
 assert len(alignment)>=len(online)+1 and len(alignment)<=len(online)+2
 cv=m.replay_alignment(p,raw)
 ns=namespace();rep=ns['replay'](p,raw,alignment,online,CONFIG[job['id'].split('/')[2]],len(online)+1)
 model=m.FrozenA(raw)
 for k,r in enumerate(online):
  for key,index in [('previous',k),('observed',k+1)]:assert np.max(abs(model.lift(alignment[index]['afterPose'],alignment[index]['afterWorldVelocity'])-r[key]))<3e-7
 flow=load_rows(p,'current_velocity_trace.csv');physics=m.rows(p/'current_physics.csv');assert len(flow)>0
 t=np.array([float(r['episode_time_s']) for r in flow])-.02
 assert np.max(abs(t-.02*np.arange(len(t))))<.003
 actual=np.array([[float(r[f'flow_{a}_world_mps']) for a in ('x','y','z')] for r in flow])
 expected=flow_at(t,int(env['FINSSIM_EDMD_MPC_BATCH_SEEDS']),env['FINSSIM_DE82_PROFILE'],float(env['FINSSIM_DE82_AMPLITUDE']))
 assert np.max(abs(actual-expected))<3e-6
 # In the reused native binary OnDisable can recreate the physical observer
 # on terminal shutdown, leaving only its final row. Preserve and disclose
 # this missing evidence; do not equate a source check with a full trace audit.
 complete_physics=len(physics)==len(flow)
 if complete_physics:
  native=np.array([[float(r[f'flow_{a}']) for a in ('x','y','z')] for r in physics]);assert np.max(abs(native-actual))<2e-7
 profile=read(p/'physical_profile_fixed.json');snap=read(p/'model_parameters_baseline_start.json')
 assert all(profile==json.loads(snap[k]) for k in ('truthBefore','truthAfter','truthEnd')) and snap['predictor']==raw['nominal']
 vv={k:m.array(physics,k) for k in ('ld','qd','fd','v','damping')}
 for short,long in [('ld','linearDamping'),('qd','quadraticDamping'),('fd','forwardSpeedDamping')]:
  v=vv[short];target=np.array([profile[long][a] for a in ('u','v','w','p','q','r')]);assert np.array_equal(v,np.repeat(v[:1],len(v),axis=0)) and np.max(abs(v-target)/np.maximum(1,abs(target)))<2e-7
 v=vv['v'];pred=-(abs(vv['ld'])+abs(vv['qd'])*abs(v)+abs(vv['fd'])*abs(v[:,0:1]))*v
 assert np.max(abs(pred-vv['damping'])/np.maximum(1,abs(pred)))<3e-6 and np.max(np.sum(v*vv['damping'],axis=1))<=1e-8
 m.zero_added_loads(p,env,len(flow)/5)
 return dict(status='verified_available_evidence_incomplete_episode',terminal=term,observation=cv,update_replay=rep,flow_samples=len(flow),retained_physics_samples=len(physics),full_physical_trace_available=complete_physics,limitation=None if complete_physics else 'Native terminal shutdown recreates physical observer; final row retained, whole-prefix physical audit unavailable. This is NOT a full audit pass.',full_episode_metrics_available=False)

if __name__=='__main__':acceptance()
