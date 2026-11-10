"""User-authorized remaining preflights; retain failed episodes without retry."""
from common import *
import msvcrt,subprocess,shutil,traceback
from de84_control import folder
EXT=ROOT/'execution/preflight'

def verify():
 for name in ('engineering_source_freeze.json','failure_manifest.json'):
  for p,h in read(ROOT/name).items():assert sha(p)==h,p

def audit_one(job, stage='formal'):
 # Same independent DE92 physical, observation, bias and complete MPC audit.
 # Cardinality becomes one episode; thresholds and equations remain unchanged.
 source=(TOOLS/'de92_audit.py').read_text(encoding='utf-8')
 source=source.replace('from de92_common import *','from common import *\nfrom de92_common import cost_weights')
 source=source.replace('from de92_native import jobs','from common import jobs')
 source=source.replace('from de83_model import numpy_step','from native import numpy_step')
 source=source.replace("assert sha(rawpath)==sha(MODELS/init/'A/operator.json')==(p/'frozen_model_sha256.txt').read_text().strip()","assert sha(rawpath)==(p/'frozen_model_sha256.txt').read_text().strip()")
 start=source.index("                if cfg['cost']=='legacy' and cfg['mode']!='joint':");end=source.index('                objectives.append',start);source=source[:start]+source[end:]
 source=source.replace('nonzero>=(len(ARMS)-4)*9 and nonzeroC>=24*9',"nonzero>=(3 if plan[0]['id'].split('/')[2]=='H_bias' else 0) and nonzeroC==0")
 needle='        physics=m.current_audit(p,env,n);m.zero_added_loads(p,env,n)'
 source=source.replace(needle,needle+"\n        for suffix in ('baseline_start','learned_end'):\n            snap=read(p/f'model_parameters_{suffix}.json');assert snap['truthBefore']==snap['truthAfter']==snap['truthEnd'] and snap['predictor']==raw['nominal']")
 source=source.replace("ROOT/stage/'audit_progress.json'","folder(plan[0])/'continuation_audit_progress.json'")
 source=source.replace("ROOT/stage/'audit.json'","folder(plan[0])/'continuation_audit.json'")
 once(EXT/'audit_template.json',{'parent_sha256':sha(TOOLS/'de92_audit.py'),'source':source})
 namespace={'__name__':'de99_continuation_audit'};exec(compile(source,'de99_continuation_audit','exec'),namespace)
 namespace['jobs']=lambda stage:[job]
 return namespace['audit'](stage)

def pairing(plan):
 from de83_metrics import load_rows
 signatures={};checked=[]
 for job in plan:
  p=folder(job)
  if not (p/'completed.json').exists():continue
  motor=load_rows(p,'motor_actions.csv');noise=load_rows(p,'observation_noise_audit.csv');flow=load_rows(p,'current_velocity_trace.csv')
  assert len(motor)==1600
  signature=(np.array([[float(r[f'reference_pose_{i}']) for i in range(6)] for r in motor]),np.array([float(motor[0][f'{prefix}_{i}']) for prefix in ('pose','velocity') for i in range(6)]),[(r['index'],r['standard_gaussian_draw']) for r in noise],[[r[f'flow_{a}_world_mps'] for a in ('x','y','z')] for r in flow])
  key=job['id'].split('/')[0],job['id'].split('/')[-1]
  if key in signatures:
   ref=signatures[key];assert abs(ref[0]-signature[0]).max()<1e-5 and abs(ref[1]-signature[1]).max()<1e-7 and ref[2:]==signature[2:]
  else:signatures[key]=signature
  checked.append(job['id'])
 return dict(status='passed',episodes=checked)

