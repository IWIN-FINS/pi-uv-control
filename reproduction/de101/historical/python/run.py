"""Serial 75-attempt runner. Expected model failures continue; engineering errors stop."""
from common import *
import subprocess,time,traceback,msvcrt,math
from de84_control import folder
from validation import observation_module,audit_complete,audit_partial

def verify():
 for name in ('input_freeze.json','execution_source_freeze.json'):
  for p,h in read(ROOT/name).items():assert sha(p)==h,p

def freeze():
 files=list(HERE.glob('*.py'))+list(TOOLS.glob('*.py'))+list((REPO/'experiments/mainline_comparison').glob('*.py'))+list((REPO/'src/finshybrid').rglob('*.py'))
 files+=list((PARENT/'unity_source/Assets/Scripts').glob('*.cs'))+[ROOT/'player/FinsROVHybridBasis_Data/Managed/Assembly-CSharp.dll',ROOT/'acceptance.json']
 once(ROOT/'execution_source_freeze.json',{str(p):sha(p) for p in files})

def run_one(index,job):
 p=folder(job);out=ROOT/f'execution/queues/job{index:02}';out.mkdir(parents=True,exist_ok=True)
 assert not p.exists(),f'Incomplete output preserved; manual engineering review required: {p}'
 queue=out/'queue.json';once(queue,dict(jobs=[job],assembly_sha256=read(ROOT/'build_audit.json')['assembly_sha256'],design_sha256=sha(ROOT/'design_protocol.json')))
 env={k:v for k,v in os.environ.items() if not k.startswith('FINSSIM_')};env.update({r['key']:r['value'] for r in job['environment']});env['FINSSIM_BODY_JOB_QUEUE']=str(queue)
 log=out/'player.log';start=time.time()
 proc=subprocess.Popen([str(ROOT/'player/FinsROVHybridBasis.exe'),'-batchmode','-nographics','-logFile',str(log)],cwd=ROOT/'player',env=env)
 dump(out/'process.json',dict(pid=proc.pid,start_unix=start,id=job['id']))
 while proc.poll() is None:
  status('formal_running',attempted=index,total=75,current=job['id'],unity_pid=proc.pid,elapsed_wall_s=time.time()-start)
  time.sleep(10)
  if proc.poll() is None and log.exists() and 'Exception:' in log.read_text(encoding='utf-8',errors='replace'):
   proc.terminate();proc.wait();raise RuntimeError('Unexpected native exception; logs retained '+str(log))
 code=proc.returncode;dump(out/'exit.json',dict(code=code,wall_seconds=time.time()-start))
 terminal=p/'de100_terminal_failure.json';complete=p/'completed.json'
 if code==20 and terminal.exists() and not complete.exists():
  term=read(terminal);assert term['site']=='De83Observe' and term['reason']=='Body-frame Euler chart near singularity',term
  outcome='terminal_domain_failure'
 else:
  assert code==0 and complete.exists() and Path(str(queue)+'.completed.json').exists(),(code,job['id'])
  outcome='complete'
 result=dict(id=job['id'],folder=str(p),outcome=outcome,completed=outcome=='complete',exit_code=code,wall_seconds=time.time()-start)
 # Persist termination before auditing: a failed audit never causes a rerun.
 dump(out/'outcome.json',result);return result

def finish_episode(job,result):
 from de83_metrics import load_rows
 p=folder(job);motor=load_rows(p,'motor_actions.csv');assert motor
 rmse=math.sqrt(sum(sum((float(r[f'pose_{i}'])-float(r[f'reference_pose_{i}']))**2 for i in (0,2)) for r in motor)/len(motor))
 result.update(physics_samples=len(motor),observed_duration_s=float(motor[-1]['phase_time_s']),saved_prefix_path_rmse_m=rmse)
 if result['completed']:
  audit=audit_complete(job);rec=audit['records'][0];result.update(audit=audit,audit_status='passed',operable=rec['operable'],metrics=rec['metrics'])
  assert abs(result['metrics']['path_rmse_m']-rmse)<1e-12
 else:
  result.update(audit=audit_partial(job),audit_status='incomplete_available_evidence_only',operable=False,metrics=None,terminal_failure=read(p/'de100_terminal_failure.json'))
 result['output_hashes']={str(f):sha(f) for f in p.rglob('*') if f.is_file()}
 return result

def main():
 ext=ROOT/'execution';ext.mkdir(exist_ok=True)
 lock=(ext/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 active=subprocess.run(['powershell','-NoProfile','-Command',"Get-Process FinsROVHybridBasis -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id"],capture_output=True,text=True)
 assert not active.stdout.strip(),'Existing Unity process; do not duplicate'
 assert read(ROOT/'acceptance.json')['status']=='passed'
 observation_module();freeze();verify()
 import de84_pcore_resume as affinity
 affinity.ROOT=ROOT;affinity.install_affinity()
 records=[];plan=read(ROOT/'frozen_plan.json')
 for index,job in enumerate(plan):
  verify();saved=ext/f'episodes/episode_{index:02}.json'
  if saved.exists():
   r=read(saved)
   for f,h in r['output_hashes'].items():assert sha(f)==h,f
  else:
   outcome=ROOT/f'execution/queues/job{index:02}/outcome.json'
   result=read(outcome) if outcome.exists() else run_one(index,job)
   status('auditing_episode',attempted=index+1,total=75,current=job['id'])
   r=finish_episode(job,result);dump(saved,r)
  records.append(r)
  dump(ROOT/'progress_summary.json',dict(attempted=len(records),total=75,completed=sum(x['completed'] for x in records),terminal_failures=sum(not x['completed'] for x in records),records=[{k:v for k,v in x.items() if k not in ('output_hashes','audit')} for x in records]))
  print(f"ATTEMPT {index+1}/75 {job['id']} {r['outcome']} audit={r['audit_status']}",flush=True)
 verify();status('reporting',attempted=75,total=75)
 env=dict(os.environ);env.pop('PYTHONPATH',None)
 subprocess.run(['D:/ProgramData/anaconda3/python.exe',str(HERE/'report.py')],env=env,check=True)
 verify();dump(ROOT/'computed.json',dict(status='computed_pending_independent_visual_review',attempted=75,report=str(ROOT/'report/report.md')))
 status('computed_pending_independent_visual_review',attempted=75,total=75)
if __name__=='__main__':
 try:main()
 except Exception:status('engineering_failed',traceback=traceback.format_exc());raise
