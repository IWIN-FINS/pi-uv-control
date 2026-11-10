from finshybrid.experiment_paths import config_path, load_record
"""Independent final scalar verification and readable review figures; no simulation."""
import csv,json,math,hashlib,datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=config_path("de101")
OUT=ROOT/'independent_review'
def read(p):return load_record(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return [r for r in csv.DictReader(f) if r.get('phase','KoopmanMpc')=='KoopmanMpc']
def main():
 OUT.mkdir(exist_ok=True);hashes={}
 for name in ('input_freeze.json','execution_source_freeze.json','report/manifest.json'):
  for p,h in read(ROOT/name).items():
   if p in hashes:assert hashes[p]==h
   hashes[p]=h
 repair=read(ROOT/'execution/truncated_log_repair/repair_protocol.json')
 script=Path(__file__).parent/'resume_truncated_log.py';assert sha(script)==repair['script_sha256']
 records=[read(p) for p in sorted((ROOT/'execution/episodes').glob('episode_*.json'))]
 plan=read(ROOT/'frozen_plan.json');assert len(records)==75 and [r['id'] for r in records]==[r['id'] for r in plan]
 for r in records:hashes.update(r['output_hashes'])
 for p,h in hashes.items():assert sha(p)==h,p
 sigs={};curves=[];maxerr=0;paired=0;stats=read(ROOT/'report/statistics.json');arms=('F','Knu','Kx','H','H_bias');conditions=('still','steady','slow','fast','nonperiodic')
 for r in records:
  p=Path(r['folder']);motor=rows(p/'motor_actions.csv');c,init,arm,seed=r['id'].split('/')
  t=[float(x['phase_time_s']) for x in motor];sq=[sum((float(x[f'pose_{i}'])-float(x[f'reference_pose_{i}']))**2 for i in (0,2)) for x in motor]
  score=math.sqrt(math.fsum(sq)/len(sq));maxerr=max(maxerr,abs(score-r['saved_prefix_path_rmse_m']))
  if r['completed']:
   assert len(sq)==4000 and r['audit_status']=='passed'
   loaded=[s for s,ti in zip(sq,t) if 20<=ti<50]
   maxerr=max(maxerr,abs(score-r['metrics']['path_rmse_m']),abs(math.sqrt(math.fsum(loaded)/len(loaded))-r['metrics']['loaded_path_rmse_m']))
  else:assert r['metrics'] is None and r['exit_code']==20 and not (p/'completed.json').exists()
  ref=np.array([[float(x[f'reference_pose_{i}']) for i in range(6)] for x in motor]);initial=np.array([float(motor[0][f'{k}_{i}']) for k in ('pose','velocity') for i in range(6)])
  noise=[(x['index'],x['standard_gaussian_draw']) for x in rows(p/'observation_noise_audit.csv')];flow=[[x[f'flow_{a}_world_mps'] for a in ('x','y','z')] for x in rows(p/'current_velocity_trace.csv')]
  key=(c,seed)
  if key in sigs:
   r0,i0,n0,f0=sigs[key];nr=min(len(ref),len(r0));nn=min(len(noise),len(n0));nf=min(len(flow),len(f0))
   assert np.max(abs(ref[:nr]-r0[:nr]))<1e-5 and np.max(abs(initial-i0))<1e-7 and noise[:nn]==n0[:nn] and flow[:nf]==f0[:nf]
   paired+=1
  else:sigs[key]=(ref,initial,noise,flow)
  cs=np.r_[0,np.cumsum(sq)];end=np.arange(1,len(sq)+1);start=np.maximum(0,end-50)
  curves.append(dict(condition=c,arm=arm,completed=r['completed'],t=t,rolling=np.sqrt((cs[end]-cs[start])/(end-start))*1000,cumulative=np.sqrt(cs[1:]/end)*1000))
 assert maxerr<1e-10 and paired==70
 for a in stats['aggregate']:
  group=[r for r in records if r['id'].split('/')[0]==a['condition'] and r['id'].split('/')[2]==a['arm']]
  assert a['complete']==sum(r['completed'] for r in group) and a['operable']==sum(r['operable'] for r in group)
  for r,full,loaded in zip(group,a['full_mm'],a['loaded_mm']):
   if r['completed']:assert abs(full-1000*r['metrics']['path_rmse_m'])<1e-9 and abs(loaded-1000*r['metrics']['loaded_path_rmse_m'])<1e-9
   else:assert full is None and loaded is None
 colors=dict(zip(arms,('tab:blue','tab:orange','tab:green','tab:red','tab:purple')))
 for key in ('rolling','cumulative'):
  fig,axes=plt.subplots(5,1,figsize=(12,15),sharex=True,layout='constrained')
  for ax,c in zip(axes,conditions):
   for r in [x for x in curves if x['condition']==c]:
    ax.plot(r['t'],r[key],color=colors[r['arm']],lw=1,alpha=.7,ls='-' if r['completed'] else '--')
    if not r['completed']:ax.scatter(r['t'][-1],r[key][-1],color=colors[r['arm']],marker='x',s=35)
   ax.axvspan(20,50,color='gray',alpha=.1);ax.set_title(c);ax.set_yscale('log');ax.set_ylim(.1,200000);ax.set_ylabel('Horizontal RMSE (mm)');ax.grid(alpha=.2)
  axes[0].legend([Line2D([0],[0],color=colors[a]) for a in arms],arms,ncol=5,loc='upper right')
  axes[-1].set_xlim(0,80);axes[-1].set_xlabel('Simulation time (s)')
  fig.suptitle('DE101 '+('Trailing 1s RMSE' if key=='rolling' else 'Cumulative RMSE from 0s')+' | individual initializations; x = terminal failure')
  fig.savefig(OUT/f'{key}.png',dpi=150);plt.close(fig)
 report=(ROOT/'report/report.md').read_text(encoding='utf-8')
 lines=[]
 for line in report.splitlines():
  if line.startswith(tuple('|'+c+'|' for c in conditions)):
   parts=line.split('|');parts[-2]=parts[-2].replace('未完成','—');line='|'.join(parts)
  lines.append(line)
 lines+=['','说明：提前终止列的“—”表示完整结束、未发生提前终止。两张复核图统一对数纵轴范围，避免零点导致刻度和布局挤压；原图和原报告保留。']
 (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
 result=dict(status='numerical_review_passed_pending_visual',attempts=75,complete=sum(r['completed'] for r in records),terminal=sum(not r['completed'] for r in records),operable=sum(r['operable'] for r in records),verified_unique_hashes=len(hashes),scalar_rmse_max_difference_m=maxerr,paired_baseline_groups=5,paired_prefix_comparisons=paired,repair_script_sha256=sha(script),review_script_sha256=sha(__file__),by_arm={a:dict(complete=sum(r['completed'] for r in records if r['id'].split('/')[2]==a),operable=sum(r['operable'] for r in records if r['id'].split('/')[2]==a)) for a in arms})
 dump(OUT/'review.json',result);dump(OUT/'manifest.json',{str(p):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'});print(json.dumps(result))
if __name__=='__main__':main()
