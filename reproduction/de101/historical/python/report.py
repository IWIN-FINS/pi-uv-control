from finshybrid.experiment_paths import config_path, load_record
"""All-attempt descriptive report, pairing, scalar RMSE and figures; no simulation."""
import json,csv,hashlib,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=config_path("de101")
ARMS=('F','Knu','Kx','H','H_bias');CONDS=('still','steady','slow','fast','nonperiodic')
COLORS=dict(zip(ARMS,('tab:blue','tab:orange','tab:green','tab:red','tab:purple')))
def read(p):return load_record(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return [r for r in csv.DictReader(f) if r.get('phase','KoopmanMpc')=='KoopmanMpc']
def main():
 out=ROOT/'report';out.mkdir(exist_ok=True)
 rr=[read(p) for p in sorted((ROOT/'execution/episodes').glob('episode_*.json'))];assert len(rr)==75
 sigs={};checks=[];curves=[]
 for rec in rr:
  for name,digest in rec['output_hashes'].items():assert sha(Path(name))==digest,name
  p=Path(rec['folder']);motor=rows(p/'motor_actions.csv');flow=rows(p/'current_velocity_trace.csv');noise=rows(p/'observation_noise_audit.csv')
  condition,init,arm,seed=rec['id'].split('/');t=np.array([float(r['phase_time_s']) for r in motor]);err=np.array([[float(r[f'pose_{i}'])-float(r[f'reference_pose_{i}']) for i in (0,2)] for r in motor]);sq=(err*err).sum(1)
  scalar=math.sqrt(sum(sum((float(r[f'pose_{i}'])-float(r[f'reference_pose_{i}']))**2 for i in (0,2)) for r in motor)/len(motor))
  assert abs(scalar-rec['saved_prefix_path_rmse_m'])<1e-12
  if rec['completed']:
   assert len(motor)==4000 and abs(scalar-rec['metrics']['path_rmse_m'])<1e-12
   mask=(t>=20)&(t<50);assert abs(float(np.sqrt(np.mean(sq[mask])))-rec['metrics']['loaded_path_rmse_m'])<1e-12
  ref=np.array([[float(r[f'reference_pose_{i}']) for i in range(6)] for r in motor]);initial=np.array([float(motor[0][f'{k}_{i}']) for k in ('pose','velocity') for i in range(6)])
  ns=[(r['index'],r['standard_gaussian_draw']) for r in noise];fs=[[r[f'flow_{a}_world_mps'] for a in ('x','y','z')] for r in flow]
  key=(condition,seed)
  if key in sigs:
   r0,i0,n0,f0=sigs[key];n=min(len(ref),len(r0));nn=min(len(ns),len(n0));nf=min(len(fs),len(f0))
   assert np.max(abs(ref[:n]-r0[:n]))<1e-5 and np.max(abs(initial-i0))<1e-7 and ns[:nn]==n0[:nn] and fs[:nf]==f0[:nf],rec['id']
   checks.append(dict(id=rec['id'],reference_samples=n,noise_samples=nn,flow_samples=nf))
  else:sigs[key]=(ref,initial,ns,fs)
  cs=np.r_[0,np.cumsum(sq)];end=np.arange(1,len(sq)+1);start=np.maximum(0,end-50)
  curves.append(dict(id=rec['id'],condition=condition,arm=arm,init=init,completed=rec['completed'],t=t,cumulative=np.sqrt(cs[1:]/end),rolling=np.sqrt((cs[end]-cs[start])/(end-start))))
 dump(out/'pairing.json',dict(status='passed',baseline_groups=5,compared_prefixes=checks,all_attempts_included=75))
 aggregate=[]
 for c in CONDS:
  for a in ARMS:
   group=[r for r in rr if r['id'].split('/')[0]==c and r['id'].split('/')[2]==a];assert len(group)==3
   aggregate.append(dict(condition=c,arm=a,attempted=3,complete=sum(r['completed'] for r in group),operable=sum(r['operable'] for r in group),full_mm=[r['metrics']['path_rmse_m']*1000 if r['completed'] else None for r in group],loaded_mm=[r['metrics']['loaded_path_rmse_m']*1000 if r['completed'] else None for r in group],failure_time_s=[r.get('terminal_failure',{}).get('t') for r in group]))
 pairs=[]
 for c in CONDS:
  for a,b in [('H_bias','H'),('H','F'),('H','Knu'),('H','Kx'),('H_bias','Knu'),('H_bias','Kx')]:
   vals=[]
   for i in (281,282,283):
    aa=next(r for r in rr if r['id'].startswith(f'{c}/train{i}/{a}/'));bb=next(r for r in rr if r['id'].startswith(f'{c}/train{i}/{b}/'))
    vals.append(dict(init=i,mutually_complete=aa['completed'] and bb['completed'],loaded_delta_mm=(aa['metrics']['loaded_path_rmse_m']-bb['metrics']['loaded_path_rmse_m'])*1000 if aa['completed'] and bb['completed'] else None,full_delta_mm=(aa['metrics']['path_rmse_m']-bb['metrics']['path_rmse_m'])*1000 if aa['completed'] and bb['completed'] else None))
   pairs.append(dict(condition=c,comparison=f'{a}-{b}',paired_results=vals))
 dump(out/'statistics.json',dict(aggregate=aggregate,pairs=pairs,records=[{k:v for k,v in r.items() if k not in ('audit','output_hashes')} for r in rr],descriptive_only=True,missing_RMSE_not_imputed=True))
 for key,title in [('rolling','Trailing 1s horizontal RMSE'),('cumulative','Cumulative horizontal RMSE from 0s')]:
  fig,axs=plt.subplots(5,1,figsize=(12,16),sharex=True)
  for ax,c in zip(axs,CONDS):
   for rec in [r for r in curves if r['condition']==c]:
    ax.plot(rec['t'],rec[key]*1000,color=COLORS[rec['arm']],alpha=.7,lw=1,ls='-' if rec['completed'] else '--')
    if not rec['completed']:ax.scatter(rec['t'][-1],rec[key][-1]*1000,color=COLORS[rec['arm']],marker='x',s=35)
   ax.axvspan(20,50,color='gray',alpha=.1);ax.set_title(c);ax.set_ylabel('RMSE (mm), log scale');ax.set_yscale('log');ax.grid(alpha=.2)
  axs[-1].set_xlabel('Simulation time (s)');axs[-1].set_xlim(0,80)
  fig.legend([Line2D([0],[0],color=COLORS[a]) for a in ARMS],ARMS,ncol=5,loc='upper center',bbox_to_anchor=(.5,.97))
  fig.suptitle('DE101 '+title+' | one line per init, x = early termination',y=.995)
  fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/f'{key}.png',dpi=150);plt.close(fig)
 lines=['# DE101 五方法闭环对比','', '固定3060条训练转移、1200次更新，三初始化、五环境、一个新配对种子，共75次尝试。仅描述性结果，不宣称独立环境种子上的统计显著性。','', '所有失败保留。以下每行按281/282/283列出；未完成回合不计算全程或加载段正式RMSE，不能按成功者平均值进行总体排名。','', '|环境|方法|完成/3|可运行/3|20–50s路径mm|0–80s路径mm|提前终止s|','|---|---|---:|---:|---|---|---|']
 fmt=lambda vv:'/'.join('未完成' if v is None else f'{v:.3f}' for v in vv)
 for a in aggregate:lines.append(f"|{a['condition']}|{a['arm']}|{a['complete']}|{a['operable']}|{fmt(a['loaded_mm'])}|{fmt(a['full_mm'])}|{fmt(a['failure_time_s'])}|")
 lines+=['','## 解释与限制','',f"完整回合：{sum(r['completed'] for r in rr)}/75；满足原可运行标准：{sum(r['operable'] for r in rr)}/75。完整回合逐拍审计通过，提前终止仅核验留存证据，不标作完整审计通过。",'','终止时原生物理观察器可能重建日志，仅留末行；这类回合的全部物理前缀无法重审，已在逐回合记录中明确标注。海流、观测、更新和配对前缀分别核验。','', 'F沿用最终主线冻结的Fossen参数及既有失配，不是新精确标定理想模型。Kx参数量和训练目标不同；H与H_bias共享完全相同离线模型。','', '配对差、深度、SO3姿态、恢复、努力、饱和、无效/保持/超时以及更新/求解/完整周期耗时均在statistics.json逐回合保存。所有计时为串行CPU性能核4095；同步仿真不等于实机实时部署。','', '![1秒滑动RMSE](rolling.png)','', '![累计RMSE](cumulative.png)']
 (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
 dump(out/'manifest.json',{str(p):sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'})
 print('Report, pairing, scalar RMSE, figures and manifest written; visual review pending',flush=True)
if __name__=='__main__':main()
