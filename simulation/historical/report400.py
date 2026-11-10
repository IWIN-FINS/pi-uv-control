"""Produce paired tracking and prediction tables and one figure-eight sheet per dataset."""
from pathlib import Path
import csv
import statistics
import itertools
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collect import ROOT, ARMS, CONDS, read, sha, write

def save(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    report=ROOT/'report';report.mkdir(exist_ok=True)
    details=[];paired=[];forecasts=[]
    for g in range(10):
        group=ROOT/f'g{g+1:02}'
        fig,axes=plt.subplots(2,5,figsize=(19,7),layout='constrained')
        for k in range(10):
            armrows={}
            ax=axes[k%2,k//2]
            for j in range(4):
                rec=read(group/f'execution/episodes/episode_{k*4+j:02}.json')
                cond,_,arm,seed=rec['id'].split('/')
                m=rec['metrics']; folder=Path(rec['folder'])
                row=dict(dataset=group.name,condition=cond,seed=seed,method=arm,completed=rec['completed'],
                    operable=rec['operable'],outcome=rec['outcome'],duration_s=rec['observed_duration_s'],
                    rmse_80s_mm=m['path_rmse_m']*1000 if m else None,rmse_20_50s_mm=m['loaded_path_rmse_m']*1000 if m else None)
                details.append(row);armrows[arm]=row
                with (folder/'motor_actions.csv').open(encoding='utf-8-sig',newline='') as f:
                    samples=[r for r in csv.DictReader(f) if r['phase']=='KoopmanMpc'][::5]
                def xy(key):
                    return np.array([[float(r[f'{key}_{a}']) for r in samples] for a in (0,2)])
                if arm=='F':ax.plot(*xy('reference_pose'),'k--',label='reference',lw=1)
                line,=ax.plot(*xy('pose'),color=f'C{ARMS.index(arm)}',label=arm,lw=.8)
                if not rec['completed']:ax.plot(*xy('pose')[:,-1],'x',color=line.get_color())
            for a,b in itertools.combinations(ARMS,2):
                both=armrows[a]['completed'] and armrows[b]['completed']
                paired.append(dict(dataset=group.name,condition=cond,seed=seed,a=a,b=b,both_complete=both,
                    **{key+'_b_minus_a':armrows[b][key]-armrows[a][key] if both else None for key in ('rmse_80s_mm','rmse_20_50s_mm')}))
            pred=read(group/f'predictions/{cond}-{seed}.json')
            for line in pred['summary']:
                forecasts.append(dict(dataset=group.name,condition=cond,seed=seed,**line))
            ax.set(title=f'{cond} / {seed}',xlabel='world X (m)',ylabel='world Z (m)',aspect='equal')
        axes[0,0].legend(fontsize=7);fig.suptitle(group.name+' | full extent, x = early termination')
        fig.savefig(report/f'{group.name}-figure-eight.png',dpi=140);plt.close(fig)
    assert len(details)==400 and len(paired)==600
    save(report/'closed_loop.csv',details);save(report/'paired_differences.csv',paired)
    if forecasts:save(report/'forecasts.csv',forecasts)
    summary=[]
    for c,arm in itertools.product(CONDS,ARMS):
        rows=[r for r in details if r['condition']==c and r['method']==arm]
        assert len(rows)==20
        summary.append(dict(condition=c,method=arm,attempted=20,complete=sum(r['completed'] for r in rows),
            operable=sum(r['operable'] for r in rows),**{k:statistics.mean(r[k] for r in rows) if all(r['completed'] for r in rows) else None for k in ('rmse_80s_mm','rmse_20_50s_mm')}))
    save(report/'condition_method.csv',summary)
    write(report/'manifest.json',{p.name:sha(p) for p in report.iterdir() if p.is_file() and p.name!='manifest.json'})

if __name__=='__main__':main()
