"""Prespecified retrospective summaries, paired crossed intervals and figures."""
import json,hashlib
from pathlib import Path
import numpy as np


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=json.loads((root/'prediction_summary.json').read_text())['records']
    methods=['F','Knu','Kx','H','H_bias'];conditions=['still','steady','slow','fast','nonperiodic']
    shape=(5,3,3,5,8,5);rms=np.full(shape,np.nan);invalid=0
    for r in records:
        if r['interval']!='loaded':continue
        invalid+=r['invalid']
        if r['rmse_h1_to_h8'] is not None:rms[conditions.index(r['condition']),r['init']-281,r['trace_seed']-1092211,methods.index(r['method'])]=r['rmse_h1_to_h8']
    # Four prespecified pairs x two physical endpoints at H8 = family8.
    # Mean per-trajectory RMSE, not sqrt of pooled MSE. Crossed init/trace seeds.
    comparisons=[];rng=np.random.default_rng(20261027);ii=rng.integers(0,3,(20000,3));ss=rng.integers(0,3,(20000,3))
    for baseline,candidate in [('F','H'),('H','H_bias'),('Knu','H'),('Knu','Kx')]:
        for metric in [0,3]:
            diff=(rms[1:,:,:,methods.index(candidate),7,metric]-rms[1:,:,:,methods.index(baseline),7,metric]).mean(0)
            if not np.isfinite(diff).all():
                comparisons.append(dict(baseline=baseline,candidate=candidate,metric=metric,status='invalid_predictions_no_interval'));continue
            samples=diff[ii[:,:,None],ss[:,None,:]].mean((1,2))
            comparisons.append(dict(baseline=baseline,candidate=candidate,metric=['path_m','depth_m','SO3_rad','linear_velocity_mps','angular_velocity_radps'][metric],mean=float(diff.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),ci99375=np.quantile(samples,[.003125,.996875]).tolist()))
    analysis=dict(scope='retrospective prediction only; not closed-loop RMSE or independent confirmation',family=8,bootstrap=20000,invalid_predictions=invalid,comparisons=comparisons)
    (root/'analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False))
    mean=rms.mean((1,2))
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,metric,label,scale in [(axes[0],0,'H8 horizontal prediction RMSE (mm)',1000),(axes[1],3,'H8 linear velocity prediction RMSE (m/s)',1)]:
        for j,name in enumerate(methods):ax.plot(conditions,mean[:,j,7,metric]*scale,'o-',label=name)
        ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(root/'conditions.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,metric,label,scale in [(axes[0],0,'Position (mm)',1000),(axes[1],3,'Linear velocity (m/s)',1),(axes[2],2,'SO(3) (deg)',180/np.pi)]:
        for j,name in enumerate(methods):ax.plot(np.arange(1,9),mean[1:,j,:,metric].mean(0)*scale,'o-',label=name)
        ax.set_xlabel('Prediction horizon');ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(root/'horizons.png',dpi=170);plt.close(fig)
    text=['# Five-arm offline comparison','', '**Retrospective forecasts, not closed-loop tracking results.**', '', 'Mean per-trajectory loaded H8 horizontal prediction RMSE (mm):','', '|Method|Still|Steady|Slow|Fast|Nonperiodic|','|---|---:|---:|---:|---:|---:|']
    for j,name in enumerate(methods):text.append('|'+name+'|'+'|'.join(f'{v:.4f}' for v in mean[:,j,7,0]*1000)+'|')
    text+=['',f'Invalid forecast flags: {invalid}. No failed trajectories removed.', '', 'Primary family: four pairs × path/linear velocity at H8, equal-weight four disturbed environments. 20,000 paired crossed resamples of initialization and trace-seed blocks; 95% and 99.375% intervals. These are retrospective descriptive evidence, not new independent confirmation.', '', 'F has no trained initialization; its predictions still differ across the three recorded controller-trajectory blocks. This blocking does not create three trained F models.', '', 'H_bias reuses H weights. All future inputs are recorded completed estimated inputs supplied as conditioned replay; no future observations enter the predictor. Training and online observation distributions remain distinct.', '', 'Native five-arm MPC, physical pairing and timing preflight are still required before closed-loop. Source/input freeze and all original outputs retained. Independent audit and visual inspection pending.']
    (root/'report.md').write_text('\n'.join(text),encoding='utf-8')
    paths=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.png','.md') and p.name not in ('status.json','artifact_manifest.json')]
    (root/'artifact_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
