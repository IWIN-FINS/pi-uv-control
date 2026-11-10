"""Horizontal figure-eight tracking plots from recorded trajectories (no model/control)."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np


def load_track(path,format='horizontal',phase='KoopmanMpc'):
    """horizontal CSV: time_s,x_m,y_m,x_ref_m,y_ref_m.

    Here x/y are the two declared horizontal axes, not implicit Unity/ROS axes.
    Unity input uses world pose_0/pose_2 directly; no frame rotation/recentering.
    Empty/NaN positions remain gaps. Infinity and invalid timestamps are errors.
    """
    path=Path(path)
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);names=reader.fieldnames or [];rows=list(reader)
    if format=='unity':
        rows=[r for r in rows if r.get('phase',phase)==phase]
        keys=['phase_time_s','pose_0','pose_2','reference_pose_0','reference_pose_2']
    elif format=='horizontal':keys=['time_s','x_m','y_m','x_ref_m','y_ref_m']
    else:raise ValueError('Unknown CSV format')
    if any(k not in names for k in keys):raise ValueError('Missing columns: '+str(keys))
    def value(v):return float(v) if v is not None and v.strip() else np.nan
    data=np.array([[value(r[k]) for k in keys] for r in rows],dtype=float)
    if data.ndim!=2 or data.shape[1]!=5 or len(data)<2:raise ValueError('Need at least two rows in one episode')
    if np.isinf(data).any():raise ValueError('Infinite position/reference values')
    if not np.isfinite(data[:,0]).all() or np.any(np.diff(data[:,0])<=0):
        raise ValueError('Timestamps must be finite and strictly increasing; select a single episode/phase')
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                t=data[:,0],actual=data[:,1:3],reference=data[:,3:5])


def metrics(track):
    valid=np.isfinite(track['actual']).all(1)&np.isfinite(track['reference']).all(1)
    err=np.linalg.norm(track['actual'][valid]-track['reference'][valid],axis=1)
    return dict(rows=len(valid),valid_pairs=int(valid.sum()),missing_pairs=int((~valid).sum()),
                start_s=float(track['t'][0]),end_s=float(track['t'][-1]),
                available_sample_rmse_m=None if not len(err) else float(np.sqrt(np.mean(err**2))),
                available_sample_peak_m=None if not len(err) else float(err.max()),
                meaning='sample-weighted diagnostic on available recorded pairs; not proof of full-episode completion')


def line_with_gaps(t,xy,gap_s=None):
    dt=np.diff(t);limit=3*float(np.median(dt)) if gap_s is None else float(gap_s)
    if not np.isfinite(limit) or limit<=0:raise ValueError('gap_s must be positive')
    out=[]
    for i,row in enumerate(xy):
        if i and t[i]-t[i-1]>limit:out.append([np.nan,np.nan])
        out.append(row if np.isfinite(row).all() else [np.nan,np.nan])
    return np.array(out),int(np.sum(dt>limit)),limit


def plot_tracks(inputs,labels,output,format='horizontal',phase='KoopmanMpc',
                title='Horizontal figure-eight tracking',frame='unspecified',
                axis_labels=None,gap_s=None,overwrite=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if len(inputs)!=len(labels) or not len(inputs):raise ValueError('One label required per input')
    output=Path(output);summary=output.with_suffix('.json')
    if output.suffix.lower() not in ('.png','.svg','.pdf'):raise ValueError('Output must be PNG/SVG/PDF')
    if not overwrite and (output.exists() or summary.exists()):raise FileExistsError('Previous figure/summary retained')
    if axis_labels is None:axis_labels=['World X (m)','World Z (m)'] if format=='unity' else ['Horizontal x (m)','Horizontal y (m)']
    tracks=[load_track(p,format,phase) for p in inputs]
    fig,ax=plt.subplots(figsize=(9,5.6),layout='constrained');references=[];records=[]
    try:
        for index,(label,track) in enumerate(zip(labels,tracks)):
            ref=track['reference'];seen=any(ref.shape==r.shape and np.array_equal(ref,r,equal_nan=True) for r in references)
            if not seen:
                xy,_,_=line_with_gaps(track['t'],ref,gap_s)
                ax.plot(xy[:,0],xy[:,1],ls='--',color='black' if not references else '.5',lw=1.7,
                        label='Reference' if not references else 'Reference: '+label,zorder=1)
                references.append(ref.copy())
            xy,gaps,limit=line_with_gaps(track['t'],track['actual'],gap_s)
            ax.plot(xy[:,0],xy[:,1],lw=1.3,alpha=.9,label=label,zorder=2)
            finite=np.flatnonzero(np.isfinite(track['actual']).all(1))
            if len(finite):
                start=track['actual'][finite[0]];end=track['actual'][finite[-1]]
                ax.scatter(*start,s=28,facecolors='none',edgecolors='black',linewidths=.8,zorder=3)
                ax.scatter(*end,s=22,marker='x',color='black',linewidths=.8,zorder=3)
            records.append(dict(label=label,input=track['path'],sha256=track['sha256'],
                                gaps=gaps,gap_threshold_s=limit,**metrics(track)))
        ax.set(xlabel=axis_labels[0],ylabel=axis_labels[1],title=title)
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.25);ax.legend(loc='best',fontsize=9)
        fig.text(.5,.005,'Recorded positions only | circle: first valid point | x: last valid point (not a completion claim)',ha='center',fontsize=8,color='.35')
        output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=180)
    finally:plt.close(fig)
    result=dict(kind='horizontal trajectory visualization, not new simulation',frame=frame,format=format,
                axis_labels=axis_labels,reference_curves=len(references),records=records,
                transformations='none: no smoothing, shifting, rotation, rescaling, interpolation or synthetic path fitting',
                limitation='Real positions must already share the declared frame, origin, units and reference point. Input provenance/completeness is supplied by the experiment host.')
    summary.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',action='append',required=True);p.add_argument('--label',action='append',required=True)
    p.add_argument('--output',required=True);p.add_argument('--format',choices=['horizontal','unity'],default='horizontal')
    p.add_argument('--phase',default='KoopmanMpc');p.add_argument('--frame',required=True)
    p.add_argument('--title',default='Horizontal figure-eight tracking');p.add_argument('--axis-labels',nargs=2)
    p.add_argument('--gap-s',type=float);p.add_argument('--overwrite',action='store_true')
    a=p.parse_args();result=plot_tracks(a.input,a.label,a.output,a.format,a.phase,a.title,a.frame,a.axis_labels,a.gap_s,a.overwrite)
    print(json.dumps(dict(output=a.output,records=result['records']),indent=2))
if __name__=='__main__':main()
