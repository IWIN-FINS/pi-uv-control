"""Offline completed-observation replay; no ROS/Unity/hardware dependency."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .predictor import Model
from .deployment import ShadowCycle


def replay(path, model_path=None, seed=281, family='de83'):
    if family=='de101':
        from .de101 import DE101Model
        if model_path is None:raise ValueError('Explicit DE101 H_bias model path required')
        model=DE101Model(model_path)
        if model.arm!='H_bias':raise ValueError('Online replay requires H_bias; use DE101Model.predict for fixed methods')
    elif family=='de83':model = Model(model_path,seed)
    else:raise ValueError('Unknown model family')
    cycle = ShadowCycle(model)
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(rows)<2:raise ValueError('Need at least two timestamped observations')
    outputs=[]
    for row in rows:
        started=time.perf_counter()
        previous=None if cycle.online.previous is None else cycle.online.previous.copy()
        report=cycle.observe(row['timestamp_s'],row['state'],row.get('completed_mean_wrench'))
        item=dict(timestamp_s=row['timestamp_s'],bias=cycle.online.online.b.tolist(),
                  accepted=False if report is None else report['accepted'])
        if previous is not None:
            base=model.predict(previous,np.array(row['completed_mean_wrench'])[None])[0]
            item['baseline_next_state']=base.tolist()
            # Optional recorded expectations are scored, never fed into adaptation.
            if 'expected_bias' in row:
                item['bias_max_error']=float(np.max(abs(cycle.online.online.b-row['expected_bias'])))
            if 'expected_baseline_velocity' in row:
                item['baseline_velocity_max_error']=float(np.max(abs(base[6:]-row['expected_baseline_velocity'])))
        item['replay_seconds']=time.perf_counter()-started
        outputs.append(item)
    return dict(kind='offline historical-observation replay; not new closed-loop or hardware timing',
                input_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                nominal=model.raw['nominal'],samples=len(rows),records=outputs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input');parser.add_argument('--model');parser.add_argument('--seed',type=int,default=281)
    parser.add_argument('--family',choices=['de83','de101'],default='de83')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();dest=Path(args.output)
    if dest.exists():raise FileExistsError('Refusing to overwrite a previous replay')
    result=replay(args.input,args.model,args.seed,args.family)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(samples=result['samples'],output=str(dest),
                         max_bias_error=max((r.get('bias_max_error',0) for r in result['records'])),
                         max_velocity_error=max((r.get('baseline_velocity_max_error',0) for r in result['records'])))))
if __name__=='__main__':main()
