"""Maintainer recipe: fixed DE101 H/H_bias init281/seed1101221, all five currents."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent/'data'
    out.mkdir(exist_ok=False);records=[]
    stats=read(a.source/'report/statistics.json');saved={r['id']:r for r in stats['records']}
    for condition in ('still','steady','slow','fast','nonperiodic'):
        for arm in ('H','H_bias'):
            identifier=f'{condition}/train281/{arm}/seed1101221'
            folder=a.source/'formal'/identifier/'invariant-seed1101221';source=folder/'motor_actions.csv'
            assert saved[identifier]['completed'] and (folder/'completed.json').exists()
            with source.open(encoding='utf-8-sig',newline='') as f:rows=[r for r in csv.DictReader(f) if r['phase']=='KoopmanMpc']
            assert len(rows)==4000
            target=out/f'{condition}_{arm}.csv'
            fields=['time_s','x_m','y_m','x_ref_m','y_ref_m'];source_fields=['phase_time_s','pose_0','pose_2','reference_pose_0','reference_pose_2']
            with target.open('w',encoding='utf-8',newline='') as f:
                writer=csv.writer(f,lineterminator='\n');writer.writerow(fields)
                writer.writerows([[r[k] for k in source_fields] for r in rows])
            with target.open(encoding='utf-8',newline='') as f:data=np.array([[float(r[k]) for k in fields] for r in csv.DictReader(f)])
            sq=np.sum((data[:,1:3]-data[:,3:5])**2,axis=1);mask=(data[:,0]>=20)&(data[:,0]<50)
            full=float(np.sqrt(sq.mean()));loaded=float(np.sqrt(sq[mask].mean()))
            assert abs(full-saved[identifier]['metrics']['path_rmse_m'])<1e-12
            assert abs(loaded-saved[identifier]['metrics']['loaded_path_rmse_m'])<1e-12
            model=repo/f'reproduction/de101/models/train281/{arm}/operator.json'
            records.append(dict(id=identifier,path=target.relative_to(repo).as_posix(),sha256=sha(target),
                                source=a.source.name+'/formal/'+identifier+'/invariant-seed1101221/motor_actions.csv',
                                source_sha256=sha(source),model_sha256=sha(model),rows=4000,
                                start_s=float(data[0,0]),end_s=float(data[-1,0]),completed=True,
                                full_path_rmse_m=full,loaded_path_rmse_m=loaded))
    manifest=dict(kind='historical DE101 simulation truth, not real-robot measurements or new simulations',
                  frame='Unity world XZ, metres, body-origin position',columns='time_s; x_m=pose_0; y_m=pose_2; x_ref_m=reference_pose_0; y_ref_m=reference_pose_2',
                  selection='Fixed init281 and seed1101221, all five current conditions and both H/H_bias; no outcome selection or downsampling',
                  training='3060 training transitions / 1200 updates / DE54 mismatched physical prior; not DE103 calibrated family',
                  records=records)
    (out/'provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(exported=len(records),rows=sum(r['rows'] for r in records))))
if __name__=='__main__':main()
