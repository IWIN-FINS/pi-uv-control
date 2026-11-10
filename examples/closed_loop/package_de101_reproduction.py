"""Maintainer export of exact DE101 weights/configuration and FinsSim-owned source."""
import argparse,hashlib,json,shutil
from pathlib import Path

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2];source=args.source;dest=repo/'reproduction/de101'
    dest.mkdir(parents=True,exist_ok=False);items=[]
    def copy(src,relative):
        p=dest/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,p)
        assert sha(p)==sha(src)
        items.append(dict(path=p.relative_to(repo).as_posix(),source=str(src),sha256=sha(p)))
    assert read(source/'completed.json')['stage']=='complete'
    for p in sorted((source/'models').glob('train*/*/*.json')):copy(p,Path('models')/p.relative_to(source/'models'))
    for name in ['frozen_plan.json','input_freeze.json','execution_source_freeze.json','model_mapping.json','acceptance.json','build_audit.json','flow_acceptance.json']:
        copy(source/name,Path('historical')/name)
    plan=read(source/'frozen_plan.json');assert len(plan)==75
    curve=Path(next(e['value'] for e in plan[0]['environment'] if e['key']=='FINSSIM_EDMD_THRUSTER_CURVE_PATH'))
    copy(curve,Path('config/thruster_curve_config.json'))
    for p in sorted((source/'flows').glob('1101221-*.csv')):copy(p,Path('flows')/p.name)
    for key,m in read(source/'model_mapping.json').items():
        init,arm=key.split('/')
        if arm not in ('Knu','Kx','H'):continue
        p=Path(m['path']);assert sha(p)==m['sha256'];copy(p,Path('training')/('train'+init)/arm/'model.json')
        copy(p.with_name('training.json'),Path('training')/('train'+init)/arm/'training.json')
    targets=[]
    for i,r in enumerate(read(source/'acceptance.json')['objectives']):
        p=Path(r['file']);assert sha(p)==r['sha256'];parts=p.parts;init=next(v for v in parts if v.startswith('train'));arm=parts[parts.index(init)+1]
        relative=Path('objectives')/init/arm/p.name;copy(p,relative);targets.append(dict(path=relative.as_posix(),init=int(init[5:]),arm=arm,sha256=sha(p)))
    (dest/'objectives/index.json').write_text(json.dumps(targets,indent=2),encoding='utf-8')
    # Full FinsSim-authored script set, not third-party engine/package code.
    for p in sorted((source/'unity_source/Assets/Scripts').glob('*.cs')):
        assert p.name.startswith(('FinsRov','BlueRov','WaterScene'))
        copy(p,Path('native/Assets/Scripts')/p.name)
    for p in sorted((repo/'experiments/mainline_closed_loop_de101').glob('*.py')):copy(p,Path('historical/python')/p.name)
    for name in ['models.py','mainline_nominal.json']:copy(repo/'experiments/mainline_comparison'/name,Path('historical/comparison')/name)
    # Runtime manifest permits an exact external player to be checked on another host.
    player=source/'player';runtime={p.relative_to(player).as_posix():sha(p) for p in player.rglob('*') if p.is_file() and 'DoNotShip' not in str(p) and 'BurstDebugInformation' not in str(p)}
    (dest/'config/runtime_manifest.json').write_text(json.dumps(runtime,indent=2),encoding='utf-8')
    portable=[]
    for job in plan:
        env={r['key']:r['value'] for r in job['environment']};init=job['id'].split('/')[1];arm=job['id'].split('/')[2];condition=job['id'].split('/')[0]
        replacements={'FINSSIM_EDMD_MPC_DIR':'${OUTPUT_ROOT}/formal/'+job['id'],
                      'FINSSIM_EDMD_MPC_MODEL_PATH':'${BUNDLE_ROOT}/models/'+init+'/'+arm+'/operator.json',
                      'FINSSIM_EDMD_THRUSTER_CURVE_PATH':'${BUNDLE_ROOT}/config/thruster_curve_config.json',
                      'FINSSIM_DE90_FLOW_FILE':'${BUNDLE_ROOT}/flows/1101221-'+condition+'.csv'}
        for k,v in env.items():
            if ':/' in v.replace('\\','/'):assert k in replacements,(k,v)
        env.update(replacements);portable.append(dict(id=job['id'],environment=env))
    (dest/'config/plan.json').write_text(json.dumps(portable,indent=2),encoding='utf-8')
    manifest=dict(copied=items,generated=[dict(path=p.relative_to(repo).as_posix(),sha256=sha(p)) for p in [dest/'config/plan.json',dest/'config/runtime_manifest.json',dest/'objectives/index.json']])
    (dest/'publication_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(copied_files=len(items),runtime_files=len(runtime),jobs=len(plan),objectives=len(targets))))
if __name__=='__main__':main()
