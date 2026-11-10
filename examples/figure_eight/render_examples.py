"""Render five horizontal figure-eight examples from committed real simulation logs."""
import argparse,hashlib,json
from pathlib import Path
from finshybrid.plot_trajectory import plot_tracks

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    here=Path(__file__).resolve().parent;repo=here.parents[1]
    source=json.loads((here/'data/provenance.json').read_text(encoding='utf-8'))
    for r in source['records']:assert hashlib.sha256((repo/r['path']).read_bytes()).hexdigest()==r['sha256']
    a.output_dir.mkdir(parents=True,exist_ok=False)
    for c in ('still','steady','slow','fast','nonperiodic'):
        plot_tracks([here/f'data/{c}_H.csv',here/f'data/{c}_H_bias.csv'],
                    ['G1 offline (H)','G1 + online bias (H_bias)'],a.output_dir/(c+'.png'),
                    title=f'DE101 {c} | init281, seed1101221 | recorded 0-80s',
                    frame='Unity world XZ, body origin',axis_labels=['World X (m)','World Z (m)'])
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output_dir.iterdir() if p.is_file()}
    (a.output_dir/'manifest.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
    print('Five recorded figure-eight plots rendered; no simulation or controller executed')
if __name__=='__main__':main()
