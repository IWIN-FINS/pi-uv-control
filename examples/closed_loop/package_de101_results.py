"""Publish finite DE101 results without rerunning historical simulations."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2];dest=repo/'research/closed_loop/de101'
    source=args.source;done=read(source/'completed.json');stats=read(source/'report/statistics.json')
    assert done['stage']=='complete' and done['attempted']==75
    records=stats['records'];assert len(records)==75 and len({r['id'] for r in records})==75
    assert sum(r['completed'] for r in records)==60 and sum(r['operable'] for r in records)==45
    assert stats['descriptive_only'] and stats['missing_RMSE_not_imputed']
    for r in records:
        if not r['completed']:assert r['metrics'] is None and r['exit_code']==20
    for p,h in read(source/'independent_review/manifest.json').items():assert sha(Path(p))==h
    dest.mkdir(exist_ok=False);manifest=[]
    files=[('design_protocol.json','design_protocol.json'),('completed.json','completed.json'),
           ('independent_review/report.md','review.md'),('independent_review/review.json','numerical_review.json'),
           ('independent_review/rolling.png','rolling.png'),('independent_review/cumulative.png','cumulative.png'),
           ('report/statistics.json','statistics.json'),('report/pairing.json','pairing.json')]
    for original,relative in files:
        src=source/original;dst=dest/relative;shutil.copyfile(src,dst);assert sha(src)==sha(dst)
        manifest.append(dict(path=dst.relative_to(repo).as_posix(),source=source.name+'/'+original,source_sha256=sha(src),sha256=sha(dst)))
    table=[]
    for r in records:
        condition,init,arm,seed=r['id'].split('/');m=r['metrics'] or {}
        row=dict(id=r['id'],condition=condition,init=int(init[5:]),arm=arm,seed=int(seed[4:]),
                 outcome=r['outcome'],completed=r['completed'],operable=r['operable'],audit_status=r['audit_status'],
                 loaded_path_rmse_mm=None if not m else m['loaded_path_rmse_m']*1000,
                 full_path_rmse_mm=None if not m else m['path_rmse_m']*1000,
                 terminal_time_s=r.get('terminal_failure',{}).get('t'),wall_seconds=r['wall_seconds'])
        row.update(m);table.append(row)
    columns=list(dict.fromkeys(k for r in table for k in r))
    with (dest/'episodes.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,lineterminator='\n');writer.writeheader();writer.writerows(table)
    with (dest/'paired_differences.csv').open('w',encoding='utf-8',newline='') as f:
        keys=['condition','comparison','init','mutually_complete','loaded_delta_mm','full_delta_mm']
        writer=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');writer.writeheader()
        for pair in stats['pairs']:
            for r in pair['paired_results']:writer.writerow(dict(condition=pair['condition'],comparison=pair['comparison'],**r))
    for name in ['episodes.csv','paired_differences.csv']:
        p=dest/name;manifest.append(dict(path=p.relative_to(repo).as_posix(),derived_from='report/statistics.json',source_sha256=sha(source/'report/statistics.json'),sha256=sha(p)))
    (dest/'publication_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(attempted=75,complete=60,terminal=15,operable=45,published_files=len(manifest))))
if __name__=='__main__':main()
