"""Maintainer-only finite archival export. Existing outputs are never replaced."""
import argparse,hashlib,json,shutil
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('archive_root',type=Path);args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2];dest=repo/'research/closed_loop';manifest=[]
    for study,folder in [('de104','de104-independent-confirmation-20261028'),('de105','de105-bias-boundary-20261029')]:
        source=args.archive_root/folder
        assert json.loads((source/'independent_review/completed.json').read_text())['status']=='complete'
        (dest/study).mkdir(exist_ok=False)
        names=[('design_protocol.json','design_protocol.json'),('independent_review/report.md','review.md'),('independent_review/completed.json','completed.json')]
        names += [('report/rolling.png','rolling.png'),('report/cumulative.png','cumulative.png')] if study=='de104' else [('independent_review/rolling_linear.png','rolling.png'),('independent_review/cumulative_linear.png','cumulative.png')]
        for src,relative in names:
            original=source/src;target=dest/study/relative;shutil.copyfile(original,target)
            assert sha(target)==sha(original)
            manifest.append(dict(path=str(target.relative_to(repo)).replace('\\','/'),source=folder+'/'+src,source_sha256=sha(original),sha256=sha(target)))
    (dest/'archive_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
if __name__=='__main__':main()
