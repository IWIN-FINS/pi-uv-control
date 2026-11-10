"""Prepare or run DE101's exact 75-job plan against an external frozen Unity Player.

Default is preparation only. No hardware transport. Requires Windows for execution.
"""
import argparse,ctypes,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from verify import verify,BUNDLE,read,sha
from audit_run import audit

def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8');temp.replace(p)
def materialize(root):
    out=[]
    for job in read(BUNDLE/'config/plan.json'):
        env={k:v.replace('${OUTPUT_ROOT}',str(root)).replace('${BUNDLE_ROOT}',str(BUNDLE)) for k,v in job['environment'].items()}
        out.append(dict(id=job['id'],environment=[dict(key=k,value=v) for k,v in env.items()]))
    return out
def runtime_check(player):
    assert player.name=='FinsROVHybridBasis.exe' and player.is_file()
    for name,h in read(BUNDLE/'config/runtime_manifest.json').items():
        assert sha(player.parent/name)==h,'Wrong/missing frozen runtime file: '+name
def affinity(process):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    if not kernel.SetProcessAffinityMask(int(process._handle),4095):raise RuntimeError('Cannot apply original CPU affinity4095')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--player',type=Path);p.add_argument('--run',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--job-id',help='Explicit diagnostic subset; not a full75 reproduction')
    a=p.parse_args();root=a.output_root.resolve()
    assert root!=BUNDLE and BUNDLE not in root.parents,'Outputs must be outside frozen reproduction bundle'
    verify(full_objectives=a.run)
    plan=materialize(root)
    if a.job_id:
        plan=[r for r in plan if r['id']==a.job_id];assert len(plan)==1,'Unknown job id'
    config=dict(plan=plan,bundle_manifest_sha256=sha(BUNDLE/'publication_manifest.json'),scope='diagnostic_subset' if a.job_id else 'full75')
    if root.exists():
        assert a.resume,'Existing output retained; use --resume only after review'
        assert read(root/'execution_plan.json')==config,'Changed plan or bundle'
    else:root.mkdir(parents=True);dump(root/'execution_plan.json',config)
    if not a.run:print(json.dumps(dict(stage='prepared_only',jobs=len(plan),output=str(root))));return
    assert os.name=='nt','Frozen Unity runtime and affinity policy require Windows'
    assert a.player is not None,'Pass --player /path/to/FinsROVHybridBasis.exe'
    player=a.player.resolve();runtime_check(player)
    import msvcrt
    lock=(root/'runner.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    active=subprocess.run(['powershell','-NoProfile','-Command','Get-Process FinsROVHybridBasis -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id'],capture_output=True,text=True)
    assert not active.stdout.strip(),'Another Unity process is running'
    results=[]
    for index,job in enumerate(plan):
        env={e['key']:e['value'] for e in job['environment']};folder=Path(env['FINSSIM_EDMD_MPC_DIR'])/'invariant-seed1101221'
        queue_dir=root/f'execution/job{index:02}';queue_dir.mkdir(parents=True,exist_ok=True)
        mark=queue_dir/'result.json';outcome=queue_dir/'outcome.json'
        if mark.exists():
            result=read(mark)
            for name,h in result['output_hashes'].items():assert sha(Path(name))==h
            results.append(result);continue
        if outcome.exists():result=read(outcome)
        else:
            assert not folder.exists(),'Partial output preserved; no automatic simulation retry'
            queue=queue_dir/'queue.json';dump(queue,dict(jobs=[job]))
            process_env={k:v for k,v in os.environ.items() if not k.startswith('FINSSIM_')};process_env.update(env)
            process_env.update(FINSSIM_BODY_JOB_QUEUE=str(queue),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            start=time.perf_counter()
            proc=subprocess.Popen([str(player),'-batchmode','-nographics','-logFile',str(queue_dir/'player.log')],cwd=player.parent,env=process_env,creationflags=subprocess.CREATE_NO_WINDOW)
            try:affinity(proc)
            except Exception:proc.terminate();proc.wait();raise
            dump(queue_dir/'process.json',dict(pid=proc.pid,id=job['id']))
            while proc.poll() is None:
                time.sleep(1)
                log=queue_dir/'player.log'
                if proc.poll() is None and log.exists() and 'Exception:' in log.read_text(encoding='utf-8',errors='replace'):
                    proc.terminate();proc.wait();raise RuntimeError('Unexpected native exception; preserved logs')
            completed=folder/'completed.json';terminal=folder/'de100_terminal_failure.json'
            if proc.returncode==20 and terminal.exists() and not completed.exists():
                term=read(terminal);assert term['site']=='De83Observe' and term['reason']=='Body-frame Euler chart near singularity';complete=False
            else:assert proc.returncode==0 and completed.exists() and Path(str(queue)+'.completed.json').exists();complete=True
            result=dict(id=job['id'],completed=complete,exit_code=proc.returncode,wall_seconds=time.perf_counter()-start)
            dump(outcome,result)
        result['audit']=audit(folder,Path(env['FINSSIM_EDMD_MPC_MODEL_PATH']),Path(env['FINSSIM_DE90_FLOW_FILE']),result['completed'])
        result['output_hashes']={str(f):sha(f) for f in folder.rglob('*') if f.is_file()}
        dump(mark,result);results.append(result)
        dump(root/'progress.json',dict(attempted=len(results),planned=len(plan),complete=sum(r['completed'] for r in results)))
    dump(root/'portable_summary.json',dict(stage='portable_checks_complete_historical_full_audit_not_repeated',scope=config['scope'],records=results))
if __name__=='__main__':main()
