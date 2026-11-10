"""One serial handoff on this machine. Exit on an unknown error, preserving all evidence."""
import csv
import datetime
import msvcrt
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from paths import ROOT,LOADED,read,save,check,sha,verify

EXEC=ROOT/"execution"

def status(stage,**kwargs):
    save(EXEC/"status.json",dict(stage=stage,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 planned=200,**kwargs))

def predecessor_ready(root):
    s=read(root/"execution/status.json")
    if s.get("stage")!="complete":
        return False,s
    check(s.get("recorded")==400 and s.get("rounds")==100,"inconsistent predecessor terminal count")
    manifest=root/"execution/final_manifest.json"
    check(sha(manifest)==s["manifest_sha256"],"predecessor final manifest")
    return True,s

def active_players():
    import psutil
    return [p.pid for p in psutil.process_iter(["name"]) if (p.info["name"] or "").lower()=="finsrovhybridbasis.exe"]

def validate_predecessor():
    manifest=read(LOADED/"execution/final_manifest.json")
    episodes=[p for p in manifest if Path(p).parent.name=="episodes"]
    check(len(episodes)==400 and len({(Path(p).parents[2].name,read(p)["id"]) for p in episodes})==400,"400 unique archived attempts")
    verify(manifest)
    summary=read(LOADED/"report/summary.json")
    check(summary["rounds"]==100 and summary["recorded"]==400,"predecessor reports")

def outcome_for(job,dest,queue,code):
    term=dest/"de100_terminal_failure.json"
    if code==20 and term.exists() and not (dest/"completed.json").exists():
        r=read(term)
        check(r["site"]=="De83Observe" and r["reason"]=="Body-frame Euler chart near singularity",r)
        return "terminal_domain_failure"
    check(code==0 and (dest/"completed.json").is_file() and read(str(queue)+".completed.json")=={"jobs":1},
          f"unexpected Unity exit {code} for {job['id']}")
    return "complete"

def run_unity(index,j,work,dest):
    import psutil
    from adapter import serve
    out=work/"outcome.json"
    if out.exists():
        return read(out)["outcome"]
    check(not work.exists() and not dest.exists(), f"unresolved attempt, preserved {work}")
    work.mkdir(parents=True)
    player=ROOT/j["load"]/"player/FinsROVHybridBasis.exe"
    queue=work/"queue.json"
    settings=[dict(key=k,value=v) for k,v in j["env"].items()]
    save(queue,dict(jobs=[dict(id=j["id"],environment=settings)]))
    env={k:v for k,v in os.environ.items() if not k.startswith("FINSSIM_")}
    env.update(j["env"],FINSSIM_BODY_JOB_QUEUE=str(queue))
    logfile=work/"player.log"
    check(not active_players(),"another Unity player is active")
    with serve(env["FINSSIM_EDMD_MPC_MODEL_PATH"],work/"updates.jsonl") as side:
        start=time.monotonic()
        proc=subprocess.Popen([str(player),"-batchmode","-nographics","-logFile",str(logfile)],cwd=player.parent,
                              env={**env,**side},creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            target=psutil.Process(proc.pid)
            target.cpu_affinity(list(range(12)))
            check(set(target.cpu_affinity())==set(range(12)),"Unity affinity")
            save(work/"process.json",dict(pid=proc.pid,job=j["id"],mask=4095))
            while proc.poll() is None:
                status("running",recorded=index,load=j["load"],group=j["group"],current=j["id"],unity_pid=proc.pid)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if logfile.exists() and "Exception:" in logfile.read_text(encoding="utf-8",errors="replace"):
                        raise RuntimeError(f"Unity exception: {logfile}")
            save(work/"exit.json",dict(code=proc.returncode,wall_seconds=time.monotonic()-start))
            outcome=outcome_for(j,dest,queue,proc.returncode)
            save(out,dict(outcome=outcome,id=j["id"],load=j["load"],group=j["group"]))
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
    return outcome

def atomic_csv(name,data):
    p=ROOT/"report"/name
    p.parent.mkdir(exist_ok=True)
    temp=p.with_suffix(".tmp")
    with temp.open("w",newline="",encoding="utf-8-sig") as s:
        writer=csv.DictWriter(s,fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    os.replace(temp,p)

def report(all_records):
    closed=[];pred=[];effects=[]
    for r in all_records:
        key={k:r[k] for k in ("load","group","id")}
        score=r["tracking"]
        row=dict(**key,complete=score["completed"],eligible=score["eligible"],outcome=r["outcome"])
        for endpoint in ("path_rmse_m","depth_rmse_m","attitude_rmse_deg","loaded_path_rmse_m","loaded_depth_rmse_m","loaded_attitude_rmse_deg","timeout_fraction","cycle_p95_ms"):
            old=(score["old"] or {}).get(endpoint);new=(score["new"] or {}).get(endpoint)
            row[endpoint+"_old"]=old;row[endpoint+"_new"]=new
            row[endpoint+"_old_minus_new"]=old-new if score["eligible"] and None not in (old,new) else None
        closed.append(row)
        f=r["prediction"]
        for item in f["scores"]:
            fr=dict(**key,horizon=item["horizon"],phase=item["phase"],origins=item["origins"],valid=item["valid"],eligible=f["estimable"])
            for i,k in enumerate(f["metric_order"]):
                old=item["old"][i] if item["old"] is not None else None
                new=item["new"][i] if item["new"] is not None else None
                fr[k+"_old"]=old;fr[k+"_new"]=new
                fr[k+"_old_minus_new"]=old-new if f["estimable"] and None not in (new,old) else None
            pred.append(fr)
    pairs={(r["group"],r["id"],r["load"]):r for r in closed}
    for (group,jid,load),a in pairs.items():
        b=pairs.get((group,jid,"ballast300"))
        if load!="empty" or b is None:
            continue
        e=dict(group=group,id=jid)
        for k in a:
            if k.endswith("_old_minus_new"):
                e[k+"_ballast_minus_empty"]=b[k]-a[k] if a[k] is not None and b[k] is not None else None
        effects.append(e)
    atomic_csv("closed_loop_paired.csv",closed)
    if pred:
        atomic_csv("forecast_paired.csv",pred)
    if effects:
        atomic_csv("payload_interaction.csv",effects)
    save(ROOT/"report/summary.json",dict(recorded=len(closed),planned=200,matched_load_pairs=len(effects),
         completed=sum(r["complete"] for r in closed),eligible=sum(r["eligible"] for r in closed)))

def main():
    import psutil
    EXEC.mkdir(exist_ok=True)
    with (EXEC/"successor.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            sys.exit("Another successor owns this queue.")
        try:
            freeze=read(ROOT/"source_freeze.json")
            verify(freeze)
            evidence=read(ROOT/"acceptance.json")
            check(evidence["status"]=="passed","missing adapter acceptance")
            verify(evidence["inputs"])
            while True:
                ready,old=predecessor_ready(LOADED)
                if ready and not active_players():
                    break
                status("waiting_predecessor",recorded=0,predecessor_stage=old.get("stage"),
                       predecessor_recorded=old.get("recorded"),predecessor_pid=old.get("pid"))
                if not ready and (list((LOADED/"execution").glob("error_*.json")) or
                                  not psutil.pid_exists(old["pid"])):
                    raise RuntimeError("Predecessor stopped before confirmed completion")
                time.sleep(30)
            status("validating_predecessor",recorded=0)
            validate_predecessor()
            psutil.Process().cpu_affinity(list(range(12)))
            from scoring import predict,score_episode,folder
            jobs=read(ROOT/"plan200.json")
            check(len(jobs)==200,"plan length")
            if not (EXEC/"forecast_acceptance.json").exists():
                status("validating_forecasts",recorded=0)
                for j in jobs[:2]:
                    predict(j,parity=True)
                save(EXEC/"forecast_acceptance.json",dict(status="passed",loads=["empty","ballast300"]))
            all_records=[]
            for i,j in enumerate(jobs):
                saved=EXEC/f"episodes/episode_{i:03}.json"
                if saved.exists():
                    r=read(saved)
                    check(all(r[k]==j[k] for k in ("load","group","id")),"resume key")
                    verify(r["hashes"])
                else:
                    work=EXEC/f"queues/job{i:03}"
                    dest=folder(j["env"]["FINSSIM_EDMD_MPC_DIR"],j["id"].split("/")[-1])
                    outcome=run_unity(i,j,work,dest)
                    status("scoring",recorded=i,current=j["id"],load=j["load"])
                    tracking=score_episode(j,dest,outcome,work/"updates.jsonl")
                    prediction=predict(j)
                    r=dict(load=j["load"],group=j["group"],id=j["id"],outcome=outcome,tracking=tracking,prediction=prediction,
                           hashes={str(p):sha(p) for base in (dest,work) for p in base.rglob("*") if p.is_file()})
                    save(saved,r)
                all_records.append(r)
                report(all_records)
            verify(freeze)
            manifest={str(p):sha(p) for pattern in ("execution/episodes/*.json","report/*","acceptance.json","source_freeze.json") for p in ROOT.glob(pattern)}
            save(EXEC/"final_manifest.json",manifest)
            status("complete",recorded=200,manifest_sha256=sha(EXEC/"final_manifest.json"))
        except Exception:
            prior=read(EXEC/"status.json") if (EXEC/"status.json").exists() else {}
            status("engineering_failed",prior=prior,traceback=traceback.format_exc())
            raise

if __name__=="__main__":
    main()
