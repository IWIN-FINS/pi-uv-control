import csv
import json
from pathlib import Path
from paths import ROOT,EMPTY,LOADED,read,check,sha
from adapter import Adapter,rows,packet,physical,np
from finshybrid.geometry import rotation,start_pose_average,euler
from finshybrid.physics import nominal
from finshybrid.scales import PS,PC,VS,US,CENTER
from finshybrid.bias import BiasUpdater

def table(p):
    with p.open(encoding="utf-8-sig",newline="") as s:
        return [r for r in csv.DictReader(s) if r.get("phase","KoopmanMpc")=="KoopmanMpc"]

def cols(data,keys):
    return np.array([[float(r[k]) for k in keys] for r in data])

def six(data,key):
    return cols(data,[f"{key}_{i}" for i in range(6)])

def folder(parent,seed):
    return Path(parent)/("invariant-"+seed)

def errors(guess,truth):
    delta=guess-truth
    angle=np.arccos(np.clip((np.sum(rotation(guess[:,:6])*rotation(truth[:,:6]),axis=(1,2))-1)/2,-1,1))
    return np.c_[np.sum(delta[:,[0,2]]**2,axis=1),delta[:,1]**2,angle**2,
                 np.sum(delta[:,6:9]**2,axis=1),np.sum(delta[:,9:]**2,axis=1)]

def advance(model,z,u,b):
    # DE101 H step with one frozen bias for each forecast origin.
    out=z@model.a.T+u@model.b.T
    out[:,7:13]+=nominal(z,u,model.raw["nominal"])+b
    pose=z[:,1:7]*PS+PC
    pf=start_pose_average(pose,z[:,7:13]*VS,out[:,7:13]*VS)
    rr=rotation(pf)
    pos=pf[:,:3]-np.einsum("nij,j->ni",rr-rotation(pose),CENTER)
    out[:,1:7]=(np.c_[pos,euler(rr)]-PC)/PS
    return out

def predict(j, *, parity=False):
    cond,init,_,seed=j["id"].split("/")
    root=EMPTY if j["load"]=="empty" else LOADED
    g=root/j["group"]
    fp=folder(g/"formal"/cond/init/"F"/seed,seed)
    archive_path=g/"predictions"/f"{cond}-{seed}.json"
    archive=read(archive_path)
    if not archive["full"]:
        return {"estimable":False,"reason":"incomplete common F record","scores":[]}
    check(sha(archive_path.with_suffix(".npz"))==archive["sha256"],"F forecast archive")
    stream=list(rows(fp/"online_adaptation.jsonl"))
    check(len(stream)==799 and Path(archive["source"]).resolve()==fp.resolve(),"common F history")
    updater=Adapter(j["env"]["FINSSIM_EDMD_MPC_MODEL_PATH"])
    model=updater.model
    states=np.array([physical(np.array(r[1:13])) for r in [stream[0]["previous"],*[r["observed"] for r in stream]]])
    u=np.array([r["input"] for r in stream]);z=model.lift(states)
    biases=np.zeros((len(states),6))
    if parity:
        old=BiasUpdater()
        base=advance(model,z[:-1],u,np.zeros_like(biases[:-1]))
        for i,err in enumerate(z[1:,7:13]-base[:,7:13]):
            biases[i+1]=old.update(err)["b"]
    else:
        for i,r in enumerate(stream):
            _,biases[i+1]=updater.step(packet(r))
    ix=np.arange(len(z)-8)
    pred=z[ix].copy(); alive=np.ones(len(ix),bool)
    rollout=[]
    for k in range(8):
        use=np.flatnonzero(alive)
        if len(use):
            with np.errstate(all="ignore"):
                pred[use]=advance(model,pred[use],u[ix[use]+k],biases[ix[use]])
            alive[use]=np.isfinite(pred[use]).all(1)&(abs(pred[use,1:]).max(1)<=100)&(abs(pred[use,4]*PS[3]+PC[3])<=1.3)
        pred[~alive]=np.nan
        rollout.append((model.physical(pred).copy(),alive.copy()))
    motor=table(fp/"motor_actions.csv")[::5]
    check(len(motor)==len(states) and max(abs(float(r["phase_time_s"])-.1*i) for i,r in enumerate(motor))<.003,"forecast truth timing")
    p=six(motor,"pose"); w=six(motor,"velocity"); rt=rotation(p).transpose(0,2,1)
    truth=np.c_[p,np.einsum("nij,nj->ni",rt,w[:,:3]),np.einsum("nij,nj->ni",rt,w[:,3:])]
    scores=[]
    for item in archive["summary"]:
        if item["method"]!="H_bias":
            continue
        h=item["horizon"];guess,valid=rollout[h-1]
        sel=np.ones(len(ix),bool) if item["phase"]=="all" else (ix>=200)&(ix<=491)
        keep=sel&valid
        full=errors(guess[keep],truth[ix[keep]+h]).mean(0)**.5 if keep.any() and keep.sum()==sel.sum() else None
        baseline=item["full_window_rmse"]
        if parity:
            check(int(keep.sum())==item["valid"] and ((full is None)==(baseline is None)),"archived validity")
            if full is not None:
                check(np.max(abs(full-baseline))<2e-8,"archived RMSE")
        scores.append(dict(horizon=h,phase=item["phase"],origins=int(sel.sum()),valid=int(keep.sum()),
                           old=baseline,new=full.tolist() if full is not None else None))
    ballast_ok=True
    if j["load"]=="ballast300":
        ballast_ok=read(fp/"ballast_physics.json")["maxDriftM"]<=.002
    return dict(estimable=ballast_ok,common_source=str(fp),metric_order=archive["metric_order"],scores=scores)

def score_episode(j,dest,outcome,worker_log):
    model_path=j["env"]["FINSSIM_EDMD_MPC_MODEL_PATH"]
    c,i,_,seed=j["id"].split("/")
    ref=folder(j["old_folder"],seed)
    archroot=EMPTY if j["load"]=="empty" else LOADED
    baseline=read(archroot/j["group"]/f"execution/episodes/episode_{j['old_index']:02}.json")
    check(baseline["id"]==j["id"] and Path(baseline["folder"])==ref,"baseline ID")
    result=dict(baseline_completed=baseline["completed"],old=baseline["metrics"],new=None,completed=outcome=="complete",
                eligible=False,old_ballast_valid=baseline.get("ballast_valid",True))
    if outcome!="complete":
        result["terminal"]=read(dest/"de100_terminal_failure.json")
        return result
    check(sha(model_path)==(dest/"frozen_model_sha256.txt").read_text().strip(),"model mismatch")
    check(read(dest/"completed.json")["compensated_samples"]==800,"control sample count")
    check(read(dest/"rate_contract.json")==read(ref/"rate_contract.json"),"rate contract")
    check(read(dest/"physical_profile_fixed.json")==read(ref/"physical_profile_fixed.json"),"hull physics")
    actual=read(dest/"model_parameters_learned_end.json")
    expected=read(ref/"model_parameters_learned_end.json")
    check(actual["predictor"]==expected["predictor"] and actual["truthBefore"]==actual["truthAfter"]==actual["truthEnd"]==expected["truthBefore"],"physics changed")
    motor=table(dest/"motor_actions.csv"); prior=table(ref/"motor_actions.csv")
    check(len(motor)==4000,"physics sample count")
    pose=six(motor,"pose"); reference=six(motor,"reference_pose")
    check(np.max(abs(reference[:len(prior)]-six(prior,"reference_pose")))<1e-5 and
          np.max(abs(pose[0]-six(prior[:1],"pose")[0]))<1e-7 and
          np.max(abs(six(motor[:1],"velocity")-six(prior[:1],"velocity")))<1e-7,"trajectory / initial state")
    for name,keys in [("current_velocity_trace.csv",[f"flow_{axis}_world_mps" for axis in "xyz"]),
                      ("observation_noise_audit.csv",["index","standard_gaussian_draw"])]:
        a,b=(cols(table(d/name),keys) for d in (dest,ref))
        check(len(b)>0 and np.array_equal(a[:len(b)],b),name)
    native_flow=cols(table(dest/"current_physics.csv"),[f"flow_{ax}" for ax in "xyz"])
    check(np.max(abs(native_flow-np.loadtxt(j["env"]["FINSSIM_DE90_FLOW_FILE"],delimiter=",")))<3e-6,"backend current")
    records=list(rows(dest/"online_adaptation.jsonl"));wire=list(rows(worker_log))
    check(len(records)==len(wire)==799,"online sample count")
    history=six(table(dest/"rolling_history.csv")[1:],"history")/US
    check(np.max(abs(history-np.array([r["input"] for r in records])))<2e-6,"applied inputs")
    replay=Adapter(model_path)
    for r,w in zip(records,wire):
        info,b=replay.step(packet(r))
        check(info["reason"]==w["reason"] and info["accepted"]==r["acceptedV"] and
              np.max(abs(b-r["velocityBias"]))<2e-8 and np.max(abs(b-w["bias"]))<2e-8 and
              not any(r["poseBias"]),"causal bias replay")
    t=cols(motor,["phase_time_s"])[:,0]
    e=errors(np.c_[pose,np.zeros((len(t),6))],np.c_[reference,np.zeros((len(t),6))])
    metrics={}
    for prefix,sel in [("",np.ones(len(t),bool)),("loaded_",(t>=20)&(t<50))]:
        v=np.sqrt(e[sel,:3].mean(0))
        for k,x in zip(("path_rmse_m","depth_rmse_m","attitude_rmse_deg"),(v[0],v[1],np.rad2deg(v[2]))):
            metrics[prefix+k]=float(x)
    timing=table(dest/"full_cycle_timing.csv")
    check(len(timing)==800,"cycle timing coverage")
    metrics.update(timeout_fraction=sum(r["full_overrun"]=="true" for r in timing)/len(timing),
                   cycle_p95_ms=float(np.percentile(cols(timing,["cycle_ms"]),95)),
                   accepted_updates=replay.update.accepted)
    valid=True
    if j["load"]=="ballast300":
        ballast=read(dest/"ballast_physics.json")
        check(abs(ballast["kg"]-.3)<1e-6 and ballast["samples"]==4000 and
              max(abs(ballast["mount"][k]-v) for k,v in zip("xyz",[-.19,.015,.13]))<1e-6,"ballast mount")
        valid=0<=ballast["maxDriftM"]<=.002
        result["max_drift_mm"]=ballast["maxDriftM"]*1000
    else:
        check(not (dest/"ballast_physics.json").exists(),"unexpected ballast")
    result.update(new=metrics,eligible=bool(valid and result["old_ballast_valid"] and baseline["completed"]))
    return result
