from finshybrid.experiment_paths import config_path, load_record
"""DE99 five-arm final-mainline closed-loop protocol; isolated artifacts."""
import sys,json,hashlib,os,datetime
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=config_path("repository")
TOOLS=config_path("finssim_tools")
sys.path.insert(0,str(TOOLS));sys.path.insert(0,str(config_path("comparison_source")));sys.path.insert(0,str(REPO/'src'))
PARENT=config_path("de100")
ROOT=config_path("de101")
STUDY=ROOT
OFFLINE=config_path("offline")
SOURCE=config_path("de92")
ARMS=('F','Knu','Kx','H','H_bias');INITS=(281,282,283)
CONDITIONS=('still','steady','slow','fast','nonperiodic')
PRE=1100000;SEEDS=(1101221,)
CONFIG={a:dict(mode='bias' if a=='H_bias' else 'fixed',cost='legacy',window=40,forget=.98) for a in ARMS}
def read(p):return load_record(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def once(p,v):
 if Path(p).exists():assert read(p)==json.loads(json.dumps(v)),str(p)
 else:dump(p,v)
def status(stage,**kw):dump(ROOT/'execution/status.json',dict(stage=stage,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
def prepare():
 import shutil
 ROOT.mkdir(parents=True,exist_ok=True)
 once(ROOT/'design_protocol.json',read(STUDY/'design_protocol.json'))
 once(ROOT/'build_audit.json',read(PARENT/'build_audit.json'))
 from de90_flows import table
 for seed in (PRE,)+SEEDS:
  for condition in CONDITIONS:
   p=ROOT/'flows'/f'{seed}-{condition}.csv';p.parent.mkdir(exist_ok=True)
   v=table(seed,condition)
   if not p.exists():np.savetxt(p,v,delimiter=',',fmt='%.17g')
   assert np.max(abs(np.loadtxt(p,delimiter=',')-v))<1e-14
 once(ROOT/'flow_acceptance.json',{str(p):sha(p) for p in (ROOT/'flows').glob('*.csv')})

def jobs(stage):
 from de51_control import settings
 from run_de21_body_jobs import make_job
 pre=stage=='preflight';out=[]
 for ci,c in enumerate(('fast',) if pre else CONDITIONS):
  for si,s in enumerate((PRE,) if pre else SEEDS):
   for i in INITS:
    offset=(ci+si+i-281)%5
    for a in ARMS[offset:]+ARMS[:offset]:
     cfg=settings('combined',pre)
     cfg.update(FINSSIM_EDMD_MPC_MISMATCH_SCALE=0,FINSSIM_HYBRID_HYDRO_SCALE=0,FINSSIM_DE57_FAULT='',FINSSIM_DE62_COLLECTION=0,FINSSIM_DE69_COMPACT=1,
      FINSSIM_DE83_ONLINE='de92' if a=='H_bias' else 'none',FINSSIM_DE92_MODE=CONFIG[a]['mode'],FINSSIM_DE92_COST='legacy',FINSSIM_DE92_WINDOW=40,FINSSIM_DE92_FORGET=.98,
      FINSSIM_DE84_OBSERVATION='legacy',FINSSIM_DE84_DIRECTION_DEG=60,FINSSIM_DE84_PHASE_DEG=180,
      FINSSIM_DE90_FLOW_FILE=str(ROOT/'flows'/f'{s}-{c}.csv'),FINSSIM_DE82_PROFILE=c,FINSSIM_DE82_AMPLITUDE=0 if c=='still' else .1,
      FINSSIM_HYBRID_CURRENT_MODE='constant',FINSSIM_HYBRID_CURRENT_SIGMA_XYZ_MPS='0,0,0',FINSSIM_HYBRID_CURRENT_MAX_SPEED_MPS=1.,FINSSIM_BODY_EVAL_SECONDS=32 if pre else 80)
     out.append(make_job(ROOT/stage,ROOT/f'models/train{i}',f'{c}/train{i}/{a}/seed{s}',a,seeds=(s,),budget=128,preflight=pre,FINSSIM_MISMATCH_MODE='model_parameters',FINSSIM_DE55_ARM='C_A',FINSSIM_DE56_PRIOR='mismatched',**cfg))
 return out

