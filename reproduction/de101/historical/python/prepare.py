from finshybrid.experiment_paths import config_path, load_record
"""Freeze a new failure-inclusive study; historical studies are read-only."""
from common import *
from native import export
import subprocess

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 assert read(PARENT/'completed.json')['stage']=='complete_no_qualified_cohort'
 # Audit existing artifact names before materializing the new seed's outputs.
 if not (ROOT/'seed_audit.json').exists():
  hits=[]
  for base in (str(config_path("unity_artifacts")),str(config_path("diagnostics"))):
   r=subprocess.run(['rg','--files',base,'-g','*1101221*'],capture_output=True,text=True)
   assert r.returncode in (0,1)
   hits.extend(x for x in r.stdout.splitlines() if 'de101-mainline-failure-inclusive' not in x)
  assert not hits,hits
  once(ROOT/'seed_audit.json',dict(seed=SEEDS[0],existing_artifact_name_collisions=hits,scope='Unity and diagnostics filenames; training split identities separately frozen'))
 protocol=dict(study='DE101',purpose='Failure-inclusive paired five-arm closed-loop comparison',
  authorization='User approved after planning; performance qualification waived, numerical correctness mandatory',
  arms=list(ARMS),initializations=list(INITS),conditions=list(CONDITIONS),seeds=list(SEEDS),attempts=75,duration_s=80,
  training=dict(transitions=3060,validation=1020,updates=1200,source=str(OFFLINE),reuse_all_three_initializations=True,H_bias_shares_H=True,F_not_trained=True,selection='Existing frozen models, no new fitting or checkpoint selection'),
  control=dict(dt=.1,horizon=8,sqp_budget=128,cost='legacy: seven unit weights, terminal weight3',window=40,forget=.98,deadline_hold=True,cpu_affinity=4095,serial=True),
  physics='Final mainline Fossen nominal with existing M+D mismatch; unchanged physical plant, no added external force or torque',
  environment='DE90 frozen flow generator; 20-50s active, 2s ramps; identical reference/initial/noise/flow per paired condition and seed',
  gate='Identical repaired DE100 native binary and models; independent numerical objective gradients, causal bias replay and physical/source audits. Poor RMSE is NOT an admission gate.',
  failures='Existing invalid observation domain terminates with exit20 and de100_terminal_failure.json. No fallback, no retries, no dropping failures. Unexpected exceptions or audit disagreement stop as engineering failures.',
  metrics=['completion and operability per all attempted episodes','20-50s horizontal RMSE','0-80s horizontal RMSE','depth and SO3','recovery','effort and saturation','invalid/hold/timeout','update and full-cycle timing'],
  statistics='One paired environment seed x three model initializations: descriptive only. Full-episode RMSE unavailable on early failure; complete-only statistics labelled with denominators. Pairwise differences only on mutually complete episodes, not an overall survivor ranking.',
  ordering='condition then init; arm order rotated by condition index + init-281',
  limits='75 attempts only; no old600/pilot375, no tuning or seed addition. Historical protocols and outputs unchanged.')
 once(ROOT/'design_protocol.json',protocol)
 for name in ('player','unity_source'):
  p=ROOT/name
  if not p.exists():subprocess.run(['cmd','/c','mklink','/J',str(p),str(PARENT/name)],check=True,capture_output=True)
  assert p.resolve()==(PARENT/name).resolve()
 prepare();export()
 for p in (ROOT/'models').glob('train*/*/operator.json'):
  old=read(PARENT/'n3060_u1200/models'/p.relative_to(ROOT/'models'));new=read(p)
  old.pop('experiment');new.pop('experiment');assert old==new,p
 for i in INITS:
  h=read(ROOT/f'models/train{i}/H/operator.json');b=read(ROOT/f'models/train{i}/H_bias/operator.json')
  h.pop('de99Arm');b.pop('de99Arm');assert h==b
 plan=jobs('formal');assert len(plan)==75 and len({j['id'] for j in plan})==75
 once(ROOT/'frozen_plan.json',plan)
 inputs={str(p):sha(p) for p in (ROOT/'models').rglob('*.json')}
 inputs.update({str(p):sha(p) for p in (ROOT/'flows').glob('*.csv')})
 for p in [ROOT/'design_protocol.json',ROOT/'model_mapping.json',ROOT/'frozen_plan.json',ROOT/'build_audit.json',PARENT/'completed.json',OFFLINE/'acceptance.json',REPO/'data/validation.npz']:
  inputs[str(p)]=sha(p)
 for rec in read(ROOT/'model_mapping.json').values():inputs[rec['path']]=rec['sha256']
 once(ROOT/'input_freeze.json',inputs)
 status('prepared_pending_numerical_acceptance',attempted=0,total=75)
 print('Prepared75; all model coefficients exactly equal DE100 n3060_u1200; H_bias shares H',flush=True)
if __name__=='__main__':main()
