from common import *
import shutil,subprocess
from models import nstep,load_model
from finshybrid.encoder import encode,materialize
from finshybrid.train import load
from finshybrid.scales import US
def replace(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
def export():
 data=load(REPO/'data/validation.npz');template=read(REPO/'src/finshybrid/assets/train281/operator.json')
 for i in INITS:
  for arm in ARMS:
   source=OFFLINE/f'models/train{i}/{arm if arm in ("Knu","Kx") else "H"}/model.json'
   model,meta,arrays=load_model(source);k=arrays[0].copy()
   if arm=='F':k[:]=0
   aa,bb=materialize(k[:16],'G1')
   if arm=='Kx':aa[1:7,36:]=k[16:,:10];bb[1:7]=k[16:,10:]
   raw=dict(template);raw.update(experiment='DE101',de99Arm=arm,de99DeltaScale=model['delta_scale'],a=aa.ravel().tolist(),b=bb.ravel().tolist(),**{key:value.tolist() for key,value in meta.items()})
   for key,v in zip(('w1','b1','w2','b2','w3','b3','layerNormGamma','layerNormBeta'),arrays[1:]):raw[key]=v.T.ravel().tolist() if v.ndim==2 else v.tolist()
   dest=ROOT/f'models/train{i}/{arm}';once(dest/'operator.json',raw);cases=[];steps=[]
   from de81_common import base
   for idx in (0,100,200,400,800):
    p=data['pose'][idx].astype(np.float32).astype(float);v=data['velocity'][idx].astype(np.float32).astype(float);prev=data['previous'][idx].astype(np.float32).astype(float)
    x=base.physical_features(p[None],v[None],prev[None]/US);z=np.c_[x[:,:13],np.zeros((1,23)),encode(x,meta,arrays[1:],'G1')];u=data['u'][idx:idx+1]
    kind=arm if arm in ('F','Knu','Kx') else 'H'
    cases.append(dict(p=p.tolist(),v=v.tolist(),previousInput=prev.tolist(),z=z[0].tolist()))
    steps.append(dict(z=z[0].tolist(),u=u[0].tolist(),next=nstep(z,u,k,kind,np.array(model['delta_scale']))[0].tolist()))
   once(dest/'encoder_cases.json',dict(cases=cases));once(dest/'step_cases.json',dict(cases=steps))
 once(ROOT/'model_mapping.json',{f'{i}/{a}':dict(path=str(OFFLINE/f'models/train{i}/{a if a in ("Knu","Kx") else "H"}/model.json'),sha256=sha(OFFLINE/f'models/train{i}/{a if a in ("Knu","Kx") else "H"}/model.json')) for i in INITS for a in ARMS})

def numpy_step(z,u,a,b,mode,bias=None,pose_bias=None):
 # Model identity comes from all exported coefficients; no simulation truth used.
 key=hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest()
 if not hasattr(numpy_step,'lookup'):
  lookup={}
  for p in (ROOT/'models').glob('train*/*/operator.json'):
   raw=read(p);aa=np.array(raw['a']).reshape(46,46);bb=np.array(raw['b']).reshape(46,6);arm=raw['de99Arm']
   k=np.c_[np.r_[aa[7:13,36:],aa[36:,36:],aa[1:7,36:]],np.r_[bb[7:13],bb[36:],bb[1:7]]]
   lookup[hashlib.sha256(aa.tobytes()+bb.tobytes()).hexdigest()]=(k,arm if arm in ('F','Knu','Kx') else 'H',np.array(raw['de99DeltaScale']))
  numpy_step.lookup=lookup
 assert pose_bias is None or not np.any(pose_bias)
 k,arm,scale=numpy_step.lookup[key];return nstep(z,u,k,arm,scale,bias)

