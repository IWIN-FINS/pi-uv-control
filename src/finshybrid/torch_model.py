"""Differentiable original G1/A kernels, isolated from experiment launchers.
The B branch remains only inside the inherited source; the public training CLI uses A.
"""
import json
from pathlib import Path
import numpy as np
import torch
from .scales import PC,PS,VS,US,DT,H,BASIC,CENTER
from .encoder import columns, dimension
def nominal():
    return json.loads((Path(__file__).parent/'assets/train281/operator.json').read_text())['nominal']
class BasePredictor(torch.nn.Module):
    def __init__(self, arm, encoder, weights, k, device='cpu'):
        super().__init__(); self.arm=arm; self.device=device
        self.t=lambda x:torch.as_tensor(x,dtype=torch.float64,device=device)
        self.pc,self.ps,self.vs,self.us=map(self.t,(PC,PS,VS,US))
        self.m,self.ld,self.qd,self.sp,self.bi=map(self.t,(nominal()[x] for x in ('mass','linearDrag','quadraticDrag','spring','bias')))
        self.meta={key:self.t(v) for key,v in encoder.items()}
        self.w=torch.nn.ParameterList([torch.nn.Parameter(self.t(v).clone()) for v in weights])
        self.k=torch.nn.Parameter(self.t(k).clone())
        mask=np.ones((51,58));mask[:,[1,3]]=0
        if arm.endswith('A'):mask[:6]=0
        self.mask=self.t(mask);self.inputs=torch.tensor(BASIC,device=device)
        bounds=np.full(52,100.);bounds[13:34]=10000.;self.bounds=self.t(bounds)
        self.chart_valid=torch.ones((),dtype=torch.bool,device=device)

    def encode(self,x):
        inp=(x.index_select(1,self.inputs)-self.meta['mean'])/self.meta['scale']
        q=torch.relu(torch.relu(torch.relu(inp@self.w[0]+self.w[1])@self.w[2]+self.w[3])@self.w[4]+self.w[5])
        a=(q-self.meta['latentMean'])/self.meta['latentScale']+.01*inp
        c=a-a.mean(-1,keepdim=True)
        return c/torch.sqrt((c*c).mean(-1,keepdim=True)+1e-5)*self.w[6]+self.w[7]

    def rotation(self,p):
        a,b,c=p[:,3],p[:,4],p[:,5];ca,sa,cb,sb,cc,sc=a.cos(),a.sin(),b.cos(),b.sin(),c.cos(),c.sin()
        return torch.stack((cb*cc+sb*sa*sc,-cb*sc+sb*sa*cc,sb*ca,ca*sc,ca*cc,-sa,-sb*cc+cb*sa*sc,sb*sc+cb*sa*cc,cb*ca),-1).reshape(-1,3,3)

    def rates(self,p,v):
        a,c=p[:,3],p[:,5];w=v[:,3:];sc,cc=c.sin(),c.cos()
        self.chart_valid=self.chart_valid & (a.cos().abs()>=.1).all()
        ang=torch.stack((cc*w[:,0]-sc*w[:,1],(sc*w[:,0]+cc*w[:,1])/a.cos(),a.tan()*(sc*w[:,0]+cc*w[:,1])+w[:,2]),-1)
        return torch.cat((torch.bmm(self.rotation(p),v[:,:3,None]).squeeze(-1),ang),-1)

    def rhs(self,s,u):
        p=s[:,:6]*self.ps+self.pc;v=s[:,6:]*self.vs;r=self.rotation(p)
        mom=v[:,:3]*self.m[:3];ang=v[:,3:]*self.m[3:]
        cor=torch.cat((torch.linalg.cross(v[:,3:],mom),torch.linalg.cross(v[:,:3],mom)+torch.linalg.cross(v[:,3:],ang)),-1)
        zero=torch.zeros_like(v[:,0])
        restoring=torch.cat((self.bi[1]*r[:,1,:],torch.stack((self.sp[3]*r[:,1,2],zero,-self.sp[5]*r[:,1,0]),-1)),-1)
        acc=(u*self.us-self.ld*v-self.qd*v*v.abs()+restoring-cor)/self.m
        return torch.cat((self.rates(p,v)/self.ps,acc/self.vs),-1)

    def fossen(self,s,u):
        for _ in range(4):
            k1=self.rhs(s,u);k2=self.rhs(s+.0125*k1,u);k3=self.rhs(s+.0125*k2,u);k4=self.rhs(s+.025*k3,u)
            s=s+.025*(k1+2*k2+2*k3+k4)/6
        return s[:,6:]

    def step(self,z,u):
        k=self.k*self.mask
        raw=z@k[:,:52].T+u@k[:,52:].T
        velocity=raw[:,6:12]
        if self.arm.startswith('C'):velocity=velocity+self.fossen(z[:,1:13],u)
        if self.arm.endswith('A'):
            p=z[:,1:7]*self.ps+self.pc
            pose=(p+DT*self.rates(p,.5*(z[:,7:13]+velocity)*self.vs)-self.pc)/self.ps
        else:pose=z[:,1:7]+raw[:,:6]
        return torch.cat((z[:,:1],pose,velocity,raw[:,12:]),-1)

    def rollout(self,x,us):
        z=torch.cat((x[:,:36],self.encode(x)),-1);states=[]
        self.chart_valid=torch.ones((),dtype=torch.bool,device=self.device)
        for u in us:
            z=self.step(z,u);states.append(z)
            self.chart_valid=self.chart_valid & torch.isfinite(z).all() & (z[:,1:].abs()<=self.bounds[1:]).all() & ((z[:,4]*self.ps[3]).abs()<1.3).all()
        return torch.stack(states)

    def objective(self,x,us,ys,latent_scale,anchor):
        zz=self.rollout(x,us)
        error=zz[:,:,1:13]-ys[:,:,1:13]
        angles=error[:,:,3:6]*self.ps[3:]
        error=torch.cat((error[:,:,:3],torch.atan2(angles.sin(),angles.cos())/self.ps[3:],error[:,:,6:]),-1)
        target=self.encode(ys.reshape(-1,42)).reshape(len(us),len(x),16)
        loss=(error*error).mean()+.01*(((zz[:,:,36:]-target)/latent_scale)**2).mean()
        loss=loss+1e-4*(((self.k-anchor[0])*self.mask)**2).sum()/(51*58)
        loss=loss+1e-5*sum(((w-a)**2).mean() for w,a in zip(self.w,anchor[1:]))
        return loss

    def arrays(self):
        return [self.k.detach().cpu().numpy().copy(),*[w.detach().cpu().numpy().copy() for w in self.w]]

class G1Predictor(BasePredictor):
    def __init__(self,meta,w,k,g,device='cpu'):
        super().__init__('C_A',meta,w,k,device)
        self.group=g;self.d=dimension(g);self.inputs=torch.tensor(columns(g),device=device);self.mask=torch.ones_like(self.k)
        self.bounds=self.t(np.full(36+self.d,100.))
    def encode(self,x):
        xi=(x.index_select(1,self.inputs)-self.meta['mean'])/self.meta['scale']
        q=torch.relu(torch.relu(torch.relu(xi@self.w[0]+self.w[1])@self.w[2]+self.w[3])@self.w[4]+self.w[5])
        a=(q-self.meta['latentMean'])/self.meta['latentScale']+.01*xi;c=a-a.mean(-1,keepdim=True)
        return c/torch.sqrt((c*c).mean(-1,keepdim=True)+1e-5)*self.w[6]+self.w[7]
    def step(self,z,u):
        raw=torch.cat((z[:,36:],u),-1)@self.k.T
        velocity=raw[:,:6]+self.fossen(z[:,1:13],u);p=z[:,1:7]*self.ps+self.pc
        pose=(p+DT*self.rates(p,.5*(z[:,7:13]+velocity)*self.vs)-self.pc)/self.ps
        return torch.cat((z[:,:1],pose,velocity,torch.zeros_like(z[:,13:36]),raw[:,6:]),-1)
    def rollout(self,x,us):
        z=torch.cat((x[:,:13],torch.zeros_like(x[:,13:36]),self.encode(x)),-1);states=[]
        self.chart_valid=torch.ones((),dtype=torch.bool,device=self.device)
        for u in us:
            z=self.step(z,u);states.append(z)
            self.chart_valid=self.chart_valid&torch.isfinite(z).all()&(z[:,1:].abs()<=self.bounds[1:]).all()&((z[:,4]*self.ps[3]).abs()<1.3).all()
        return torch.stack(states)
    def objective(self,x,us,ys,latent_scale,anchor):
        zz=self.rollout(x,us);e=zz[:,:,1:13]-ys[:,:,1:13];ang=e[:,:,3:6]*self.ps[3:]
        e=torch.cat((e[:,:,:3],torch.atan2(ang.sin(),ang.cos())/self.ps[3:],e[:,:,6:]),-1)
        target=self.encode(ys.reshape(-1,42)).reshape(len(us),len(x),self.d)
        loss=(e*e).mean()+.01*(((zz[:,:,36:]-target)/latent_scale)**2).mean()
        return loss+1e-4*((self.k-anchor[0])**2).mean()+1e-5*sum(((w-a)**2).mean() for w,a in zip(self.w,anchor[1:]))

class Predictor(G1Predictor):
    def __init__(self,meta,w,k,mode,device='cpu'):
        super().__init__(meta,w,k,'G1',device);self.mode=mode;self.mask=torch.ones_like(self.k)
        if mode=='A':self.mask[16:]=0
        self.center=self.t(CENTER)
    def exp(self,v):
        a2=(v*v).sum(-1);a=torch.sqrt(torch.clamp(a2,min=1e-12));s=torch.where(a2<1e-8,1-a2/6+a2*a2/120,a.sin()/a);c=torch.where(a2<1e-8,.5-a2/24+a2*a2/720,(1-a.cos())/(a*a));x,y,z=v.unbind(-1);o=torch.zeros_like(x)
        k=torch.stack((o,-z,y,z,o,-x,-y,x,o),-1).reshape(-1,3,3)
        return torch.eye(3,dtype=v.dtype,device=v.device)[None]+s[:,None,None]*k+c[:,None,None]*(k@k)
    def step(self,z,u):
        raw=torch.cat((z[:,36:],u),-1)@(self.k*self.mask).T
        nv=self.fossen(z[:,1:13],u);velocity=nv+raw[:,:6]
        p=z[:,1:7]*self.ps+self.pc;v1=velocity if self.mode=='A' else nv
        pf=p+DT*self.rates(p,.5*(z[:,7:13]+v1)*self.vs)
        rr=self.rotation(pf);pos=pf[:,:3]-torch.einsum('nij,j->ni',rr-self.rotation(p),self.center)
        if self.mode=='B':
            residual=raw[:,16:]*self.ps;pos=pos+residual[:,:3];rr=rr@self.exp(residual[:,3:])
        angles=torch.stack((torch.asin(torch.clamp(-rr[:,1,2],-1+1e-12,1-1e-12)),torch.atan2(rr[:,0,2],rr[:,2,2]),torch.atan2(rr[:,1,0],rr[:,1,1])),-1)
        pose=(torch.cat((pos,angles),-1)-self.pc)/self.ps
        return torch.cat((z[:,:1],pose,velocity,torch.zeros_like(z[:,13:36]),raw[:,6:16]),-1)
    def objective(self,x,us,ys,latent_scale,anchor):
        zz=self.rollout(x,us);e=zz[:,:,1:13]-ys[:,:,1:13];ang=e[:,:,3:6]*self.ps[3:]
        e=torch.cat((e[:,:,:3],torch.atan2(ang.sin(),ang.cos())/self.ps[3:],e[:,:,6:]),-1)
        target=self.encode(ys.reshape(-1,42)).reshape(len(us),len(x),10)
        penalty=1e-4*((self.k[:16]-anchor[0][:16])**2).mean()
        if self.mode=='B':penalty=penalty+1e-4*((self.k[16:]-anchor[0][16:])**2).mean()
        return (e*e).mean()+.01*(((zz[:,:,36:]-target)/latent_scale)**2).mean()+penalty+1e-5*sum(((w-a)**2).mean() for w,a in zip(self.w,anchor[1:]))

def fixture(data,ids,h=H):
    return (data['x'][ids],np.stack([data['u'][ids+j] for j in range(h)]),np.stack([data['y'][ids+j] for j in range(h)]))

def score(p,data,ids):
    errors=[];valid=True
    with torch.no_grad():
        for start in range(0,len(ids),256):
            batch=fixture(data,ids[start:start+256]);zz=p.rollout(p.t(batch[0]),p.t(batch[1]))
            valid=valid and bool(p.chart_valid.item())
            e=zz[-1,:,1:13]-p.t(batch[2][-1,:,1:13]);ang=e[:,3:6]*p.ps[3:]
            e=torch.cat((e[:,:3],torch.atan2(ang.sin(),ang.cos())/p.ps[3:],e[:,6:]),-1)
            errors.append(e.cpu().numpy())
    val=float(np.sqrt(np.mean(np.concatenate(errors)**2)))
    return val if valid and np.isfinite(val) else float('inf')

class GraphTrainer:
    def __init__(self,p,batch,latent_scale):
        self.p=p;self.batch=tuple(p.t(x).clone() for x in batch);self.scale=p.t(latent_scale)
        self.pars=[p.k,*p.w];self.anchor=[v.detach().clone() for v in self.pars]
        self.first=[torch.zeros_like(v) for v in self.pars];self.second=[torch.zeros_like(v) for v in self.pars]
        self.initial_snapshot=[v.detach().clone() for v in self.pars]
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self.gradient()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.value=self.gradient();self.valid=p.chart_valid
        with torch.cuda.stream(stream),torch.no_grad():
            for _ in range(3):p.objective(*self.batch,self.scale,self.anchor)
        torch.cuda.current_stream().wait_stream(stream)
        self.post=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.post,stream=stream),torch.no_grad():
            self.newvalue=p.objective(*self.batch,self.scale,self.anchor);self.newvalid=p.chart_valid

    def gradient(self):
        for v in self.pars:v.grad=None
        value=self.p.objective(*self.batch,self.scale,self.anchor);value.backward();return value

    def update(self,batch,t):
        for dst,src in zip(self.batch,batch):dst.copy_(self.p.t(src))
        self.graph.replay();loss=float(self.value.item())
        if not bool(self.valid.item()) or not np.isfinite(loss):return loss,False
        with torch.no_grad():
            gs=[v.grad for v in self.pars];norm=torch.sqrt(sum((g*g).sum() for g in gs));scale=torch.clamp(5/torch.clamp(norm,min=1e-12),max=1)
            old=[v.clone() for v in self.pars]
            ms=[.9*m+.1*g*scale for m,g in zip(self.first,gs)];vs=[.999*v+.001*(g*scale)**2 for v,g in zip(self.second,gs)]
            directions=[(m/(1-.9**t))/(torch.sqrt(v/(1-.999**t))+1e-8) for m,v in zip(ms,vs)]
            rate=.0003;accepted=False
            for _ in range(12):
                for v,o,d in zip(self.pars,old,directions):v.copy_(o-rate*d)
                self.post.replay();new=float(self.newvalue.item())
                if bool(self.newvalid.item()) and np.isfinite(new) and new<max(loss*2,loss+1):accepted=True;break
                rate*=.5
            if accepted:self.first,self.second=ms,vs
            else:
                for v,o in zip(self.pars,old):v.copy_(o)
        return loss,accepted
