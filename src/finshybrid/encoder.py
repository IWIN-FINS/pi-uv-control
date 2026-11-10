"""G1 only: 10 -> 32 -> 16 -> 10, three ReLU, identity skip and LN."""
import numpy as np
from .scales import BASIC
def columns(g="G1"):
    if g != "G1": raise ValueError("Only the G1 mainline is supported")
    return BASIC
def dimension(g="G1"): return len(columns(g))
def init_encoder(data,seed,g):
    cols=columns(g);d=len(cols);raw=data['x'][:,cols]
    # Shared semantic base arrays, separately seeded per layer; shapes do not shift later random streams.
    r1=np.random.default_rng(np.random.SeedSequence([seed,1])).normal(size=(42,32))[cols]*np.sqrt(2/d)
    r2=np.random.default_rng(np.random.SeedSequence([seed,2])).normal(size=(32,16))*np.sqrt(2/32)
    r3=np.random.default_rng(np.random.SeedSequence([seed,3])).normal(size=(16,42))[:,cols]*np.sqrt(2/16)
    w=[r1,np.zeros(32),r2,np.zeros(16),r3,np.zeros(d),np.ones(d),np.zeros(d)]
    meta=dict(mean=raw.mean(0),scale=np.maximum(raw.std(0),.05))
    xi=(raw-meta['mean'])/meta['scale'];q=np.maximum(np.maximum(np.maximum(xi@w[0]+w[1],0)@w[2]+w[3],0)@w[4]+w[5],0)
    meta.update(latentMean=q.mean(0),latentScale=np.maximum(q.std(0),.2))
    return meta,w

def encode(x,meta,w,g):
    xi=(x[:,columns(g)]-meta['mean'])/meta['scale']
    q=np.maximum(np.maximum(np.maximum(xi@w[0]+w[1],0)@w[2]+w[3],0)@w[4]+w[5],0)
    a=(q-meta['latentMean'])/meta['latentScale']+.01*xi;c=a-a.mean(-1,keepdims=True)
    return c/np.sqrt((c*c).mean(-1,keepdims=True)+1e-5)*w[6]+w[7]

def materialize(k,g):
    d=dimension(g);n=36+d;a=np.zeros((n,n));b=np.zeros((n,6));a[0,0]=1;a[1:7,1:7]=np.eye(6)
    a[7:13,36:]=k[:6,:d];b[7:13]=k[:6,d:];a[36:,36:]=k[6:,:d];b[36:]=k[6:,d:]
    return a,b

def unpack(raw):
    g=raw['factorialArm'];d=dimension(g)
    meta={k:np.array(raw[k]) for k in ('mean','scale','latentMean','latentScale')}
    w=[np.array(raw[k]).reshape(s).T if s else np.array(raw[k]) for k,s in [('w1',(32,d)),('b1',None),('w2',(16,32)),('b2',None),('w3',(d,16)),('b3',None),('layerNormGamma',None),('layerNormBeta',None)]]
    a=np.array(raw['a']).reshape(36+d,36+d);b=np.array(raw['b']).reshape(36+d,6)
    k=np.c_[np.r_[a[7:13,36:],a[36:,36:]],np.r_[b[7:13],b[36:]]]
    return meta,w,k,a,b
