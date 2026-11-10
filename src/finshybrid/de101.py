"""Explicit loader for the historical DE101 family, separate from default DE83."""
from pathlib import Path
import numpy as np
from .predictor import Model
from .physics import nominal
from .geometry import rotation,euler,exp_so3,start_pose_average
from .scales import PS,PC,VS,CENTER


class DE101Model(Model):
    """Load a published DE101 operator.json (F/Knu/Kx/H/H_bias).

    path is explicit so installed packages never silently choose another family.
    F is the frozen physical-only baseline; Knu/Kx do not call Fossen.
    Only H_bias accepts nonzero online corrections.
    """
    def __init__(self,path):
        super().__init__(Path(path))
        self.arm=self.raw.get('de99Arm')
        if self.raw.get('experiment')!='DE101' or self.arm not in ('F','Knu','Kx','H','H_bias'):
            raise ValueError('Expected a DE101 five-method exported operator')
        self.delta_scale=np.asarray(self.raw['de99DeltaScale'],dtype=float)
        if self.delta_scale.shape!=(6,) or not np.isfinite(self.delta_scale).all():
            raise ValueError('Invalid Kx pose scale')
        self.delta_scale.setflags(write=False)

    def step(self,z,normalized_wrench,bias=None):
        z=np.atleast_2d(np.asarray(z,dtype=float));u=np.atleast_2d(np.asarray(normalized_wrench,dtype=float))
        bias=np.zeros(6) if bias is None else np.asarray(bias,dtype=float)
        if z.shape[1]!=46 or u.shape!=(len(z),6) or bias.shape!=(6,):
            raise ValueError('Expected (batch,46), (batch,6), bias(6)')
        if not all(np.isfinite(a).all() for a in (z,u,bias)):raise ValueError('Nonfinite predictor input')
        if self.arm!='H_bias' and np.any(bias):raise ValueError('Only H_bias permits online bias')
        out=z@self.a.T+u@self.b.T
        if self.arm in ('F','H','H_bias'):out[:,7:13]+=nominal(z,u,self.raw['nominal'])
        out[:,7:13]+=bias
        p=z[:,1:7]*PS+PC
        if self.arm=='Kx':
            delta=((z[:,36:]@self.a[1:7,36:].T)+(u@self.b[1:7].T))*self.delta_scale
            pos=p[:,:3]+delta[:,:3];rr=rotation(p)@exp_so3(delta[:,3:])
        else:
            pf=start_pose_average(p,z[:,7:13]*VS,out[:,7:13]*VS);rr=rotation(pf)
            pos=pf[:,:3]-np.einsum('nij,j->ni',rr-rotation(p),CENTER)
        out[:,1:7]=(np.c_[pos,euler(rr)]-PC)/PS
        if not np.isfinite(out).all():raise FloatingPointError('Nonfinite DE101 prediction')
        return out
