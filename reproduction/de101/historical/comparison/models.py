"""Five-arm kernels. Full black-box targets; shared frozen mainline geometry."""
import json
from pathlib import Path
import numpy as np
import torch
from finshybrid.encoder import init_encoder, encode
from finshybrid.torch_model import Predictor
from finshybrid.geometry import rotation, euler, exp_so3, start_pose_average
from finshybrid.physics import nominal
from finshybrid.scales import PC, PS, VS, CENTER

NOMINAL = json.loads(Path(__file__).with_name('mainline_nominal.json').read_text())


class Comparison(Predictor):
    def __init__(self, meta, w, k, arm, delta_scale, device='cpu'):
        super().__init__(meta, w, k, 'A', device)
        self.arm_name = arm
        self.delta_scale = self.t(delta_scale)
        if arm == 'Kx':
            self.mask[:] = 1

    def step(self, z, u):
        raw = torch.cat((z[:,36:], u), -1) @ (self.k*self.mask).T
        velocity = raw[:,:6]
        if self.arm_name == 'H':
            velocity = velocity + self.fossen(z[:,1:13], u)
        p = z[:,1:7]*self.ps+self.pc
        if self.arm_name == 'Kx':
            d = raw[:,16:]*self.delta_scale
            pos = p[:,:3]+d[:,:3]
            rr = self.rotation(p) @ self.exp(d[:,3:])
        else:
            pf = p + .1*self.rates(p, .5*(z[:,7:13]+velocity)*self.vs)
            rr = self.rotation(pf)
            pos = pf[:,:3]-torch.einsum('nij,j->ni', rr-self.rotation(p), self.center)
        angles = torch.stack((torch.asin(torch.clamp(-rr[:,1,2],-1+1e-12,1-1e-12)),
                              torch.atan2(rr[:,0,2],rr[:,2,2]), torch.atan2(rr[:,1,0],rr[:,1,1])), -1)
        pose = (torch.cat((pos,angles),-1)-self.pc)/self.ps
        return torch.cat((z[:,:1],pose,velocity,torch.zeros_like(z[:,13:36]),raw[:,6:16]),-1)

    def objective(self, x, us, ys, latent_scale, anchor):
        # Exact mainline objective for H/Knu. Kx adds only its own head anchor.
        value = super().objective(x, us, ys, latent_scale, anchor)
        if self.arm_name == 'Kx':
            value = value + 1e-4*((self.k[16:]-anchor[0][16:])**2).mean()
        return value


def initial(data, seed, arm):
    meta, w = init_encoder(data, seed, 'G1')
    z = encode(data['x'], meta, w, 'G1')
    velocity = data['y'][:,7:13].copy()
    if arm == 'H':
        velocity -= nominal(data['x'], data['u'], NOMINAL)
    target = np.c_[velocity, encode(data['y'],meta,w,'G1')]
    p0 = data['x'][:,1:7]*PS+PC
    p1 = data['y'][:,1:7]*PS+PC
    # SciPy's rotation log is used only for offline training labels, never runtime.
    from scipy.spatial.transform import Rotation
    dr = Rotation.from_matrix(rotation(p0).transpose(0,2,1) @ rotation(p1)).as_rotvec()
    delta = np.c_[p1[:,:3]-p0[:,:3], dr]
    scale = np.maximum(delta.std(0), 1e-4)
    design = np.c_[z,data['u']]
    gram = design.T@design/len(z)+.003*np.eye(16)
    k = np.zeros((22,16))
    # Separate solve keeps common velocity/latent initialization exactly shared.
    k[:16] = np.linalg.solve(gram,design.T@target/len(z)).T
    if arm == 'Kx':
        k[16:] = np.linalg.solve(gram,design.T@(delta/scale)/len(z)).T
    return meta,w,k,np.maximum(z.std(0),.2),scale


def nstep(z,u,k,arm,delta_scale,bias=None):
    raw = np.c_[z[:,36:],u] @ k.T
    v = raw[:,:6].copy()
    if arm in ('H','F'):
        v += nominal(z,u,NOMINAL)
    if bias is not None:
        v += bias
    p=z[:,1:7]*PS+PC
    if arm == 'Kx':
        d=raw[:,16:]*delta_scale
        pos=p[:,:3]+d[:,:3]
        rr=rotation(p)@exp_so3(d[:,3:])
    else:
        pf=start_pose_average(p,z[:,7:13]*VS,v*VS)
        rr=rotation(pf)
        pos=pf[:,:3]-np.einsum('nij,j->ni',rr-rotation(p),CENTER)
    return np.c_[z[:,:1],(np.c_[pos,euler(rr)]-PC)/PS,v,np.zeros((len(z),23)),raw[:,6:16]]


def save_model(path, p, meta, scale, arm):
    raw={'arm':arm,'meta':{k:v.tolist() for k,v in meta.items()},
         'arrays':[v.tolist() for v in p.arrays()], 'delta_scale':scale.tolist()}
    path.write_text(json.dumps(raw,allow_nan=False),encoding='utf-8')


def load_model(path):
    raw=json.loads(Path(path).read_text())
    return raw,{k:np.array(v) for k,v in raw['meta'].items()},[np.array(v) for v in raw['arrays']]
