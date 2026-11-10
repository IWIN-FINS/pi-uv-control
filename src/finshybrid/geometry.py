"""Unity ZXY geometry, extracted from body_frame_math.py (see provenance)."""
import numpy as np
def rotation(p):
    p = np.asarray(p)
    a, b, c = p[..., 3], p[..., 4], p[..., 5]
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    return np.stack([cb*cc+sb*sa*sc, -cb*sc+sb*sa*cc, sb*ca,
                     ca*sc, ca*cc, -sa,
                     -sb*cc+cb*sa*sc, sb*sc+cb*sa*cc, cb*ca], axis=-1).reshape(p.shape[:-1]+(3,3))

def rate_matrix(p):
    p = np.asarray(p)
    a, c = p[..., 3], p[..., 5]
    if np.any(np.abs(np.cos(a)) < .1):
        raise ValueError('Unity ZXY Euler chart near singularity; use quaternion state for this domain')
    k = np.zeros(p.shape[:-1]+(6,6))
    k[..., :3, :3] = rotation(p)
    k[..., 3, 3], k[..., 3, 4] = np.cos(c), -np.sin(c)
    k[..., 4, 3], k[..., 4, 4] = np.sin(c)/np.cos(a), np.cos(c)/np.cos(a)
    k[..., 5, 3], k[..., 5, 4], k[..., 5, 5] = np.tan(a)*np.sin(c), np.tan(a)*np.cos(c), 1.
    return k

def rate_matrix_derivatives(p, difference_step=1e-6):
    """Return J(eta) and its pose-angle derivatives, using symmetric differences."""
    p = np.asarray(p, dtype=np.float64)
    base = rate_matrix(p)
    derivatives = np.zeros((len(p), 6, 6, 6), dtype=np.float64)
    for axis in (3, 4, 5):
        plus, minus = p.copy(), p.copy()
        plus[:, axis] += difference_step
        minus[:, axis] -= difference_step
        derivatives[:, axis] = (rate_matrix(plus) - rate_matrix(minus)) / (2 * difference_step)
    return base, derivatives

def start_pose_average(pose, velocity0, velocity1, dt=.1):
    """eta_next = eta_k + dt J(eta_k) (nu_k + nu_next) / 2."""
    return pose + dt * np.einsum('nij,nj->ni', rate_matrix(pose), .5*(velocity0+velocity1))
def skew(v):
    o=np.zeros(v.shape[:-1]+(3,3));o[...,0,1]=-v[...,2];o[...,0,2]=v[...,1];o[...,1,0]=v[...,2];o[...,1,2]=-v[...,0];o[...,2,0]=-v[...,1];o[...,2,1]=v[...,0];return o

def exp_so3(v):
    a2=np.sum(v*v,axis=-1);a=np.sqrt(np.maximum(a2,1e-12));s=np.where(a2<1e-8,1-a2/6+a2*a2/120,np.sin(a)/a);c=np.where(a2<1e-8,.5-a2/24+a2*a2/720,(1-np.cos(a))/a**2);k=skew(v)
    return np.eye(3)+s[...,None,None]*k+c[...,None,None]*(k@k)

def euler(r):return np.stack((np.arcsin(np.clip(-r[...,1,2],-1,1)),np.arctan2(r[...,0,2],r[...,2,2]),np.arctan2(r[...,1,0],r[...,1,1])),-1)
