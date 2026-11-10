"""Original Fossen/RK4 numerical kernel; imports isolated for portability."""
import sys
import numpy as np
from .scales import PC,PS,VS,US,DT
from .geometry import rotation,rate_matrix,rate_matrix_derivatives
er=sys.modules[__name__]
def nominal(z,u,q,jac=False):
    if q.get('velocityIntegrator')=='rk4_coupled':
        from .physics import velocity
        return velocity(z,u,q,int(q['velocitySubsteps']),jac)
    n=z.shape[1];p=z[:,1:7]*PS+PC;v=z[:,7:13]*VS;r=rotation(p)
    m,ld,qd,sp,bi=[np.asarray(q[k],float) for k in ('mass','linearDrag','quadraticDrag','spring','bias')]
    mom=v[:,:3]*m[:3];ang=v[:,3:]*m[3:]
    cor=np.c_[np.cross(v[:,3:],mom),np.cross(v[:,:3],mom)+np.cross(v[:,3:],ang)]
    restoring=np.zeros_like(v);restoring[:,:3]=bi[1]*r[:,1,:]
    restoring[:,3]=sp[3]*r[:,1,2];restoring[:,5]=-sp[5]*r[:,1,0]
    out=v+DT*(u*US-ld*v-qd*v*abs(v)+restoring-cor)/m
    if not jac:return out/VS
    ja=np.zeros((len(z),6,n));jb=np.broadcast_to(np.diag(DT*US/m/VS),(len(z),6,6)).copy()
    def skew(x):
        s=np.zeros((len(x),3,3));s[:,0,1]=-x[:,2];s[:,0,2]=x[:,1];s[:,1,0]=x[:,2];s[:,1,2]=-x[:,0];s[:,2,0]=-x[:,1];s[:,2,1]=x[:,0];return s
    dc=np.zeros((len(z),6,6));sv,sw,sm,sj=map(skew,(v[:,:3],v[:,3:],mom,ang))
    dc[:,:3,:3]=sw*m[None,None,:3];dc[:,:3,3:]=-sm
    dc[:,3:,:3]=sv*m[None,None,:3]-sm;dc[:,3:,3:]=sw*m[None,None,3:]-sj
    jv=-DT*dc/m[None,:,None]
    for i in range(6):jv[:,i,i]+=1-DT*(ld[i]+2*qd[i]*abs(v[:,i]))/m[i]
    ja[:,:,7:13]=jv*VS[None,None,:]/VS[None,:,None]
    for axis in (3,4,5):
        pp,pm=p.copy(),p.copy();pp[:,axis]+=1e-6;pm[:,axis]-=1e-6;dr=(rotation(pp)-rotation(pm))/2e-6
        dg=np.zeros_like(v);dg[:,:3]=bi[1]*dr[:,1,:];dg[:,3]=sp[3]*dr[:,1,2];dg[:,5]=-sp[5]*dr[:,1,0]
        ja[:,:,1+axis]=DT*dg/m/VS*PS[axis]
    return out/VS,ja,jb
def rhs(z,u,q,jac=False):
    q={key:value for key,value in q.items() if key not in ('velocityIntegrator','velocitySubsteps')}
    if jac:nv,na,nb=er.nominal(z,u,q,True);k,dk=rate_matrix_derivatives(z[:,1:7]*er.PS+er.PC)
    else:nv=er.nominal(z,u,q);k=rate_matrix(z[:,1:7]*er.PS+er.PC)
    v=z[:,7:13]*er.VS;out=np.zeros_like(z)
    out[:,1:7]=np.einsum('nij,nj->ni',k,v)/er.PS;out[:,7:13]=(nv-z[:,7:13])/er.DT
    if not jac:return out
    ja=np.zeros((len(z),z.shape[1],z.shape[1]));jb=np.zeros((len(z),z.shape[1],6))
    ja[:,7:13]=na/er.DT;ja[:,7:13,7:13]-=np.eye(6)/er.DT;jb[:,7:13]=nb/er.DT
    ja[:,1:7,7:13]=k*er.VS[None,None,:]/er.PS[None,:,None]
    ja[:,1:7,1:7]=np.einsum('nqij,nj->niq',dk,v)*er.PS[None,None,:]/er.PS[None,:,None]
    return out,ja,jb

def velocity(z,u,q,substeps=4,jac=False):
    # Nominal ODE has no dependence on residual memory or input-history slots.
    # Propagate only constant + twelve physical coordinates, then expand zeros.
    original_size=z.shape[1];out=z[:,:13].copy();h=er.DT/substeps;n=13
    if jac:a=np.broadcast_to(np.eye(n),(len(z),n,n)).copy();b=np.zeros((len(z),n,6))
    for _ in range(substeps):
        if not jac:
            k1=rhs(out,u,q);k2=rhs(out+h*k1/2,u,q);k3=rhs(out+h*k2/2,u,q);k4=rhs(out+h*k3,u,q)
        else:
            def derivative(s,sa,sb):
                f,fa,fb=rhs(s,u,q,True)
                return f,np.einsum('nij,njk->nik',fa,sa),np.einsum('nij,njk->nik',fa,sb)+fb
            k1,a1,b1=derivative(out,a,b);k2,a2,b2=derivative(out+h*k1/2,a+h*a1/2,b+h*b1/2)
            k3,a3,b3=derivative(out+h*k2/2,a+h*a2/2,b+h*b2/2);k4,a4,b4=derivative(out+h*k3,a+h*a3,b+h*b3)
            a+=h*(a1+2*a2+2*a3+a4)/6;b+=h*(b1+2*b2+2*b3+b4)/6
        out+=h*(k1+2*k2+2*k3+k4)/6
    if not jac:return out[:,7:13]
    full=np.zeros((len(z),6,original_size));full[:,:,:13]=a[:,7:13]
    return out[:,7:13],full,b[:,7:13]
