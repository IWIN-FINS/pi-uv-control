// Fossen MPC dynamics kernel. Same double-precision equations as
// finshybrid.physics and geometry; no fast-math or external runtime dependency.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {
void rotation(const double* p,double* r) {
    const double ca=std::cos(p[3]),sa=std::sin(p[3]),cb=std::cos(p[4]),sb=std::sin(p[4]),cc=std::cos(p[5]),sc=std::sin(p[5]);
    r[0]=cb*cc+sb*sa*sc;r[1]=-cb*sc+sb*sa*cc;r[2]=sb*ca;
    r[3]=ca*sc;r[4]=ca*cc;r[5]=-sa;
    r[6]=-sb*cc+cb*sa*sc;r[7]=sb*sc+cb*sa*cc;r[8]=cb*ca;
}
void cross(const double* a,const double* b,double* c) {
    c[0]=a[1]*b[2]-a[2]*b[1];c[1]=a[2]*b[0]-a[0]*b[2];c[2]=a[0]*b[1]-a[1]*b[0];
}
bool rates(const double* p,const double* v,const double* r,double* out) {
    double ca=std::cos(p[3]),sc=std::sin(p[5]),cc=std::cos(p[5]);
    if(std::fabs(ca)<.1)return false;
    for(int i=0;i<3;++i)out[i]=r[3*i]*v[0]+r[3*i+1]*v[1]+r[3*i+2]*v[2];
    out[3]=cc*v[3]-sc*v[4];out[4]=(sc*v[3]+cc*v[4])/ca;
    out[5]=std::tan(p[3])*(sc*v[3]+cc*v[4])+v[5];return true;
}
bool rhs(const double* x,const double* u,const double* q,double* out) {
    const double *v=x+6,*m=q,*ld=q+6,*qd=q+12,*sp=q+18,*bi=q+24;
    double r[9],mom[3],ang[3],a[3],b[3],c[3],rest[6];rotation(x,r);
    if(!rates(x,v,r,out))return false;
    for(int i=0;i<3;++i){mom[i]=v[i]*m[i];ang[i]=v[i+3]*m[i+3];}
    cross(v+3,mom,a);cross(v,mom,b);cross(v+3,ang,c);
    for(int i=0;i<3;++i)rest[i]=bi[1]*r[3+i];
    rest[3]=sp[3]*r[5];rest[4]=0;rest[5]=-sp[5]*r[3];
    for(int i=0;i<6;++i){double cor=i<3?a[i]:b[i-3]+c[i-3];out[i+6]=(u[i]-ld[i]*v[i]-qd[i]*v[i]*std::fabs(v[i])+rest[i]-cor)/m[i];}
    return true;
}
}

extern "C" int fossen_step(std::size_t count,const double* states,const double* wrenches,const double* params,double* result) {
    for(std::size_t row=0;row<count;++row) {
        const double *s=states+12*row,*u=wrenches+6*row;
        double x[12],tmp[12],k1[12],k2[12],k3[12],k4[12];
        std::copy(s,s+12,x);
        for(int sub=0;sub<4;++sub) {
            if(!rhs(x,u,params,k1))return 1;
            for(int i=0;i<12;++i)tmp[i]=x[i]+.0125*k1[i];
            if(!rhs(tmp,u,params,k2))return 1;
            for(int i=0;i<12;++i)tmp[i]=x[i]+.0125*k2[i];
            if(!rhs(tmp,u,params,k3))return 1;
            for(int i=0;i<12;++i)tmp[i]=x[i]+.025*k3[i];
            if(!rhs(tmp,u,params,k4))return 1;
            for(int i=0;i<12;++i)x[i]+=.025*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6;
        }
        double *out=result+12*row;
        double r0[9],r1[9],average[6],rate[6],pf[6];rotation(s,r0);
        for(int i=0;i<6;++i)average[i]=.5*(s[i+6]+x[i+6]);
        if(!rates(s,average,r0,rate))return 1;
        for(int i=0;i<6;++i)pf[i]=s[i]+.1*rate[i];
        rotation(pf,r1);
        // CENTER is a double-precision geometry constant in finshybrid.scales.
        const double centerY=-.08;
        for(int i=0;i<3;++i)out[i]=pf[i]-(r1[3*i+1]-r0[3*i+1])*centerY;
        out[3]=std::asin(std::max(-1.,std::min(1.,-r1[5])));
        out[4]=std::atan2(r1[2],r1[8]);out[5]=std::atan2(r1[3],r1[4]);
        for(int i=0;i<6;++i)out[i+6]=x[i+6];
        for(int i=0;i<12;++i)if(!std::isfinite(out[i]))return 2;
    }
    return 0;
}

// Whole recursive G1 horizon. Wrenches and the initial encoder output are
// computed once by the caller. Readout uses CURRENT latent, before A*z+B*u.
// All matrices are row-major; latent dimension is the frozen G1/A contract.
extern "C" int g1_rollout(std::size_t count,std::size_t horizon,
    const double* initial,const double* wrenches,const double* params,
    const double* initial_latent,const double* A,const double* B,
    const double* C,const double* D,const double* bias,
    const double* us,const double* vs,double* result) {
    for(std::size_t row=0;row<count;++row) {
        double state[12],latent[10];
        std::copy(initial,initial+12,state);
        std::copy(initial_latent,initial_latent+10,latent);
        for(std::size_t j=0;j<horizon;++j) {
            const double* wrench=wrenches+6*(row*horizon+j);
            double physical[12],u[6],velocity[6],next_latent[10];
            if(fossen_step(1,state,wrench,params,physical))return 1;
            if(std::fabs(physical[3])>=1. || std::fabs(physical[5])>=1.)return 2;
            for(int i=0;i<6;++i)u[i]=wrench[i]/us[i];
            for(int i=0;i<6;++i) {
                double cz=0.,du=0.;
                for(int k=0;k<10;++k)cz+=C[10*i+k]*latent[k];
                for(int k=0;k<6;++k)du+=D[6*i+k]*u[k];
                velocity[i]=physical[6+i]+((cz+du)+bias[i])*vs[i];
            }
            for(int i=0;i<10;++i) {
                double az=0.,bu=0.;
                for(int k=0;k<10;++k)az+=A[10*i+k]*latent[k];
                for(int k=0;k<6;++k)bu+=B[6*i+k]*u[k];
                next_latent[i]=az+bu;
            }
            double r0[9],r1[9],average[6],rate[6],pf[6];rotation(state,r0);
            for(int i=0;i<6;++i)average[i]=.5*(state[6+i]+velocity[i]);
            if(!rates(state,average,r0,rate))return 1;
            for(int i=0;i<6;++i)pf[i]=state[i]+.1*rate[i];
            rotation(pf,r1);
            for(int i=0;i<3;++i)state[i]=pf[i]-(r1[3*i+1]-r0[3*i+1])*(-.08);
            state[3]=std::asin(std::max(-1.,std::min(1.,-r1[5])));
            state[4]=std::atan2(r1[2],r1[8]);state[5]=std::atan2(r1[3],r1[4]);
            std::copy(velocity,velocity+6,state+6);
            std::copy(next_latent,next_latent+10,latent);
            for(double value:state)if(!std::isfinite(value))return 2;
            if(std::fabs(state[3])>=1. || std::fabs(state[5])>=1.)return 2;
            std::copy(state,state+12,result+12*(row*horizon+j));
        }
    }
    return 0;
}

// Identical projected Nesterov iterations on the caller's diagonally scaled
// QP. Eigenvalue scaling, tolerance and maximum iterations remain unchanged.
extern "C" int box_qp_scaled(std::size_t n,int iterations,double tolerance,
    double lipschitz,const double* h,const double* g,const double* lo,
    const double* hi,const double* initial,double* result,double* residual) {
    if(!n || iterations<1 || !(lipschitz>0.) || !(tolerance>0.))return -1;
    std::vector<double> x(n),y(n),next(n);
    auto clip=[&](double value,std::size_t i){return std::max(lo[i],std::min(hi[i],value));};
    auto projected_step=[&]() {
        for(std::size_t i=0;i<n;++i) {
            double value=0.;
            for(std::size_t k=0;k<n;++k)value+=h[i*n+k]*y[k];
            next[i]=clip(y[i]-(value+g[i])/lipschitz,i);
        }
    };
    for(std::size_t i=0;i<n;++i)x[i]=y[i]=clip(initial[i],i);
    double momentum=1.;int tick;
    for(tick=1;tick<=iterations;++tick) {
        projected_step();double restart=0.;
        for(std::size_t i=0;i<n;++i)restart+=(y[i]-next[i])*(next[i]-x[i]);
        if(restart>0.){momentum=1.;y=x;projected_step();}
        const double next_momentum=(1.+std::sqrt(1.+4.*momentum*momentum))/2.;
        for(std::size_t i=0;i<n;++i)y[i]=next[i]+(momentum-1.)/next_momentum*(next[i]-x[i]);
        x=next;momentum=next_momentum;*residual=0.;
        for(std::size_t i=0;i<n;++i) {
            double value=0.;
            for(std::size_t k=0;k<n;++k)value+=h[i*n+k]*x[k];
            *residual=std::max(*residual,std::fabs(x[i]-clip(x[i]-(value+g[i]),i)));
        }
        if(!std::isfinite(*residual))return -1;
        if(*residual<tolerance)break;
    }
    std::copy(x.begin(),x.end(),result);
    return std::min(tick,iterations);
}
