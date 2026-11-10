using System;

// DE51: one frozen calibrated velocity map shared by F, C_A and C_B.
// All normalized scale constants use the same Unity float->double convention.
public class FinsRovUnifiedFossenReference
{
    [Serializable] public sealed class Parameters
    {
        public double[] mass,linearDrag,quadraticDrag,spring,bias;
        public string velocityIntegrator;
        public int velocitySubsteps;
    }
    readonly Parameters q;
    public FinsRovUnifiedFossenReference(Parameters parameters)
    {
        q=parameters;
        if(q==null || q.velocityIntegrator!="rk4_coupled" || q.velocitySubsteps!=4)
            throw new InvalidOperationException("DE51 requires calibrated RK4/4 parameters");
        foreach(var v in new[]{q.mass,q.linearDrag,q.quadraticDrag,q.spring,q.bias})
            if(v==null || v.Length!=6)throw new InvalidOperationException("DE51 invalid calibrated parameters");
        foreach(double m in q.mass)if(m<=0)throw new InvalidOperationException("Nonpositive calibrated inertia");
    }
    static double[] Cross(double[] a,double[] b)=>new[]{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
    static double[,] Skew(double[] v)=>new double[,]{{0,-v[2],v[1]},{v[2],0,-v[0]},{-v[1],v[0],0}};
    void Rhs(double[] x,double[] u,out double[] f,out double[,] a,out double[,] b,bool jac)
    {
        var ps=FinsRovLiftedOperator.PoseScale;var vs=FinsRovLiftedOperator.VelocityScale;var us=FinsRovLiftedOperator.WrenchScale;
        var p=new double[6];for(int i=0;i<6;i++)p[i]=x[i]*ps[i]+FinsRovLiftedOperator.PoseCenter[i];
        var k=FinsRovBodyFrame.RateMatrix(p);var r=FinsRovBodyFrame.Rotation(p);
        var v=new double[3];var w=new double[3];var mv=new double[3];var mw=new double[3];
        for(int i=0;i<3;i++){v[i]=x[6+i]*vs[i];w[i]=x[9+i]*vs[3+i];mv[i]=q.mass[i]*v[i];mw[i]=q.mass[3+i]*w[i];}
        var cv=Cross(w,mv);var cw=Cross(v,mv);var spin=Cross(w,mw);double[,] dc=null;
        if(jac){
            dc=new double[6,6];var sv=Skew(v);var sw=Skew(w);var sm=Skew(mv);var sj=Skew(mw);
            for(int i=0;i<3;i++)for(int j=0;j<3;j++){
                dc[i,j]=sw[i,j]*q.mass[j];dc[i,j+3]=-sm[i,j];
                dc[i+3,j]=sv[i,j]*q.mass[j]-sm[i,j];dc[i+3,j+3]=sw[i,j]*q.mass[j+3]-sj[i,j];
            }
        }
        f=new double[12];a=jac?new double[12,12]:null;b=jac?new double[12,6]:null;
        for(int i=0;i<6;i++){
            for(int l=0;l<6;l++)f[i]+=k[i,l]*x[6+l]*vs[l]/ps[i];
            double vel=i<3?v[i]:w[i-3];double restoring=i<3?q.bias[1]*r[1,i]:i==3?q.spring[3]*r[1,2]:i==5?-q.spring[5]*r[1,0]:0;
            double coriolis=i<3?cv[i]:cw[i-3]+spin[i-3];
            f[6+i]=(u[i]*us[i]-q.linearDrag[i]*vel-q.quadraticDrag[i]*vel*Math.Abs(vel)+restoring-coriolis)/q.mass[i]/vs[i];
            if(jac){
                for(int j=0;j<6;j++)a[6+i,6+j]=(-dc[i,j]-(i==j?q.linearDrag[i]+2*q.quadraticDrag[i]*Math.Abs(vel):0))*vs[j]/q.mass[i]/vs[i];
                b[6+i,i]=us[i]/q.mass[i]/vs[i];
                for(int j=0;j<6;j++)a[i,6+j]=k[i,j]*vs[j]/ps[i];
            }
        }
        if(!jac)return;
        const double eps=1e-6;
        for(int axis=3;axis<6;axis++){
            double old=p[axis];p[axis]=old+eps;var plus=FinsRovBodyFrame.RateMatrix(p);p[axis]=old-eps;var minus=FinsRovBodyFrame.RateMatrix(p);p[axis]=old;
            for(int i=0;i<6;i++)for(int l=0;l<6;l++)a[i,axis]+=(plus[i,l]-minus[i,l])/(2*eps)*x[6+l]*vs[l]*ps[axis]/ps[i];
            p[axis]=old+eps;var rp=FinsRovBodyFrame.Rotation(p);p[axis]=old-eps;var rm=FinsRovBodyFrame.Rotation(p);p[axis]=old;
            for(int i=0;i<6;i++){
                double dg=i<3?q.bias[1]*(rp[1,i]-rm[1,i]):i==3?q.spring[3]*(rp[1,2]-rm[1,2]):i==5?-q.spring[5]*(rp[1,0]-rm[1,0]):0;
                a[6+i,axis]=dg/(2*eps)*ps[axis]/q.mass[i]/vs[i];
            }
        }
    }
    void Stage(double[] x,double[] u,double[,] sa,double[,] sb,out double[] f,out double[,] da,out double[,] db)
    {
        double[,] a,b;Rhs(x,u,out f,out a,out b,sa!=null);da=sa==null?null:new double[12,12];db=sa==null?null:new double[12,6];
        if(sa==null)return;
        for(int i=0;i<12;i++){
            for(int j=0;j<12;j++)for(int l=0;l<12;l++)da[i,j]+=a[i,l]*sa[l,j];
            for(int j=0;j<6;j++){db[i,j]=b[i,j];for(int l=0;l<12;l++)db[i,j]+=a[i,l]*sb[l,j];}
        }
    }
    static double[] Add(double[] x,double[] d,double h){var o=(double[])x.Clone();for(int i=0;i<o.Length;i++)o[i]+=h*d[i];return o;}
    static double[,] Add(double[,] x,double[,] d,double h){if(x==null)return null;var o=(double[,])x.Clone();for(int i=0;i<o.GetLength(0);i++)for(int j=0;j<o.GetLength(1);j++)o[i,j]+=h*d[i,j];return o;}
    public double[] Step(double[] z,double[] u,double[,] ja=null,double[,] jb=null)
    {
        const double dt=.1,h=.025;int n=z.Length;var x=new double[12];Array.Copy(z,1,x,0,12);
        var a=ja==null?null:new double[12,12];var b=ja==null?null:new double[12,6];if(a!=null)for(int i=0;i<12;i++)a[i,i]=1;
        for(int sub=0;sub<4;sub++){
            double[] k1,k2,k3,k4;double[,] a1,a2,a3,a4,b1,b2,b3,b4;
            Stage(x,u,a,b,out k1,out a1,out b1);
            Stage(Add(x,k1,h/2),u,Add(a,a1,h/2),Add(b,b1,h/2),out k2,out a2,out b2);
            Stage(Add(x,k2,h/2),u,Add(a,a2,h/2),Add(b,b2,h/2),out k3,out a3,out b3);
            Stage(Add(x,k3,h),u,Add(a,a3,h),Add(b,b3,h),out k4,out a4,out b4);
            for(int i=0;i<12;i++){
                x[i]+=h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6;
                if(a!=null){for(int j=0;j<12;j++)a[i,j]+=h*(a1[i,j]+2*a2[i,j]+2*a3[i,j]+a4[i,j])/6;for(int j=0;j<6;j++)b[i,j]+=h*(b1[i,j]+2*b2[i,j]+2*b3[i,j]+b4[i,j])/6;}
            }
        }
        var result=new double[n];result[0]=1;var p=FinsRovBodyFrame.Pose(z);var v0=new double[6];var v1=new double[6];
        var ps=FinsRovLiftedOperator.PoseScale;var vs=FinsRovLiftedOperator.VelocityScale;
        for(int i=0;i<6;i++){result[7+i]=x[6+i];v0[i]=z[7+i]*vs[i];v1[i]=x[6+i]*vs[i];}
        var pose=FinsRovBodyFrame.StartPoseAverage(p,v0,v1,dt);for(int i=0;i<6;i++)result[1+i]=(pose[i]-FinsRovLiftedOperator.PoseCenter[i])/ps[i];
        if(ja==null)return result;
        double[,] pj,vj0,vj1;FinsRovBodyFrame.StartPoseAverageJacobians(p,v0,v1,dt,out pj,out vj0,out vj1);
        for(int i=0;i<6;i++){
            for(int j=0;j<12;j++)ja[7+i,1+j]=a[6+i,j];for(int j=0;j<6;j++)jb[7+i,j]=b[6+i,j];
            for(int j=0;j<12;j++){
                double d=j<6?pj[i,j]*ps[j]:vj0[i,j-6]*vs[j-6];
                for(int l=0;l<6;l++)d+=vj1[i,l]*vs[l]*a[6+l,j];ja[1+i,1+j]=d/ps[i];
            }
            for(int j=0;j<6;j++){double d=0;for(int l=0;l<6;l++)d+=vj1[i,l]*vs[l]*b[6+l,j];jb[1+i,j]=d/ps[i];}
        }
        return result;
    }
}
