using System;

// Equivalent RK4 velocity subsystem: position and yaw do not feed roll/pitch or
// body acceleration. The inherited 12-state map remains an independent reference.
public sealed class FinsRovUnifiedFossen : FinsRovUnifiedFossenReference
{
    readonly Parameters q;
    public FinsRovUnifiedFossen(Parameters parameters):base(parameters){q=parameters;}
    static double[] Cross(double[] a,double[] b)=>new[]{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
    static double[,] Skew(double[] v)=>new double[,]{{0,-v[2],v[1]},{v[2],0,-v[0]},{-v[1],v[0],0}};
    void Rhs(double[] x,double[] u,out double[] f,out double[,] a,out double[,] b,bool jac)
    {
        var vs=FinsRovLiftedOperator.VelocityScale;var us=FinsRovLiftedOperator.WrenchScale;
        double ca=Math.Cos(x[0]),sa=Math.Sin(x[0]),cc=Math.Cos(x[1]),sc=Math.Sin(x[1]);
        if(Math.Abs(ca)<.1)throw new InvalidOperationException("Body-frame Euler chart near singularity");
        double ta=sa/ca;var v=new double[3];var w=new double[3];var mv=new double[3];var mw=new double[3];
        for(int i=0;i<3;i++){v[i]=x[2+i]*vs[i];w[i]=x[5+i]*vs[3+i];mv[i]=q.mass[i]*v[i];mw[i]=q.mass[3+i]*w[i];}
        var cv=Cross(w,mv);var cw=Cross(v,mv);var spin=Cross(w,mw);double[,] dc=null;
        if(jac){
            dc=new double[6,6];var sv=Skew(v);var sw=Skew(w);var sm=Skew(mv);var sj=Skew(mw);
            for(int i=0;i<3;i++)for(int j=0;j<3;j++){
                dc[i,j]=sw[i,j]*q.mass[j];dc[i,j+3]=-sm[i,j];
                dc[i+3,j]=sv[i,j]*q.mass[j]-sm[i,j];dc[i+3,j+3]=sw[i,j]*q.mass[j+3]-sj[i,j];
            }
        }
        var restoring=new[]{q.bias[1]*ca*sc,q.bias[1]*ca*cc,-q.bias[1]*sa,-q.spring[3]*sa,0,-q.spring[5]*ca*sc};
        f=new double[8];a=jac?new double[8,8]:null;b=jac?new double[8,6]:null;
        f[0]=cc*w[0]-sc*w[1];f[1]=ta*(sc*w[0]+cc*w[1])+w[2];
        for(int i=0;i<6;i++){
            double vel=i<3?v[i]:w[i-3],coriolis=i<3?cv[i]:cw[i-3]+spin[i-3];
            f[2+i]=(u[i]*us[i]-q.linearDrag[i]*vel-q.quadraticDrag[i]*vel*Math.Abs(vel)+restoring[i]-coriolis)/q.mass[i]/vs[i];
            if(jac){
                for(int j=0;j<6;j++)a[2+i,2+j]=(-dc[i,j]-(i==j?q.linearDrag[i]+2*q.quadraticDrag[i]*Math.Abs(vel):0))*vs[j]/q.mass[i]/vs[i];
                b[2+i,i]=us[i]/q.mass[i]/vs[i];
            }
        }
        if(!jac)return;
        a[0,1]=-sc*w[0]-cc*w[1];a[0,5]=cc*vs[3];a[0,6]=-sc*vs[4];
        a[1,0]=(sc*w[0]+cc*w[1])/(ca*ca);a[1,1]=ta*(cc*w[0]-sc*w[1]);
        a[1,5]=ta*sc*vs[3];a[1,6]=ta*cc*vs[4];a[1,7]=vs[5];
        var ga=new[]{-q.bias[1]*sa*sc,-q.bias[1]*sa*cc,-q.bias[1]*ca,-q.spring[3]*ca,0,q.spring[5]*sa*sc};
        var gc=new[]{q.bias[1]*ca*cc,-q.bias[1]*ca*sc,0,0,0,-q.spring[5]*ca*cc};
        for(int i=0;i<6;i++){a[2+i,0]=ga[i]/q.mass[i]/vs[i];a[2+i,1]=gc[i]/q.mass[i]/vs[i];}
    }
    void Stage(double[] x,double[] u,double[,] sa,double[,] sb,out double[] f,out double[,] da,out double[,] db)
    {
        double[,] a,b;Rhs(x,u,out f,out a,out b,sa!=null);da=sa==null?null:new double[8,8];db=sa==null?null:new double[8,6];
        if(sa==null)return;
        for(int i=0;i<8;i++){
            for(int j=0;j<8;j++)for(int l=0;l<8;l++)da[i,j]+=a[i,l]*sa[l,j];
            for(int j=0;j<6;j++){db[i,j]=b[i,j];for(int l=0;l<8;l++)db[i,j]+=a[i,l]*sb[l,j];}
        }
    }
    static double[] Add(double[] x,double[] d,double h){var o=(double[])x.Clone();for(int i=0;i<o.Length;i++)o[i]+=h*d[i];return o;}
    static double[,] Add(double[,] x,double[,] d,double h){if(x==null)return null;var o=(double[,])x.Clone();for(int i=0;i<o.GetLength(0);i++)for(int j=0;j<o.GetLength(1);j++)o[i,j]+=h*d[i,j];return o;}
    public new double[] Step(double[] z,double[] u,double[,] ja=null,double[,] jb=null)
    {
        const double dt=.1,h=.025;int n=z.Length;var ps=FinsRovLiftedOperator.PoseScale;var vs=FinsRovLiftedOperator.VelocityScale;
        var p=FinsRovBodyFrame.Pose(z);var x=new double[8];x[0]=p[3];x[1]=p[5];Array.Copy(z,7,x,2,6);
        var a=ja==null?null:new double[8,8];var b=ja==null?null:new double[8,6];if(a!=null)for(int i=0;i<8;i++)a[i,i]=1;
        for(int sub=0;sub<4;sub++){
            double[] k1,k2,k3,k4;double[,] a1,a2,a3,a4,b1,b2,b3,b4;
            Stage(x,u,a,b,out k1,out a1,out b1);
            Stage(Add(x,k1,h/2),u,Add(a,a1,h/2),Add(b,b1,h/2),out k2,out a2,out b2);
            Stage(Add(x,k2,h/2),u,Add(a,a2,h/2),Add(b,b2,h/2),out k3,out a3,out b3);
            Stage(Add(x,k3,h),u,Add(a,a3,h),Add(b,b3,h),out k4,out a4,out b4);
            for(int i=0;i<8;i++){
                x[i]+=h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6;
                if(a!=null){for(int j=0;j<8;j++)a[i,j]+=h*(a1[i,j]+2*a2[i,j]+2*a3[i,j]+a4[i,j])/6;for(int j=0;j<6;j++)b[i,j]+=h*(b1[i,j]+2*b2[i,j]+2*b3[i,j]+b4[i,j])/6;}
            }
        }
        var result=new double[n];result[0]=1;var v0=new double[6];var v1=new double[6];
        for(int i=0;i<6;i++){result[7+i]=x[2+i];v0[i]=z[7+i]*vs[i];v1[i]=x[2+i]*vs[i];}
        var pose=FinsRovBodyFrame.StartPoseAverage(p,v0,v1,dt);for(int i=0;i<6;i++)result[1+i]=(pose[i]-FinsRovLiftedOperator.PoseCenter[i])/ps[i];
        if(ja==null)return result;
        Array.Clear(ja,0,ja.Length);Array.Clear(jb,0,jb.Length);
        for(int i=0;i<6;i++){
            ja[7+i,4]=a[2+i,0]*ps[3];ja[7+i,6]=a[2+i,1]*ps[5];
            for(int j=0;j<6;j++){ja[7+i,7+j]=a[2+i,2+j];jb[7+i,j]=b[2+i,j];}
        }
        double[,] pj,vj0,vj1;FinsRovBodyFrame.StartPoseAverageJacobians(p,v0,v1,dt,out pj,out vj0,out vj1);
        for(int i=0;i<6;i++){
            for(int j=0;j<12;j++){
                double d=j<6?pj[i,j]*ps[j]:vj0[i,j-6]*vs[j-6];
                for(int l=0;l<6;l++)d+=vj1[i,l]*vs[l]*ja[7+l,1+j];ja[1+i,1+j]=d/ps[i];
            }
            for(int j=0;j<6;j++){double d=0;for(int l=0;l<6;l++)d+=vj1[i,l]*vs[l]*jb[7+l,j];jb[1+i,j]=d/ps[i];}
        }
        return result;
    }
    public double CheckEquivalent(double[] z,double[] u)
    {
        int n=z.Length;var a=new double[n,n];var b=new double[n,6];var ra=new double[n,n];var rb=new double[n,6];
        var value=Step(z,u,a,b);var reference=base.Step(z,u,ra,rb);double e=0;
        for(int i=0;i<n;i++){e=Math.Max(e,Math.Abs(value[i]-reference[i]));for(int j=0;j<n;j++)e=Math.Max(e,Math.Abs(a[i,j]-ra[i,j]));for(int j=0;j<6;j++)e=Math.Max(e,Math.Abs(b[i,j]-rb[i,j]));}
        if(e>2e-6)throw new InvalidOperationException("Compact Fossen reference disagreement "+e);
        return e;
    }
}
