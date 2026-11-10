using System;

public sealed partial class FinsRovHybridBasisSmoke
{
    // Same physics42 velocity coefficients, all nonlinear features recomputed
    // at every prediction step. Analytic feature Jacobians enter the same MPC.
    double[] PhysicalFeatures(double[] z,double[,] j)
    {
        var f=(double[])z.Clone();f[0]=1;
        if(j!=null)for(int i=1;i<13;i++)j[i,i]=1;
        for(int a=0;a<6;a++) {f[13+a]=z[7+a]*Math.Abs(z[7+a]);if(j!=null)j[13+a,7+a]=2*Math.Abs(z[7+a]);}
        int k=19;
        for(int a=0;a<6;a++)for(int b=a+1;b<6;b++,k++) {
            f[k]=z[7+a]*z[7+b];if(j!=null){j[k,7+a]=z[7+b];j[k,7+b]=z[7+a];}
        }
        foreach(int a in new[]{3,5}){int row=a==3?34:35;double angle=z[1+a]*FinsRovLiftedOperator.PoseScale[a];f[row]=Math.Sin(angle);if(j!=null)j[row,1+a]=Math.Cos(angle)*FinsRovLiftedOperator.PoseScale[a];}
        if(j!=null)for(int i=36;i<42;i++)j[i,i]=1;
        return f;
    }
    double[] PhysicalReliftStep(double[] z,double[] u,double[,] ja,double[,] jb)
    {
        if(hybrid.BodyVelocity)throw new Exception("Historical static relift is world-frame only; use body Koopman predictor");
        if(hybrid.Size!=42 || hybrid.Data.latent!=0)throw new Exception("Static hybrid comparison requires physical42");
        var jf=ja==null?null:new double[42,42];var f=PhysicalFeatures(z,jf);
        var next=new double[42];next[0]=1;
        for(int a=0;a<6;a++) {
            int vi=7+a,pi=1+a;double factor=.1*FinsRovLiftedOperator.VelocityScale[a]/FinsRovLiftedOperator.PoseScale[a];
            for(int j=0;j<42;j++)next[vi]+=hybrid.A[vi,j]*f[j];
            for(int j=0;j<6;j++)next[vi]+=hybrid.B[vi,j]*u[j];
            next[pi]=z[pi]+factor*next[vi];next[36+a]=u[a];
            if(ja==null)continue;
            for(int j=0;j<42;j++){
                for(int l=0;l<42;l++)ja[vi,j]+=hybrid.A[vi,l]*jf[l,j];
                ja[pi,j]=(j==pi?1:0)+factor*ja[vi,j];
            }
            for(int j=0;j<6;j++){jb[vi,j]=hybrid.B[vi,j];jb[pi,j]=factor*jb[vi,j];}
            jb[36+a,a]=1;
        }
        var jn=ja==null?null:new double[42,42];var result=PhysicalFeatures(next,jn);
        if(ja!=null){var olda=(double[,])ja.Clone();var oldb=(double[,])jb.Clone();Array.Clear(ja,0,ja.Length);Array.Clear(jb,0,jb.Length);
            for(int i=0;i<42;i++)for(int l=0;l<42;l++)if(jn[i,l]!=0){for(int j=0;j<42;j++)ja[i,j]+=jn[i,l]*olda[l,j];for(int j=0;j<6;j++)jb[i,j]+=jn[i,l]*oldb[l,j];}}
        return result;
    }
}
