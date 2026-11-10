using System;

public sealed partial class FinsRovHybridOperator
{
    public FinsRovUnifiedFossen UnifiedFossen;
    public bool Unified=>(Data.schemaVersion>=15 && Data.schemaVersion<=18) || (Data.schemaVersion>=20 && Data.schemaVersion<=23);
    double[] UnifiedBodyStep(double[] z,double[] u,double[,] ja,double[,] jb)
    {
        int n=Size;var result=new double[n];
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++)result[i]+=A[i,j]*z[j];
            for(int j=0;j<6;j++)result[i]+=B[i,j]*u[j];
        }
        if(ja!=null){Array.Copy(A,ja,A.Length);Array.Copy(B,jb,B.Length);}
        if(Data.predictionType.StartsWith("C")){
            var na=ja==null?null:new double[n,n];var nb=ja==null?null:new double[n,6];
            var physical=UnifiedFossen.Step(z,u,na,nb);
            for(int i=0;i<6;i++){
                result[7+i]+=physical[7+i];
                if(ja!=null){for(int j=0;j<n;j++)ja[7+i,j]+=na[7+i,j];for(int j=0;j<6;j++)jb[7+i,j]+=nb[7+i,j];}
            }
        }
        if(Data.predictionType.EndsWith("B"))return result;
        var ps=FinsRovLiftedOperator.PoseScale;var vs=FinsRovLiftedOperator.VelocityScale;
        var p=FinsRovBodyFrame.Pose(z);var v0=new double[6];var v1=new double[6];
        for(int i=0;i<6;i++){v0[i]=z[7+i]*vs[i];v1[i]=result[7+i]*vs[i];}
        var pose=FinsRovBodyFrame.StartPoseAverage(p,v0,v1,.1);
        for(int i=0;i<6;i++)result[1+i]=(pose[i]-FinsRovLiftedOperator.PoseCenter[i])/ps[i];
        if(ja==null)return result;
        double[,] jp,j0,j1;FinsRovBodyFrame.StartPoseAverageJacobians(p,v0,v1,.1,out jp,out j0,out j1);
        for(int i=0;i<6;i++){
            for(int j=0;j<n;j++){
                double d=j>=1&&j<7?jp[i,j-1]*ps[j-1]:j>=7&&j<13?j0[i,j-7]*vs[j-7]:0;
                for(int l=0;l<6;l++)d+=j1[i,l]*vs[l]*ja[7+l,j];ja[1+i,j]=d/ps[i];
            }
            for(int j=0;j<6;j++){double d=0;for(int l=0;l<6;l++)d+=j1[i,l]*vs[l]*jb[7+l,j];jb[1+i,j]=d/ps[i];}
        }
        return result;
    }
}
