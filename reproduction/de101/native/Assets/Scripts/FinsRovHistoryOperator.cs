using System;
using System.Collections.Generic;

// Controlled EDMDc augmented with previous requested wrench. Horizontal
// translation invariance is enforced during fitting, not by deleting coefficients
// after fitting. Six memory observables propagate exactly: h[k+1]=u[k].
public sealed class FinsRovHistoryOperator
{
    public const int Size = 25;
    public readonly double[,] A = new double[Size,Size];
    public readonly double[,] B = new double[Size,6];

    public void BuildPrior(FinsRovLiftedOperator prior)
    {
        for (int i=0;i<19;i++)
        {
            for (int j=0;j<19;j++) A[i,j]=prior.A[i,j];
            for (int j=0;j<6;j++) B[i,j]=prior.B[i,j];
        }
        for (int a=0;a<6;a++) B[19+a,a]=1d;
    }

    public static double[] Lift(float[] pose,float[] velocity,float[] previousInput)
    {
        var z=new double[Size];
        Array.Copy(FinsRovLiftedOperator.Lift(pose,velocity),z,19);
        for (int a=0;a<6;a++) z[19+a]=previousInput[a]/FinsRovLiftedOperator.WrenchScale[a];
        return z;
    }

    public void Fit(IReadOnlyList<FinsRovLiftedOperator.Transition> samples,int count,bool memory)
    {
        for (int row=7;row<19;row++)
        {
            var columns=new List<int>();
            // No absolute horizontal position can affect velocities or v|v|.
            for (int j=0;j<Size+6;j++)
                if (j!=1 && j!=3 && (memory || j<19 || j>=Size)) columns.Add(j);
            int n=columns.Count;
            var gram=new double[n,n]; var rhs=new double[n];
            for (int k=0;k<count;k++)
            {
                var s=samples[k];
                float[] previous=k>0 && samples[k-1].Segment==s.Segment ? samples[k-1].Wrench : new float[6];
                double[] z=Lift(s.Pose,s.Velocity,previous);
                double[] u=FinsRovLiftedOperator.Input(s.Wrench);
                var feature=new double[Size+6];
                Array.Copy(z,feature,Size); Array.Copy(u,0,feature,Size,6);
                double prediction=0;
                for (int j=0;j<Size;j++) prediction+=A[row,j]*z[j];
                for (int j=0;j<6;j++) prediction+=B[row,j]*u[j];
                double target=FinsRovLiftedOperator.Lift(s.NextPose,s.NextVelocity)[row]-prediction;
                for (int i=0;i<n;i++)
                {
                    rhs[i]+=feature[columns[i]]*target;
                    for (int j=0;j<n;j++) gram[i,j]+=feature[columns[i]]*feature[columns[j]];
                }
            }
            for (int i=0;i<n;i++) gram[i,i]+=row<13 ? .30d : 1d;
            var delta=FinsRovLiftedOperator.Solve(gram,rhs);
            for (int i=0;i<n;i++)
            {
                int col=columns[i];
                if (col<Size) A[row,col]+=delta[i]; else B[row,col-Size]+=delta[i];
            }
        }
        for (int a=0;a<6;a++)
        {
            int p=1+a,v=7+a;
            double factor=.1d*FinsRovLiftedOperator.VelocityScale[a]/FinsRovLiftedOperator.PoseScale[a];
            for (int j=0;j<Size;j++) A[p,j]=(j==p?1d:0d)+factor*A[v,j];
            for (int j=0;j<6;j++) B[p,j]=factor*B[v,j];
        }
    }

    public double[] Step(double[] z,double[] u)
    {
        var next=new double[Size];
        for (int i=0;i<Size;i++)
        {
            for (int j=0;j<Size;j++) next[i]+=A[i,j]*z[j];
            for (int j=0;j<6;j++) next[i]+=B[i,j]*u[j];
        }
        return next;
    }
}
