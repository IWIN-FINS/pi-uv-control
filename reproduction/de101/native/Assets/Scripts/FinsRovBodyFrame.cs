using System;

// Local Unity xyz vectors, world Unity position and Euler x/y/z angles.
// Rotation order Ry Rx Rz matches Quaternion.Euler. ROS must adapt its axes.
public static class FinsRovBodyFrame
{
    public static double[,] Rotation(double[] p)
    {
        double a=p[3],b=p[4],c=p[5],ca=Math.Cos(a),sa=Math.Sin(a),
            cb=Math.Cos(b),sb=Math.Sin(b),cc=Math.Cos(c),sc=Math.Sin(c);
        return new double[,]{{cb*cc+sb*sa*sc,-cb*sc+sb*sa*cc,sb*ca},
            {ca*sc,ca*cc,-sa},{-sb*cc+cb*sa*sc,sb*sc+cb*sa*cc,cb*ca}};
    }
    public static double[] Pose(double[] z)
    {
        var p=new double[6];
        for(int i=0;i<6;i++)p[i]=FinsRovLiftedOperator.DecodePose(z,i);
        return p;
    }
    public static double[] ToBody(double[] p,double[] world)
    {
        var r=Rotation(p);var result=new double[6];
        for(int block=0;block<6;block+=3)
            for(int i=0;i<3;i++)for(int j=0;j<3;j++)result[block+i]+=r[j,i]*world[block+j];
        return result;
    }
    public static float[] ToBody(float[] p,float[] world)
    {
        var pd=Array.ConvertAll(p,x=>(double)x);var vd=Array.ConvertAll(world,x=>(double)x);
        return Array.ConvertAll(ToBody(pd,vd),x=>(float)x);
    }
    public static double[,] RateMatrix(double[] p)
    {
        double a=p[3],c=p[5],ca=Math.Cos(a);
        if(Math.Abs(ca)<.1d)throw new InvalidOperationException("Body-frame Euler chart near singularity");
        var r=Rotation(p);var k=new double[6,6];
        for(int i=0;i<3;i++)for(int j=0;j<3;j++)k[i,j]=r[i,j];
        k[3,3]=Math.Cos(c);k[3,4]=-Math.Sin(c);
        k[4,3]=Math.Sin(c)/ca;k[4,4]=Math.Cos(c)/ca;
        k[5,3]=Math.Tan(a)*Math.Sin(c);k[5,4]=Math.Tan(a)*Math.Cos(c);k[5,5]=1d;
        return k;
    }
    static double[,] RateDerivative(double[] p,int axis)
    {
        return FinsRovHybridOperator.De100RateDerivative(p,axis);
    }
    static double[] Multiply(double[,] matrix,double[] vector)
    {
        var result=new double[6];for(int i=0;i<6;i++)for(int j=0;j<6;j++)result[i]+=matrix[i,j]*vector[j];return result;
    }
    // Endpoint-average velocity, with the body-to-world map frozen at eta_k.
    public static double[] StartPoseAverage(double[] p,double[] velocity0,double[] velocity1,double dt)
    {
        var k=RateMatrix(p);var average=new double[6];
        for(int i=0;i<6;i++)average[i]=.5*(velocity0[i]+velocity1[i]);
        var rate=Multiply(k,average);var result=(double[])p.Clone();
        for(int i=0;i<6;i++)result[i]+=dt*rate[i];return result;
    }
    public static void StartPoseAverageJacobians(double[] p,double[] velocity0,double[] velocity1,double dt,
        out double[,] poseJacobian,out double[,] velocity0Jacobian,out double[,] velocity1Jacobian)
    {
        var k=RateMatrix(p);var average=new double[6];
        for(int i=0;i<6;i++)average[i]=.5*(velocity0[i]+velocity1[i]);
        poseJacobian=new double[6,6];velocity0Jacobian=new double[6,6];velocity1Jacobian=new double[6,6];
        for(int i=0;i<6;i++) {
            poseJacobian[i,i]=1d;
            for(int j=0;j<6;j++)velocity0Jacobian[i,j]=velocity1Jacobian[i,j]=.5*dt*k[i,j];
        }
        for(int axis=3;axis<6;axis++) {
            var derivative=Multiply(RateDerivative(p,axis),average);
            for(int i=0;i<6;i++)poseJacobian[i,axis]+=dt*derivative[i];
        }
    }
    // Explicit midpoint RK2: predict the mid-pose with nu_k, then integrate
    // the endpoint-average body velocity through J at that mid-pose.
    public static double[] MidpointPose(double[] p,double[] velocity0,double[] velocity1,double dt)
    {
        var k0=RateMatrix(p);var halfRate=Multiply(k0,velocity0);var mid=(double[])p.Clone();
        for(int i=0;i<6;i++)mid[i]+=.5*dt*halfRate[i];
        var km=RateMatrix(mid);var average=new double[6];
        for(int i=0;i<6;i++)average[i]=.5*(velocity0[i]+velocity1[i]);
        var rate=Multiply(km,average);var result=(double[])p.Clone();
        for(int i=0;i<6;i++)result[i]+=dt*rate[i];return result;
    }
    // Physical-coordinate derivatives: d eta_next / d eta_k,
    // d eta_next / d nu_k, and d eta_next / d nu_{k+1}.
    public static void MidpointJacobians(double[] p,double[] velocity0,double[] velocity1,double dt,
        out double[,] poseJacobian,out double[,] velocity0Jacobian,out double[,] velocity1Jacobian)
    {
        var k0=RateMatrix(p);var midpoint=(double[])p.Clone();var k0v=Multiply(k0,velocity0);
        for(int i=0;i<6;i++)midpoint[i]+=.5*dt*k0v[i];
        var km=RateMatrix(midpoint);var average=new double[6];
        for(int i=0;i<6;i++)average[i]=.5*(velocity0[i]+velocity1[i]);
        var dRateMid=new double[6][];
        for(int q=3;q<6;q++)dRateMid[q]=Multiply(RateDerivative(midpoint,q),average);
        poseJacobian=new double[6,6];velocity0Jacobian=new double[6,6];velocity1Jacobian=new double[6,6];
        for(int i=0;i<6;i++){poseJacobian[i,i]=1d;velocity1Jacobian[i,i]=0d;}
        for(int axis=0;axis<6;axis++) {
            var dMidPose=new double[6];dMidPose[axis]=1d;
            if(axis>=3) {
                var dK0v=Multiply(RateDerivative(p,axis),velocity0);
                for(int q=0;q<6;q++)dMidPose[q]+=.5*dt*dK0v[q];
            }
            for(int i=0;i<6;i++) {
                double effect=0d;
                for(int q=3;q<6;q++)effect+=dRateMid[q][i]*dMidPose[q];
                poseJacobian[i,axis]+=dt*effect;
                double midpointVelocityEffect=0d;
                for(int q=3;q<6;q++)midpointVelocityEffect+=dRateMid[q][i]*(.5*dt*k0[q,axis]);
                velocity0Jacobian[i,axis]=.5*dt*km[i,axis]+dt*midpointVelocityEffect;
                velocity1Jacobian[i,axis]=.5*dt*km[i,axis];
            }
        }
    }
    public static double[] Rates(double[] z)
    {
        var k=RateMatrix(Pose(z));var result=new double[6];
        for(int i=0;i<6;i++)for(int j=0;j<6;j++)result[i]+=k[i,j]*FinsRovLiftedOperator.DecodeVelocity(z,j);
        return result;
    }
    public static double[,] RatesJacobian(double[] z)
    {
        var p=Pose(z);var k=RateMatrix(p);var jac=new double[6,z.Length];
        for(int i=0;i<6;i++)for(int j=0;j<6;j++)jac[i,7+j]=k[i,j]*FinsRovLiftedOperator.VelocityScale[j];
        for(int axis=3;axis<6;axis++) {
            var derivative=RateDerivative(p,axis);
            for(int i=0;i<6;i++)for(int j=0;j<6;j++)
                jac[i,1+axis]+=derivative[i,j]*FinsRovLiftedOperator.DecodeVelocity(z,j)*FinsRovLiftedOperator.PoseScale[axis];
        }
        return jac;
    }
}
