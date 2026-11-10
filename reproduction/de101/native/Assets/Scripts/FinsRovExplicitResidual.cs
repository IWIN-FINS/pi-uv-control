using System;
using System.IO;
using UnityEngine;

// Schema 7. The nominal Fossen delegate is actually evaluated at EVERY step.
// Six-axis residual acceleration states remain dynamic across the MPC horizon.
public sealed class FinsRovExplicitResidual
{
    [Serializable] public class Nominal { public double[] mass,linearDrag,quadraticDrag,spring,bias; }
    [Serializable] public class Branch { public string kind; public int offset,featureCount; public double[] G,A,E,B,D; }
    [Serializable] public class Network { public double[] mean,scale,latentMean,latentScale,w1,b1,w2,b2,w3,b3; }
    [Serializable] public class Model { public int schemaVersion,size; public string modelType,mode,residualUnits;public double dt;public double[] residualScale,poseCenter,poseScale,velocityScale,inputScale;public Nominal nominal;public Branch[] branches;public Network network; }
    [Serializable] public class Case {public float[] p,v,previousInput;public double[] z,u,worldU,next;}
    [Serializable] public class Cases {public Case[] cases;}
    public Model Data;
    public delegate double[] NominalStep(double[] z,double[] u,double[,] ja,double[,] jb);
    public NominalStep Baseline;
    public int Size=>Data.size;
    static readonly int[] Core={0,2,4,5,6,7,8,9,10,11,12,36,37,38,39,40,41};
    static readonly int[] Inputs={2,4,5,6,7,8,9,10,11,12,36,37,38,39,40,41};
    static void CheckArray(double[] values,int size,bool positive=false)
    {
        if(values==null||values.Length!=size)throw new InvalidDataException("Residual array size");
        foreach(double v in values)if(double.IsNaN(v)||double.IsInfinity(v)||(positive&&v<=0))throw new InvalidDataException("Residual nonfinite or invalid scale");
    }
    public void Load(string path)
    {
        Data=JsonUtility.FromJson<Model>(File.ReadAllText(path));
        if(Data.schemaVersion!=7||Data.modelType!="calibrated_fossen_plus_dynamic_residual"||Data.residualUnits!="acceleration"||Data.dt!=.1||Data.branches==null||Data.branches.Length<1||Data.branches.Length>2||Size!=42+6*Data.branches.Length)
            throw new InvalidDataException("Unsupported explicit residual model");
        int count=0;
        foreach(var b in Data.branches) {
            int d=b.kind=="physical"?41:b.kind=="mlp"?25:0;
            if(d==0||b.featureCount!=d||b.offset!=42+6*count++)
                throw new InvalidDataException("Residual branch dimensions");
            CheckArray(b.G,6*d);CheckArray(b.E,6*d);CheckArray(b.A,36);CheckArray(b.B,36);CheckArray(b.D,36);
        }
        // JsonUtility may materialize a null nested object with null fields.
        if(Data.network!=null&&Data.network.w1==null)Data.network=null;
        bool physical=Data.branches[0].kind=="physical";
        if((Data.mode=="physics_residual"&&(!physical||Data.branches.Length!=1||Data.network!=null))||
           (Data.mode=="mlp_residual"&&(physical||Data.branches.Length!=1||Data.network==null))||
           (Data.mode=="dual_residual"&&(!physical||Data.branches.Length!=2||Data.branches[1].kind!="mlp"||Data.network==null))||
           (Data.mode!="physics_residual"&&Data.mode!="mlp_residual"&&Data.mode!="dual_residual"))throw new InvalidDataException("Residual branch mode mismatch");
        if(Data.network!=null) {
            var n=Data.network;CheckArray(n.w1,512);CheckArray(n.b1,32);CheckArray(n.w2,512);CheckArray(n.b2,16);CheckArray(n.w3,128);CheckArray(n.b3,8);
            CheckArray(n.mean,16);CheckArray(n.scale,16,true);CheckArray(n.latentMean,8);CheckArray(n.latentScale,8,true);
        }
        CheckArray(Data.poseCenter,6);CheckArray(Data.poseScale,6,true);CheckArray(Data.velocityScale,6,true);CheckArray(Data.inputScale,6,true);CheckArray(Data.residualScale,6,true);
        if(Data.nominal==null)throw new InvalidDataException("Missing frozen nominal parameters");
        CheckArray(Data.nominal.mass,6,true);CheckArray(Data.nominal.linearDrag,6);CheckArray(Data.nominal.quadraticDrag,6);CheckArray(Data.nominal.spring,6);CheckArray(Data.nominal.bias,6);
        for(int i=0;i<6;i++) {
            if(Math.Abs(Data.poseCenter[i]-FinsRovLiftedOperator.PoseCenter[i])>1e-12||Math.Abs(Data.poseScale[i]-FinsRovLiftedOperator.PoseScale[i])>1e-12||Math.Abs(Data.velocityScale[i]-FinsRovLiftedOperator.VelocityScale[i])>1e-12||Math.Abs(Data.inputScale[i]-FinsRovLiftedOperator.WrenchScale[i])>1e-12||Math.Abs(Data.residualScale[i]-Data.velocityScale[i]/Data.dt)>1e-12)
                throw new InvalidDataException("Residual deployment units/scales differ");
        }
    }
    public void Bind(NominalStep step,float[] mass,float[] linear,float[] quadratic,float[] spring,float[] bias,string modelPath)
    {
        Baseline=step;var actual=new[]{mass,linear,quadratic,spring,bias};var frozen=new[]{Data.nominal.mass,Data.nominal.linearDrag,Data.nominal.quadraticDrag,Data.nominal.spring,Data.nominal.bias};
        for(int k=0;k<5;k++)for(int i=0;i<6;i++)if(Math.Abs(actual[k][i]-frozen[k][i])>1e-9)throw new InvalidDataException("Frozen nominal Fossen parameter mismatch");
        var cases=JsonUtility.FromJson<Cases>(File.ReadAllText(Path.Combine(Path.GetDirectoryName(modelPath),"residual_cases.json")));double error=0,zeroError=0,memory=0;
        foreach(var c in cases.cases) {
            var initial=Initialize(c.p,c.v,c.previousInput);for(int i=0;i<Size;i++)error=Math.Max(error,Math.Abs(initial[i]-c.z[i]));
            var next=Step(c.z,c.u,null,null);for(int i=0;i<Size;i++)error=Math.Max(error,Math.Abs(next[i]-c.next[i]));
            if(c.worldU!=null){var wn=StepWorld(c.z,c.worldU,null,null);for(int i=0;i<Size;i++)error=Math.Max(error,Math.Abs(wn[i]-c.next[i]));}
            var z=(double[])c.z.Clone();var baseline=(double[])c.z.Clone();
            for(int j=0;j<12;j++) {z=Step(z,c.u,null,null,false);baseline=Baseline(baseline,c.u,null,null);for(int i=1;i<13;i++)zeroError=Math.Max(zeroError,Math.Abs(z[i]-baseline[i]));}
            var pert=(double[])c.z.Clone();pert[42]+=.01;var pn=Step(pert,c.u,null,null);for(int i=0;i<Size;i++)memory=Math.Max(memory,Math.Abs(pn[i]-next[i]));
        }
        if(error>1e-5||zeroError>1e-12||memory<1e-5)throw new InvalidDataException($"Residual verification failed step={error} zero={zeroError} memory={memory}");
        Debug.Log($"FINSSIM_EXPLICIT_RESIDUAL_CHECK step={error:R} zero12={zeroError:R} memory={memory:R}");
    }
    public double[] Initialize(float[] pose,float[] velocity,float[] previous)
    {
        var z=new double[Size];z[0]=1;
        for(int i=0;i<6;i++){z[1+i]=(pose[i]-Data.poseCenter[i])/Data.poseScale[i];z[7+i]=velocity[i]/Data.velocityScale[i];z[36+i]=previous[i]/Data.inputScale[i];}
        foreach(var b in Data.branches){var phi=Features(z,b,null);for(int i=0;i<6;i++)for(int j=0;j<phi.Length;j++)z[b.offset+i]+=b.G[i*phi.Length+j]*phi[j];}
        return z;
    }
    double[] Features(double[] z,Branch b,double[,] jac)
    {
        int d=b.featureCount;var phi=new double[d];
        int coreCount=b.kind=="physical"?11:17;
        for(int j=0;j<coreCount;j++){phi[j]=z[Core[j]];if(jac!=null&&j!=0)jac[j,Core[j]]=1;}
        if(b.kind=="physical") {
            int at=11;
            for(int i=0;i<6;i++){double v=z[7+i];phi[at]=v*Math.Abs(v);if(jac!=null)jac[at,7+i]=2*Math.Abs(v);at++;}
            for(int i=0;i<6;i++)for(int j=i+1;j<6;j++){phi[at]=z[7+i]*z[7+j];if(jac!=null){jac[at,7+i]=z[7+j];jac[at,7+j]=z[7+i];}at++;}
            var p=FinsRovBodyFrame.Pose(z);var r=FinsRovBodyFrame.Rotation(p);int[] cols={0,2,1};
            phi[32]=r[1,0];phi[33]=r[1,2];phi[40]=r[1,1];
            for(int j=0;j<6;j++){phi[34+j]=z[36+j];if(jac!=null)jac[34+j,36+j]=1;}
            if(jac!=null)for(int axis=3;axis<6;axis++) {
                double old=p[axis];p[axis]=old+1e-6;var plus=FinsRovBodyFrame.Rotation(p);p[axis]=old-1e-6;var minus=FinsRovBodyFrame.Rotation(p);p[axis]=old;
                int[] rows={32,33,40};for(int j=0;j<3;j++)jac[rows[j],1+axis]=(plus[1,cols[j]]-minus[1,cols[j]])/2e-6*Data.poseScale[axis];
            }
            return phi;
        }
        var net=Data.network;var input=new double[16];for(int j=0;j<16;j++)input[j]=(z[Inputs[j]]-net.mean[j])/net.scale[j];
        var h1=new double[32];var j1=jac==null?null:new double[32,16];
        for(int i=0;i<32;i++) {
            double t=net.b1[i];for(int j=0;j<16;j++)t+=net.w1[j*32+i]*input[j];h1[i]=Math.Max(0,t);
            if(jac!=null&&t>0)for(int j=0;j<16;j++)j1[i,j]=net.w1[j*32+i]/net.scale[j];
        }
        var h2=new double[16];var j2=jac==null?null:new double[16,16];
        for(int i=0;i<16;i++) {
            double t=net.b2[i];for(int j=0;j<32;j++)t+=net.w2[j*16+i]*h1[j];h2[i]=Math.Max(0,t);
            if(jac!=null&&t>0)for(int l=0;l<16;l++)for(int j=0;j<32;j++)j2[i,l]+=net.w2[j*16+i]*j1[j,l];
        }
        for(int i=0;i<8;i++) {
            double t=net.b3[i];for(int j=0;j<16;j++)t+=net.w3[j*8+i]*h2[j];phi[17+i]=(Math.Max(0,t)-net.latentMean[i])/net.latentScale[i];
            if(jac!=null&&t>0)for(int l=0;l<16;l++)for(int j=0;j<16;j++)jac[17+i,Inputs[l]]+=net.w3[j*8+i]*j2[j,l]/net.latentScale[i];
        }
        return phi;
    }
    public double[] Step(double[] z,double[] u,double[,] ja,double[,] jb,bool enabled=true)
    {
        if(Baseline==null)throw new InvalidOperationException("Residual Fossen baseline is unbound");
        var next=Baseline(z,u,ja,jb);Array.Clear(next,13,23);
        foreach(var b in Data.branches) {
            var fj=ja==null?null:new double[b.featureCount,Size];var phi=Features(z,b,fj);
            for(int i=0;i<6;i++) {
                double accelerationToVelocity=Data.dt*Data.residualScale[i]/Data.velocityScale[i];
                next[b.offset+i]=0;
                for(int j=0;j<6;j++) {next[b.offset+i]+=b.A[i*6+j]*z[b.offset+j]+b.B[i*6+j]*u[j];if(enabled)next[7+i]+=accelerationToVelocity*b.D[i*6+j]*u[j];}
                for(int j=0;j<phi.Length;j++)next[b.offset+i]+=b.E[i*phi.Length+j]*phi[j];
                if(enabled)next[7+i]+=accelerationToVelocity*z[b.offset+i];
                if(ja==null)continue;
                for(int j=0;j<Size;j++){ja[b.offset+i,j]=0;for(int l=0;l<phi.Length;l++)ja[b.offset+i,j]+=b.E[i*phi.Length+l]*fj[l,j];}
                for(int j=0;j<6;j++){ja[b.offset+i,b.offset+j]+=b.A[i*6+j];jb[b.offset+i,j]=b.B[i*6+j];if(enabled)jb[7+i,j]+=accelerationToVelocity*b.D[i*6+j];}
                if(enabled)ja[7+i,b.offset+i]+=accelerationToVelocity;
            }
        }
        var p=FinsRovBodyFrame.Pose(z);var v0=new double[6];var v1=new double[6];
        for(int i=0;i<6;i++){v0[i]=z[7+i]*Data.velocityScale[i];v1[i]=next[7+i]*Data.velocityScale[i];}
        var pose=FinsRovBodyFrame.StartPoseAverage(p,v0,v1,Data.dt);for(int i=0;i<6;i++)next[1+i]=(pose[i]-Data.poseCenter[i])/Data.poseScale[i];
        if(ja!=null) {
            FinsRovBodyFrame.StartPoseAverageJacobians(p,v0,v1,Data.dt,out var jp,out var j0,out var j1);
            for(int i=0;i<6;i++) {
                for(int j=0;j<Size;j++) {
                    double v=0;for(int l=0;l<6;l++)v+=j1[i,l]*Data.velocityScale[l]*ja[7+l,j];
                    if(j>=1&&j<7)v+=jp[i,j-1]*Data.poseScale[j-1];if(j>=7&&j<13)v+=j0[i,j-7]*Data.velocityScale[j-7];ja[1+i,j]=v/Data.poseScale[i];
                }
                for(int j=0;j<6;j++){double v=0;for(int l=0;l<6;l++)v+=j1[i,l]*Data.velocityScale[l]*jb[7+l,j];jb[1+i,j]=v/Data.poseScale[i];}
            }
            for(int i=13;i<36;i++){for(int j=0;j<Size;j++)ja[i,j]=0;for(int j=0;j<6;j++)jb[i,j]=0;}
        }
        return next;
    }
    public double[] StepWorld(double[] z,double[] world,double[,] ja,double[,] jb)
    {
        var p=FinsRovBodyFrame.Pose(z);var r=FinsRovBodyFrame.Rotation(p);var body=new double[6];var map=new double[6,6];
        for(int block=0;block<6;block+=3)for(int i=0;i<3;i++)for(int j=0;j<3;j++){map[block+i,block+j]=r[j,i]*Data.inputScale[block+j]/Data.inputScale[block+i];body[block+i]+=map[block+i,block+j]*world[block+j];}
        var bodyJ=ja==null?null:new double[Size,6];var next=Step(z,body,ja,bodyJ);if(ja==null)return next;
        for(int i=0;i<Size;i++)for(int j=0;j<6;j++)for(int l=0;l<6;l++)jb[i,j]+=bodyJ[i,l]*map[l,j];
        for(int axis=3;axis<6;axis++) {
            double old=p[axis];p[axis]=old+1e-6;var plus=FinsRovBodyFrame.Rotation(p);p[axis]=old-1e-6;var minus=FinsRovBodyFrame.Rotation(p);p[axis]=old;
            for(int i=0;i<Size;i++)for(int block=0;block<6;block+=3)for(int l=0;l<3;l++)for(int j=0;j<3;j++)ja[i,1+axis]+=bodyJ[i,block+l]*(plus[j,l]-minus[j,l])/2e-6*Data.inputScale[block+j]/Data.inputScale[block+l]*world[block+j]*Data.poseScale[axis];
        }
        return next;
    }
}
