using System;
using System.IO;
using UnityEngine;

// Two explicit dictionaries: local Fossen features and a learned MLP encoder.
// The encoder runs at measurement initialization. Schema 3 follows Az+Bu with
// known nonlinear world-pose kinematics; historical world models use Az+Bu alone.
public sealed partial class FinsRovHybridOperator
{
    [Serializable] public class Model
    {
        public int size, hidden, latent;
        public int schemaVersion, hidden2;
        public string activation, encoderType;
        public string velocityFrame, wrenchFrame, poseFrame, kinematics;
        // Missing fields in historical JSON preserve the original 36+latent schema.
        public int physicalSize;
        // DE44: previous input remains an encoder input, outside the 36-D dictionary.
        public bool encoderHistoryInput;
        public int[] inputColumns;
        public double[] a,b,w1,b1,w2,b2,w3,b3,mean,scale,latentMean,latentScale;
        public double[] skipProjection,layerNormGamma,layerNormBeta;
        public double layerNormEpsilon;
        public double skipScalar,alpha;
        public bool de68Compact;public string factorialArm;public int[] skipIndices;
        public string dictionaryAblation;
        public double sampleDt;public string de99Arm;public double[] de99DeltaScale;
        public string de83Mode;public double[] de83Center,de83PoseBiasScale;
        public string predictionType;
        public FinsRovUnifiedFossen.Parameters nominal;
    }
    public Model Data;
    public FinsRovExplicitResidual Residual;
    [Serializable] public class TestCase { public float[] p,v,previousInput; public double[] z; }
    [Serializable] public class TestCases { public TestCase[] cases; }
    [Serializable] public class StepCase { public double[] z,u,worldU,next; }
    [Serializable] public class StepCases { public StepCase[] cases; }
    public double[,] A,B;
    public int Size=>Data.size;
    public double SampleDt=>Data.schemaVersion==14?Data.sampleDt:.1d;
    public bool BodyVelocity=>Data.velocityFrame=="body_unity_xyz";
    public bool UsesMidpointKinematics=>Data.kinematics=="unity_zxy_midpoint_rk2";
    public bool UsesStartPoseAverage=>Data.kinematics=="unity_zxy_start_pose_velocity_average" || Data.kinematics=="unity_zxy_learned_pose";
    public bool LearnsPose=>Data.kinematics=="unity_zxy_learned_pose";
    int PhysicalSize=>Data.physicalSize==0?36:Data.physicalSize;
    public void Load(string path)
    {
        Data=JsonUtility.FromJson<Model>(File.ReadAllText(path));
        Residual=null;
        if(Unified){
            if(Data.predictionType!=new[]{"K_A","K_B","C_A","C_B"}[Data.schemaVersion>=20?Data.schemaVersion-20:Data.schemaVersion-15] || Math.Abs(Data.sampleDt-.1)>1e-12)throw new Exception("DE51 prediction type/dt mismatch");
            UnifiedFossen=new FinsRovUnifiedFossen(Data.nominal);
        }
        if(Data.schemaVersion==7) {
            if(!BodyVelocity||Data.wrenchFrame!="body_unity_xyz"||Data.poseFrame!="world_unity_xyz"||Data.kinematics!="unity_zxy_start_pose_velocity_average")throw new Exception("Explicit residual coordinate mismatch");
            Residual=new FinsRovExplicitResidual();Residual.Load(path);A=new double[Size,Size];B=new double[Size,6];return;
        }
        if(!string.IsNullOrEmpty(Data.velocityFrame) && Data.velocityFrame!="world_unity_xyz" && !BodyVelocity)
            throw new Exception("Unsupported velocity coordinate frame");
        bool oldBodyStep=Data.schemaVersion==3 && Data.kinematics=="unity_zxy_semiimplicit";
        bool midpointBodyStep=Data.schemaVersion==4 && Data.kinematics=="unity_zxy_midpoint_rk2";
        bool startPoseStep=(Data.schemaVersion==5 || Data.schemaVersion==11 || Data.schemaVersion==13 || Data.schemaVersion==14 || Data.schemaVersion==15 || Data.schemaVersion==17 || Data.schemaVersion==20 || Data.schemaVersion==22) && Data.kinematics=="unity_zxy_start_pose_velocity_average";
        bool learnedPoseStep=(Data.schemaVersion==6 || Data.schemaVersion==12 || Data.schemaVersion==16 || Data.schemaVersion==18 || Data.schemaVersion==21 || Data.schemaVersion==23) && Data.kinematics=="unity_zxy_learned_pose";
        if(BodyVelocity && (!oldBodyStep && !midpointBodyStep && !startPoseStep && !learnedPoseStep || Data.wrenchFrame!="body_unity_xyz" || Data.poseFrame!="world_unity_xyz"))
            throw new Exception("Body model requires schema 3/semi-implicit or schema 4/midpoint kinematics with matching coordinates");
        if((PhysicalSize!=36&&PhysicalSize!=42) || Data.size!=PhysicalSize+Data.latent || Data.a.Length!=Size*Size || Data.b.Length!=Size*6)
            throw new Exception("Invalid hybrid dimensions");
        if(Data.schemaVersion<20 && (Data.schemaVersion==13 || Data.schemaVersion==14 || Unified) && (PhysicalSize!=36 || Data.size!=44 || !Data.encoderHistoryInput || Data.encoderType!="relu_layernorm_skip8"))
            throw new Exception("Invalid DE44 compact dictionary/encoder contract");
        if(Data.encoderHistoryInput && Data.schemaVersion!=13 && Data.schemaVersion!=14 && !Unified)
            throw new Exception("External encoder history requires DE44 schema 13");
        if(Data.schemaVersion==14 && Math.Abs(Data.sampleDt-.02d)>1e-10)throw new Exception("DE47 requires explicit native20ms model dt");
        if(!string.IsNullOrEmpty(Data.activation) && Data.activation!="relu" && Data.activation!="tanh")
            throw new Exception("Unknown encoder activation "+Data.activation);
        if(Data.latent>0) {
            int inputs=Data.inputColumns==null?10:Data.inputColumns.Length;
            int second=Data.hidden2>0?Data.hidden2:Data.latent;
            if(Data.hidden<=0 || Data.hidden2<0 || Data.b1.Length!=Data.hidden ||
               Data.w1.Length!=inputs*Data.hidden || Data.w2.Length!=Data.hidden*second || Data.b2.Length!=second ||
               Data.latentMean.Length!=Data.latent || Data.latentScale.Length!=Data.latent ||
               (Data.hidden2>0 && (Data.w3==null || Data.b3==null || Data.w3.Length!=Data.hidden2*Data.latent || Data.b3.Length!=Data.latent)))
                throw new Exception("Invalid encoder layer dimensions");
            if(Data.encoderType=="relu_layernorm_skip8" &&
               (inputs!=16 || Data.latent!=8 || Data.skipProjection==null || Data.skipProjection.Length!=8*16 ||
                Data.layerNormGamma==null || Data.layerNormGamma.Length!=8 ||
                Data.layerNormBeta==null || Data.layerNormBeta.Length!=8 || Data.layerNormEpsilon<=0d))
                throw new Exception("Invalid LayerNorm skip-projection encoder dimensions");
            if(Data.schemaVersion>=20 && Data.encoderType!="de68_factorial" && Data.encoderType!="de81_identity" && (PhysicalSize!=36 || Size!=52 || inputs!=16 || Data.hidden!=32 || Data.hidden2!=16 || Data.latent!=16 || Data.activation!="relu" || !Data.encoderHistoryInput || Data.encoderType!="relu_scalar_skip16_final_ln" || Data.alpha!=1 || Data.skipScalar!=.01 || Data.layerNormEpsilon!=1e-5 || (Data.skipProjection!=null && Data.skipProjection.Length!=0) || Data.layerNormGamma==null || Data.layerNormBeta==null || Data.layerNormGamma.Length!=16 || Data.layerNormBeta.Length!=16))throw new Exception("DE60 scalar16 encoder contract mismatch");
            if(Data.schemaVersion>=20 && Data.encoderType!="de68_factorial" && Data.encoderType!="de81_identity"){
                int[] expected={2,4,5,6,7,8,9,10,11,12,36,37,38,39,40,41};
                for(int i=0;i<16;i++)if(Data.inputColumns[i]!=expected[i] || !(Data.scale[i]>=.05) || !(Data.latentScale[i]>=.2))throw new Exception("DE60 input/statistics contract mismatch");
            }
            if(!string.IsNullOrEmpty(Data.encoderType) && Data.encoderType!="de68_factorial" && Data.encoderType!="de81_identity" && Data.encoderType!="relu_layernorm_skip8" && !(Data.schemaVersion>=20 && Data.encoderType=="relu_scalar_skip16_final_ln" || Data.encoderType=="de81_identity"))
                throw new Exception("Unknown encoder type "+Data.encoderType);
        }

        if(Data.encoderType=="de68_factorial") {
            if(Size!=52 || PhysicalSize!=36 || Data.latent!=16 || Data.hidden!=32 || Data.hidden2!=16 || Data.activation!="relu" || Data.skipScalar!=.01 || Data.alpha!=1 || Data.layerNormEpsilon!=1e-5 || Data.skipIndices==null || Data.skipIndices.Length!=16)throw new Exception("DE68 contract");
            string arm=Data.factorialArm;int d=arm=="A"?16:arm=="B"?10:arm=="C"?42:arm=="D"?36:0;
            if(d==0 || Data.inputColumns.Length!=d || Data.encoderHistoryInput!=(arm=="A"||arm=="C") || Data.de68Compact!=(arm=="C"||arm=="D"))throw new Exception("DE68 factor contract");
            int[] basic={2,4,5,6,7,8,9,10,11,12};
            for(int j=0;j<d;j++){int expected=(arm=="C"||arm=="D")?j:j<10?basic[j]:36+j-10;if(Data.inputColumns[j]!=expected || Data.scale[j]<.05)throw new Exception("DE68 input contract");}
            for(int j=0;j<16;j++){int coord=j<10?basic[j]:36+j-10;int wanted=Array.IndexOf(Data.inputColumns,coord);if(Data.skipIndices[j]!=wanted || Data.latentScale[j]<.2)throw new Exception("DE68 skip contract");}
            if(Data.de68Compact)for(int i=7;i<52;i++)for(int j=0;j<36;j++)if(Data.a[i*52+j]!=0)throw new Exception("DE68 compact physical bypass");
        }

        if(Data.encoderType=="de81_identity") {
            int d=Data.factorialArm=="G1"?10:Data.factorialArm=="G2"?16:Data.factorialArm=="G3"?36:Data.factorialArm=="G4"?42:0;
            bool history=Data.factorialArm=="G2" || Data.factorialArm=="G4";
            if(d==0 || PhysicalSize!=36 || Size!=36+d || Data.latent!=d || Data.inputColumns.Length!=d || Data.encoderHistoryInput!=history || !Data.de68Compact ||
              Data.hidden!=32 || Data.hidden2!=16 || Data.skipScalar!=.01 || Data.alpha!=1 || Data.layerNormEpsilon!=1e-5 || Data.activation!="relu" ||
              Data.layerNormGamma.Length!=d || Data.layerNormBeta.Length!=d || Data.skipIndices!=null || Data.skipProjection!=null)throw new Exception("DE81 dimensions/identity skip contract");
            int[] basic={2,4,5,6,7,8,9,10,11,12};
            for(int i=0;i<d;i++){int wanted=d>=36?i:i<10?basic[i]:36+i-10;if(Data.inputColumns[i]!=wanted || Data.scale[i]<.05 || Data.latentScale[i]<.2)throw new Exception("DE81 input/stats");}
            for(int i=7;i<Size;i++)for(int j=0;j<36;j++)if(Data.a[i*Size+j]!=0)throw new Exception("DE81 physical/constant bypass");
            for(int i=13;i<36;i++){for(int j=0;j<Size;j++)if(Data.a[i*Size+j]!=0)throw new Exception("DE81 auxiliary row");for(int j=0;j<6;j++)if(Data.b[i*6+j]!=0)throw new Exception("DE81 auxiliary input");}
        }
        A=new double[Size,Size]; B=new double[Size,6];
        for(int i=0;i<Size;i++) {
            for(int j=0;j<Size;j++) A[i,j]=Data.a[i*Size+j];
            for(int j=0;j<6;j++) B[i,j]=Data.b[i*6+j];
        }
        if(Path.GetFileName(path)=="hybrid44.json" || Data.physicalSize!=0) {
            var tests=JsonUtility.FromJson<TestCases>(File.ReadAllText(Path.Combine(Path.GetDirectoryName(path),"encoder_cases.json")));
            double error=0;
            foreach(var test in tests.cases) {
                var z=BodyVelocity?LiftBody(test.p,test.v,test.previousInput):Lift(test.p,test.v,test.previousInput);
                for(int i=0;i<Size;i++) error=Math.Max(error,Math.Abs(z[i]-test.z[i]));
            }
            if(error>1e-5) throw new Exception("Encoder deployment mismatch "+error);
            Debug.Log("FINSSIM_HYBRID_ENCODER_CHECK "+error);
        }
        if(BodyVelocity) {
            double conventionError=0;
            foreach(float yaw in new[]{-140f,0f,90f,170f}) {
                var p=new double[]{0,0,0,.13,yaw*Math.PI/180,-.19};var matrix=FinsRovBodyFrame.Rotation(p);
                var quaternion=Quaternion.Euler((float)(p[3]*180/Math.PI),(float)(p[4]*180/Math.PI),(float)(p[5]*180/Math.PI));
                for(int j=0;j<3;j++) {
                    var basis=quaternion*(j==0?Vector3.right:j==1?Vector3.up:Vector3.forward);
                    for(int i=0;i<3;i++)conventionError=Math.Max(conventionError,Math.Abs(matrix[i,j]-basis[i]));
                }
            }
            if(conventionError>1e-5)throw new Exception("Body rotation convention mismatch "+conventionError);
            Debug.Log("FINSSIM_BODY_ROTATION_CHECK "+conventionError);
            string stepPath=Path.Combine(Path.GetDirectoryName(path),"step_cases.json");
            if(!File.Exists(stepPath))throw new Exception("Schema 3 body model requires step_cases.json");
            var steps=JsonUtility.FromJson<StepCases>(File.ReadAllText(stepPath));double stepError=0;
            foreach(var test in steps.cases) {
                if(Unified)Debug.Log("FINSSIM_COMPACT_FOSSEN_CHECK "+UnifiedFossen.CheckEquivalent(test.z,test.u));
                var next=StepBody(test.z,test.u);
                for(int i=0;i<Size;i++)stepError=Math.Max(stepError,Math.Abs(next[i]-test.next[i]));
                if(test.worldU!=null) {
                    var adapted=StepWorld(test.z,test.worldU);
                    for(int i=0;i<Size;i++)stepError=Math.Max(stepError,Math.Abs(adapted[i]-test.next[i]));
                }
            }
            if(stepError>1e-5)throw new Exception("Body step deployment mismatch "+stepError);
            Debug.Log("FINSSIM_BODY_STEP_CHECK "+stepError);
        }
    }
    public double[] Lift(float[] p,float[] v,float[] previousInput=null)
    {
        return LiftBody(p,v,previousInput);
    }
    public double[] LiftWorld(float[] p,float[] worldVelocity,float[] previousBodyInput=null) =>
        LiftBody(p,BodyVelocity?FinsRovBodyFrame.ToBody(p,worldVelocity):worldVelocity,previousBodyInput);
    // Hardware callers already holding local six-axis velocities use this API.
    public double[] LiftBody(float[] p,float[] v,float[] previousInput=null)
    {
        if(Residual!=null)return Residual.Initialize(p,v,previousInput);
        var z=new double[Size];
        Array.Copy(FinsRovLiftedOperator.Lift(p,v),z,19);
        int k=19;
        for(int i=0;i<6;i++) for(int j=i+1;j<6;j++) z[k++]=z[7+i]*z[7+j];
        z[34]=Math.Sin(p[3]); z[35]=Math.Sin(p[5]);
        if(PhysicalSize==42) {
            if(previousInput==null || previousInput.Length!=6) throw new Exception("History model requires previous six-axis command");
            var normalized=FinsRovLiftedOperator.Input(previousInput);
            Array.Copy(normalized,0,z,36,6);
        }
        if(Data.dictionaryAblation=="N")Array.Clear(z,13,23);
        if(Data.latent==0 || Data.dictionaryAblation=="P") return z;
        double[] encoderInput=z;
        if(Data.encoderHistoryInput) {
            if(previousInput==null || previousInput.Length!=6) throw new Exception("DE44 encoder requires previous six-axis command");
            encoderInput=new double[42];Array.Copy(z,encoderInput,36);
            Array.Copy(FinsRovLiftedOperator.Input(previousInput),0,encoderInput,36,6);
        }
        int[] columns=Data.inputColumns??new[]{2,4,5,6,7,8,9,10,11,12};
        if(Data.mean.Length!=columns.Length || Data.scale.Length!=columns.Length || Data.w1.Length!=columns.Length*Data.hidden)
            throw new Exception("Invalid encoder input dimensions");
        var h=new double[Data.hidden];
        for(int i=0;i<h.Length;i++) {
            double t=Data.b1[i];
            for(int j=0;j<columns.Length;j++) t+=Data.w1[i*columns.Length+j]*(encoderInput[columns[j]]-Data.mean[j])/Data.scale[j];
            h[i]=Activate(t);
        }
        if(Data.hidden2>0) {
            var second=new double[Data.hidden2];
            for(int i=0;i<second.Length;i++) {
                double t=Data.b2[i];
                for(int j=0;j<h.Length;j++) t+=Data.w2[i*h.Length+j]*h[j];
                second[i]=Activate(t);
            }
            if(Data.encoderType=="relu_layernorm_skip8" || Data.encoderType=="relu_scalar_skip16_final_ln" || Data.encoderType=="de68_factorial" || Data.encoderType=="de81_identity") {
                var feature=new double[Data.latent];
                for(int i=0;i<Data.latent;i++) {
                    double t=Data.b3[i];
                    for(int j=0;j<second.Length;j++) t+=Data.w3[i*second.Length+j]*second[j];
                    double skip=0d;
                    if(Data.encoderType=="de68_factorial") { int si=Data.skipIndices[i];skip=si<0?0d:Data.skipScalar*(encoderInput[columns[si]]-Data.mean[si])/Data.scale[si]; }
                    else if(Data.encoderType=="relu_scalar_skip16_final_ln" || Data.encoderType=="de81_identity")skip=Data.skipScalar*(encoderInput[columns[i]]-Data.mean[i])/Data.scale[i];
                    else for(int j=0;j<columns.Length;j++) skip+=Data.skipProjection[i*columns.Length+j]*(encoderInput[columns[j]]-Data.mean[j])/Data.scale[j];
                    feature[i]=(Activate(t)-Data.latentMean[i])/Data.latentScale[i]+skip;
                }
                double mean=0d;for(int i=0;i<feature.Length;i++)mean+=feature[i]/feature.Length;
                double variance=0d;for(int i=0;i<feature.Length;i++){double centered=feature[i]-mean;variance+=centered*centered/feature.Length;}
                double inv=1d/Math.Sqrt(variance+Data.layerNormEpsilon);
                for(int i=0;i<feature.Length;i++)z[PhysicalSize+i]=Data.layerNormGamma[i]*(feature[i]-mean)*inv+Data.layerNormBeta[i];
                if(Data.de68Compact)Array.Clear(z,13,23);
                return z;
            }
            for(int i=0;i<Data.latent;i++) {
                double t=Data.b3[i];
                for(int j=0;j<second.Length;j++) t+=Data.w3[i*second.Length+j]*second[j];
                z[PhysicalSize+i]=(Activate(t)-Data.latentMean[i])/Data.latentScale[i];
            }
            return z;
        }
        for(int i=0;i<Data.latent;i++) {
            double t=Data.b2[i];
            for(int j=0;j<h.Length;j++) t+=Data.w2[i*h.Length+j]*h[j];
            z[PhysicalSize+i]=(Activate(t)-Data.latentMean[i])/Data.latentScale[i];
        }
        return z;
    }
    double Activate(double value) => Data.activation=="relu" ? Math.Max(0d,value) : Math.Tanh(value);
    public double[] Step(double[] z,double[] u,double[,] ja,double[,] jb)
    {
        if(BodyVelocity)return StepBody(z,u,ja,jb);
        var n=new double[Size];
        for(int i=0;i<Size;i++) {
            for(int j=0;j<Size;j++) n[i]+=A[i,j]*z[j];
            for(int j=0;j<6;j++) n[i]+=B[i,j]*u[j];
        }
        if(ja!=null) {Array.Copy(A,ja,A.Length);Array.Copy(B,jb,B.Length);}
        return n;
    }
    public double[] StepWorld(double[] z,double[] worldCommand,double[,] ja=null,double[,] jb=null) =>
        Unified?throw new InvalidOperationException("DE51 experiments require explicit body-command API StepBody"):Residual!=null?Residual.StepWorld(z,worldCommand,ja,jb):BodyVelocity?BodyStep(z,worldCommand,ja,jb):Step(z,worldCommand,ja,jb);
    public double[] StepBody(double[] z,double[] bodyCommand,double[,] ja=null,double[,] jb=null)
    {
        if(!string.IsNullOrEmpty(Data.de83Mode))return De83Padded(z,bodyCommand,ja,jb);
        if(Unified)return UnifiedBodyStep(z,bodyCommand,ja,jb);
        if(Residual!=null)return Residual.Step(z,bodyCommand,ja,jb);
        if(!BodyVelocity)throw new InvalidOperationException("StepBody requires body-frame model");
        return BodyStep(z,bodyCommand,ja,jb,true);
    }
    double[] BodyStep(double[] z,double[] worldCommand,double[,] ja,double[,] jb,bool directBody=false)
    {
        if(Data.kinematics=="unity_zxy_semiimplicit")
            return BodyStepSemiImplicit(z,worldCommand,ja,jb,directBody);
        if(Data.kinematics=="unity_zxy_midpoint_rk2" || UsesStartPoseAverage)
            return BodyStepMidpoint(z,worldCommand,ja,jb,directBody);
        throw new InvalidOperationException("Unsupported body pose integrator: "+Data.kinematics);
    }
    double[] BodyStepSemiImplicit(double[] z,double[] worldCommand,double[,] ja,double[,] jb,bool directBody)
    {
        var p=FinsRovBodyFrame.Pose(z);var r=FinsRovBodyFrame.Rotation(p);
        var mapping=new double[6,6];var ub=new double[6];
        var scale=FinsRovLiftedOperator.WrenchScale;
        for(int block=0;block<6;block+=3)for(int i=0;i<3;i++)for(int j=0;j<3;j++) {
            mapping[block+i,block+j]=directBody?(i==j?1d:0d):r[j,i]*scale[block+j]/scale[block+i];
            ub[block+i]+=mapping[block+i,block+j]*worldCommand[block+j];
        }
        var next=new double[Size];
        for(int i=0;i<Size;i++) {
            for(int j=0;j<Size;j++)next[i]+=A[i,j]*z[j];
            for(int j=0;j<6;j++)next[i]+=B[i,j]*ub[j];
        }
        var k=FinsRovBodyFrame.RateMatrix(p);
        for(int i=0;i<6;i++) {
            double rate=0;for(int j=0;j<6;j++)rate+=k[i,j]*FinsRovLiftedOperator.DecodeVelocity(next,j);
            next[1+i]=z[1+i]+.1*rate/FinsRovLiftedOperator.PoseScale[i];
        }
        if(ja==null)return next;
        Array.Copy(A,ja,A.Length);
        for(int i=0;i<Size;i++)for(int j=0;j<6;j++)for(int l=0;l<6;l++)jb[i,j]+=B[i,l]*mapping[l,j];
        for(int axis=3;axis<6 && !directBody;axis++) {
            double old=p[axis],h=1e-6;
            p[axis]=old+h;var plus=FinsRovBodyFrame.Rotation(p);
            p[axis]=old-h;var minus=FinsRovBodyFrame.Rotation(p);p[axis]=old;
            for(int i=0;i<Size;i++)for(int block=0;block<6;block+=3)for(int j=0;j<3;j++)for(int l=0;l<3;l++)
                ja[i,1+axis]+=B[i,block+j]*(plus[l,j]-minus[l,j])/(2*h)*scale[block+l]/scale[block+j]*worldCommand[block+l]*FinsRovLiftedOperator.PoseScale[axis];
        }
        var dynamicA=(double[,])ja.Clone();var dynamicB=(double[,])jb.Clone();
        for(int i=0;i<6;i++) {
            for(int j=0;j<Size;j++) {
                ja[1+i,j]=j==1+i?1d:0d;
                for(int l=0;l<6;l++)ja[1+i,j]+=.1*k[i,l]*FinsRovLiftedOperator.VelocityScale[l]/FinsRovLiftedOperator.PoseScale[i]*dynamicA[7+l,j];
            }
            for(int j=0;j<6;j++) {
                jb[1+i,j]=0;
                for(int l=0;l<6;l++)jb[1+i,j]+=.1*k[i,l]*FinsRovLiftedOperator.VelocityScale[l]/FinsRovLiftedOperator.PoseScale[i]*dynamicB[7+l,j];
            }
        }
        for(int axis=3;axis<6;axis++) {
            double old=p[axis],h=1e-6;
            p[axis]=old+h;var plus=FinsRovBodyFrame.RateMatrix(p);
            p[axis]=old-h;var minus=FinsRovBodyFrame.RateMatrix(p);p[axis]=old;
            for(int i=0;i<6;i++)for(int l=0;l<6;l++)ja[1+i,1+axis]+=.1*(plus[i,l]-minus[i,l])/(2*h)*FinsRovLiftedOperator.DecodeVelocity(next,l)*FinsRovLiftedOperator.PoseScale[axis]/FinsRovLiftedOperator.PoseScale[i];
        }
        return next;
    }
    double[] BodyStepMidpoint(double[] z,double[] worldCommand,double[,] ja,double[,] jb,bool directBody)
    {
        double dt=SampleDt;
        var p=FinsRovBodyFrame.Pose(z);var r=FinsRovBodyFrame.Rotation(p);
        var mapping=new double[6,6];var ub=new double[6];var scale=FinsRovLiftedOperator.WrenchScale;
        for(int block=0;block<6;block+=3)for(int i=0;i<3;i++)for(int j=0;j<3;j++) {
            mapping[block+i,block+j]=directBody?(i==j?1d:0d):r[j,i]*scale[block+j]/scale[block+i];
            ub[block+i]+=mapping[block+i,block+j]*worldCommand[block+j];
        }
        var next=new double[Size];
        for(int i=0;i<Size;i++) {
            for(int j=0;j<Size;j++)next[i]+=A[i,j]*z[j];
            for(int j=0;j<6;j++)next[i]+=B[i,j]*ub[j];
        }
        var velocity0=new double[6];var velocity1=new double[6];
        for(int i=0;i<6;i++){velocity0[i]=FinsRovLiftedOperator.DecodeVelocity(z,i);velocity1[i]=FinsRovLiftedOperator.DecodeVelocity(next,i);}
        if(!LearnsPose) {
            var nextPose=UsesStartPoseAverage?FinsRovBodyFrame.StartPoseAverage(p,velocity0,velocity1,dt):FinsRovBodyFrame.MidpointPose(p,velocity0,velocity1,dt);
            for(int i=0;i<6;i++)next[1+i]=(nextPose[i]-FinsRovLiftedOperator.PoseCenter[i])/FinsRovLiftedOperator.PoseScale[i];
        }
        if(ja==null)return next;
        Array.Copy(A,ja,A.Length);
        for(int i=0;i<Size;i++)for(int j=0;j<6;j++)for(int l=0;l<6;l++)jb[i,j]+=B[i,l]*mapping[l,j];
        for(int axis=3;axis<6 && !directBody;axis++) {
            double old=p[axis],h=1e-6;
            p[axis]=old+h;var plus=FinsRovBodyFrame.Rotation(p);
            p[axis]=old-h;var minus=FinsRovBodyFrame.Rotation(p);p[axis]=old;
            for(int i=0;i<Size;i++)for(int block=0;block<6;block+=3)for(int j=0;j<3;j++)for(int l=0;l<3;l++)
                ja[i,1+axis]+=B[i,block+j]*(plus[l,j]-minus[l,j])/(2*h)*scale[block+l]/scale[block+j]*worldCommand[block+l]*FinsRovLiftedOperator.PoseScale[axis];
        }
        if(LearnsPose)return next;
        var dynamicA=(double[,])ja.Clone();var dynamicB=(double[,])jb.Clone();
        double[,] poseJacobian,velocity0Jacobian,velocity1Jacobian;
        if(UsesStartPoseAverage)FinsRovBodyFrame.StartPoseAverageJacobians(p,velocity0,velocity1,dt,
            out poseJacobian,out velocity0Jacobian,out velocity1Jacobian);
        else FinsRovBodyFrame.MidpointJacobians(p,velocity0,velocity1,dt,
            out poseJacobian,out velocity0Jacobian,out velocity1Jacobian);
        for(int i=0;i<6;i++) {
            for(int j=0;j<Size;j++) {
                double derivative=0d;
                for(int l=0;l<6;l++)derivative+=velocity1Jacobian[i,l]*FinsRovLiftedOperator.VelocityScale[l]*dynamicA[7+l,j];
                if(j>=7&&j<13)derivative+=velocity0Jacobian[i,j-7]*FinsRovLiftedOperator.VelocityScale[j-7];
                if(j>=1&&j<7)derivative+=poseJacobian[i,j-1]*FinsRovLiftedOperator.PoseScale[j-1];
                ja[1+i,j]=derivative/FinsRovLiftedOperator.PoseScale[i];
            }
            for(int j=0;j<6;j++) {
                double derivative=0d;
                for(int l=0;l<6;l++)derivative+=velocity1Jacobian[i,l]*FinsRovLiftedOperator.VelocityScale[l]*dynamicB[7+l,j];
                jb[1+i,j]=derivative/FinsRovLiftedOperator.PoseScale[i];
            }
        }
        return next;
    }
}

// Exact learned16 plus external physical12; compact propagation buffer29.
public sealed partial class FinsRovHybridOperator {
 public static double[] De69Pack(double[] z){var q=new double[z.Length-23];System.Array.Copy(z,0,q,0,13);System.Array.Copy(z,36,q,13,z.Length-36);return q;}
 public double[] De69Step(double[] z,double[] u,double[,] ja=null,double[,] jb=null)=>string.IsNullOrEmpty(Data.de83Mode)?De69LegacyStep(z,u,ja,jb):De83Step(z,u,ja,jb);
 public double[] De69LegacyStep(double[] z,double[] u,double[,] ja=null,double[,] jb=null){
  int latentCount=Data.latent,n=13+latentCount;var result=new double[n];result[0]=1;
  for(int i=0;i<latentCount;i++){
   for(int j=0;j<latentCount;j++)result[13+i]+=A[36+i,36+j]*z[13+j];
   for(int j=0;j<6;j++)result[13+i]+=B[36+i,j]*u[j];
  }
  for(int i=0;i<6;i++){
   for(int j=0;j<latentCount;j++)result[7+i]+=A[7+i,36+j]*z[13+j];
   for(int j=0;j<6;j++)result[7+i]+=B[7+i,j]*u[j];
  }
  var na=ja==null?null:new double[n,n];var nb=ja==null?null:new double[n,6];var physical=UnifiedFossen.Step(z,u,na,nb);
  for(int i=0;i<6;i++)result[7+i]+=physical[7+i];
  if(ja!=null){
   System.Array.Clear(ja,0,ja.Length);System.Array.Clear(jb,0,jb.Length);ja[0,0]=1;
   for(int i=0;i<latentCount;i++){for(int j=0;j<latentCount;j++)ja[13+i,13+j]=A[36+i,36+j];for(int j=0;j<6;j++)jb[13+i,j]=B[36+i,j];}
   for(int i=0;i<6;i++){
    for(int j=0;j<13;j++)ja[7+i,j]=na[7+i,j];
    for(int j=0;j<latentCount;j++)ja[7+i,13+j]=A[7+i,36+j];
    for(int j=0;j<6;j++)jb[7+i,j]=B[7+i,j]+nb[7+i,j];
   }
  }
  var ps=FinsRovLiftedOperator.PoseScale;var vs=FinsRovLiftedOperator.VelocityScale;var p=FinsRovBodyFrame.Pose(z);var v0=new double[6];var v1=new double[6];
  for(int i=0;i<6;i++){v0[i]=z[7+i]*vs[i];v1[i]=result[7+i]*vs[i];}
  var pose=FinsRovBodyFrame.StartPoseAverage(p,v0,v1,.1);for(int i=0;i<6;i++)result[1+i]=(pose[i]-FinsRovLiftedOperator.PoseCenter[i])/ps[i];
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
// DE83 shared compact kernel. Physical12+latent10, no extra dynamic state.
public sealed partial class FinsRovHybridOperator {
 public double[] De83VB=new double[6],De83PB=new double[6];
 public static double[,] De83Mul(double[,] a,double[,] b){var c=new double[3,3];for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++)c[i,j]+=a[i,k]*b[k,j];return c;}
 public static double[,] De83Exp(double[] v){
  double a2=0;foreach(double x in v)a2+=x*x;double a=System.Math.Sqrt(System.Math.Max(a2,1e-12));
  double s=a2<1e-8?1-a2/6+a2*a2/120:System.Math.Sin(a)/a,c=a2<1e-8?.5-a2/24+a2*a2/720:(1-System.Math.Cos(a))/(a*a);
  var k=new double[,]{{0,-v[2],v[1]},{v[2],0,-v[0]},{-v[1],v[0],0}};var kk=De83Mul(k,k);var r=new double[3,3];
  for(int i=0;i<3;i++)for(int j=0;j<3;j++)r[i,j]=(i==j?1d:0d)+s*k[i,j]+c*kk[i,j];return r;
 }
 public static double[] De83Log(double[,] r){double co=System.Math.Max(-1,System.Math.Min(1,(r[0,0]+r[1,1]+r[2,2]-1)/2));double t=System.Math.Acos(co),f=t<1e-5?.5+t*t/12:t/(2*System.Math.Max(System.Math.Sin(t),1e-12));return new[]{(r[2,1]-r[1,2])*f,(r[0,2]-r[2,0])*f,(r[1,0]-r[0,1])*f};}
 public static double[] De83RotationError(double[] p,double[] q){var a=FinsRovBodyFrame.Rotation(p);var b=FinsRovBodyFrame.Rotation(q);var r=new double[3,3];for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++)r[i,j]+=a[k,i]*b[k,j];return De83Log(r);}
 static double[] De83Euler(double[,] r)=>new[]{System.Math.Asin(System.Math.Max(-1,System.Math.Min(1,-r[1,2]))),System.Math.Atan2(r[0,2],r[2,2]),System.Math.Atan2(r[1,0],r[1,1])};
 double[] De83Pose(double[] p,double[] pf,double[] correction){
  var result=(double[])pf.Clone();var r0=FinsRovBodyFrame.Rotation(p);var r1=FinsRovBodyFrame.Rotation(pf);
  for(int i=0;i<3;i++){for(int j=0;j<3;j++)result[i]-=(r1[i,j]-r0[i,j])*Data.de83Center[j];result[i]+=correction[i]+Data.de83PoseBiasScale[i]*De83PB[i];}
  var rv=new double[3];var rb=new double[3];for(int i=0;i<3;i++){rv[i]=correction[3+i];rb[i]=Data.de83PoseBiasScale[3+i]*De83PB[3+i];}
  var angles=De83Euler(De83Mul(De83Mul(r1,De83Exp(rv)),De83Exp(rb)));for(int i=0;i<3;i++)result[3+i]=angles[i];return result;
 }
 public double[] De83Step(double[] z,double[] u,double[,] ja=null,double[,] jb=null){
 if(Data.de99Arm=="Kx")return De99Kx(z,u,ja,jb);
  const int d=10,n=23;var result=new double[n];result[0]=1;bool dual=Data.de83Mode=="B";
  var na=ja==null?null:new double[n,n];var nb=ja==null?null:new double[n,6];var physical=Data.de99Arm=="Knu"?new double[n]:UnifiedFossen.Step(z,u,na,nb);
  for(int i=0;i<d;i++){for(int j=0;j<d;j++)result[13+i]+=A[36+i,36+j]*z[13+j];for(int j=0;j<6;j++)result[13+i]+=B[36+i,j]*u[j];}
  for(int i=0;i<6;i++){result[7+i]=physical[7+i]+De83VB[i]*z[0];for(int j=0;j<d;j++)result[7+i]+=A[7+i,36+j]*z[13+j];for(int j=0;j<6;j++)result[7+i]+=B[7+i,j]*u[j];}
  if(ja!=null){
   System.Array.Clear(ja,0,ja.Length);System.Array.Clear(jb,0,jb.Length);ja[0,0]=1;
   for(int i=0;i<d;i++){for(int j=0;j<d;j++)ja[13+i,13+j]=A[36+i,36+j];for(int j=0;j<6;j++)jb[13+i,j]=B[36+i,j];}
   for(int i=0;i<6;i++){for(int j=0;j<13;j++)ja[7+i,j]=na[7+i,j];ja[7+i,0]+=De83VB[i];for(int j=0;j<d;j++)ja[7+i,13+j]=A[7+i,36+j];for(int j=0;j<6;j++)jb[7+i,j]=nb[7+i,j]+B[7+i,j];}
  }
  var ps=FinsRovLiftedOperator.PoseScale;var vs=FinsRovLiftedOperator.VelocityScale;var p=FinsRovBodyFrame.Pose(z);var v0=new double[6];var v1=new double[6];
  for(int i=0;i<6;i++){v0[i]=z[7+i]*vs[i];v1[i]=(Data.de83Mode=="A"?result[7+i]:physical[7+i])*vs[i];}
  var pf=FinsRovBodyFrame.StartPoseAverage(p,v0,v1,.1);var corr=new double[6];
  if(dual)for(int i=0;i<6;i++){for(int j=0;j<d;j++)corr[i]+=A[1+i,36+j]*z[13+j];for(int j=0;j<6;j++)corr[i]+=B[1+i,j]*u[j];corr[i]*=ps[i];}
  var pose=De83Pose(p,pf,corr);for(int i=0;i<6;i++)result[1+i]=(pose[i]-FinsRovLiftedOperator.PoseCenter[i])/ps[i];
  if(ja==null)return result;
  double[,] jp,j0,j1;FinsRovBodyFrame.StartPoseAverageJacobians(p,v0,v1,.1,out jp,out j0,out j1);
  var pa=new double[6,n];var pb=new double[6,6];var va=Data.de83Mode=="A"?ja:na;var vb=Data.de83Mode=="A"?jb:nb;
  for(int i=0;i<6;i++){
   for(int j=0;j<n;j++){double val=j>=1&&j<7?jp[i,j-1]*ps[j-1]:j>=7&&j<13?j0[i,j-7]*vs[j-7]:0;for(int l=0;l<6;l++)val+=j1[i,l]*vs[l]*va[7+l,j];pa[i,j]=val;}
   for(int j=0;j<6;j++)for(int l=0;l<6;l++)pb[i,j]+=j1[i,l]*vs[l]*vb[7+l,j];
  }
  // Only tiny geometry partials use central differences; expensive Fossen
  // derivatives are the existing analytic RK4 sensitivities, evaluated once.
  var jo=new double[6,6];var jf=new double[6,6];var jc=new double[6,6];for(int i=0;i<3;i++){jf[i,i]=1;jc[i,i]=1;}
  const double eps=1e-6;
  for(int block=0;block<3;block++)for(int k=3;k<6;k++){
   var target=block==0?p:block==1?pf:corr;double old=target[k];target[k]=old+eps;var plus=De83Pose(p,pf,corr);target[k]=old-eps;var minus=De83Pose(p,pf,corr);target[k]=old;
   var dest=block==0?jo:block==1?jf:jc;for(int i=0;i<6;i++){double diff=plus[i]-minus[i];if(i>=3)diff=System.Math.Atan2(System.Math.Sin(diff),System.Math.Cos(diff));dest[i,k]=diff/(2*eps);}
  }
  for(int i=0;i<6;i++){
   for(int j=0;j<n;j++){double val=0;for(int l=0;l<6;l++){val+=jf[i,l]*pa[l,j];if(j==1+l)val+=jo[i,l]*ps[l];if(dual && j>=13)val+=jc[i,l]*ps[l]*A[1+l,36+j-13];}ja[1+i,j]=val/ps[i];}
   for(int j=0;j<6;j++){double val=0;for(int l=0;l<6;l++){val+=jf[i,l]*pb[l,j];if(dual)val+=jc[i,l]*ps[l]*B[1+l,j];}jb[1+i,j]=val/ps[i];}
  }
  return result;
 }
 public double[] De83Padded(double[] z,double[] u,double[,] ja,double[,] jb){
  var ca=ja==null?null:new double[23,23];var cb=ja==null?null:new double[23,6];var q=De83Step(De69Pack(z),u,ca,cb);var result=new double[46];System.Array.Copy(q,0,result,0,13);System.Array.Copy(q,13,result,36,10);
  if(ja!=null){System.Array.Clear(ja,0,ja.Length);System.Array.Clear(jb,0,jb.Length);for(int i=0;i<23;i++){int oi=i<13?i:i+23;for(int j=0;j<23;j++)ja[oi,j<13?j:j+23]=ca[i,j];for(int j=0;j<6;j++)jb[oi,j]=cb[i,j];}}
  return result;
 }
}

public sealed partial class FinsRovHybridOperator {
 double[] De99KxValue(double[] z,double[] u){
  var q=new double[23];q[0]=1;
  for(int i=0;i<10;i++){for(int j=0;j<10;j++)q[13+i]+=A[36+i,36+j]*z[13+j];for(int j=0;j<6;j++)q[13+i]+=B[36+i,j]*u[j];}
  for(int i=0;i<6;i++){for(int j=0;j<10;j++)q[7+i]+=A[7+i,36+j]*z[13+j];for(int j=0;j<6;j++)q[7+i]+=B[7+i,j]*u[j];}
  var p=FinsRovBodyFrame.Pose(z);var d=new double[6];
  for(int i=0;i<6;i++){for(int j=0;j<10;j++)d[i]+=A[1+i,36+j]*z[13+j];for(int j=0;j<6;j++)d[i]+=B[1+i,j]*u[j];d[i]*=Data.de99DeltaScale[i];}
  var rot=De83Mul(FinsRovBodyFrame.Rotation(p),De83Exp(new[]{d[3],d[4],d[5]}));var angles=De83Euler(rot);
  for(int i=0;i<3;i++){q[1+i]=(p[i]+d[i]-FinsRovLiftedOperator.PoseCenter[i])/FinsRovLiftedOperator.PoseScale[i];q[4+i]=angles[i]/FinsRovLiftedOperator.PoseScale[3+i];}
  return q;
 }
 double[] De99Kx(double[] z,double[] u,double[,] ja,double[,] jb){
  var q=De99KxValue(z,u);if(ja==null)return q;
  System.Array.Clear(ja,0,ja.Length);System.Array.Clear(jb,0,jb.Length);ja[0,0]=1;
  for(int i=0;i<10;i++){for(int j=0;j<10;j++)ja[13+i,13+j]=A[36+i,36+j];for(int j=0;j<6;j++)jb[13+i,j]=B[36+i,j];}
  for(int i=0;i<6;i++){for(int j=0;j<10;j++)ja[7+i,13+j]=A[7+i,36+j];for(int j=0;j<6;j++)jb[7+i,j]=B[7+i,j];}
  De100KxPoseJacobian(z,u,ja,jb);return q;
 }
}

// DE100: analytic derivatives of unchanged Ry Rx Rz and Rodrigues maps.
public sealed partial class FinsRovHybridOperator {
 public static double[,] De100RotationDerivative(double[] p,int axis){
  double a=p[3],b=p[4],c=p[5],ca=System.Math.Cos(a),sa=System.Math.Sin(a),cb=System.Math.Cos(b),sb=System.Math.Sin(b),cc=System.Math.Cos(c),sc=System.Math.Sin(c);
  if(axis==3)return new double[,]{{sb*ca*sc,sb*ca*cc,-sb*sa},{-sa*sc,-sa*cc,-ca},{cb*ca*sc,cb*ca*cc,-cb*sa}};
  if(axis==4)return new double[,]{{-sb*cc+cb*sa*sc,sb*sc+cb*sa*cc,cb*ca},{0,0,0},{-cb*cc-sb*sa*sc,cb*sc-sb*sa*cc,-sb*ca}};
  if(axis==5)return new double[,]{{-cb*sc+sb*sa*cc,-cb*cc-sb*sa*sc,0},{ca*cc,-ca*sc,0},{sb*sc+cb*sa*cc,sb*cc-cb*sa*sc,0}};
  return new double[3,3];
 }
 public static double[,] De100ExpDerivative(double[] v,int axis){
  double a2=0;foreach(double x in v)a2+=x*x;double a=System.Math.Sqrt(System.Math.Max(a2,1e-12));
  double s=a2<1e-8?1-a2/6+a2*a2/120:System.Math.Sin(a)/a,c=a2<1e-8?.5-a2/24+a2*a2/720:(1-System.Math.Cos(a))/a2;
  double ds=a2<1e-8?v[axis]*(-1d/3+a2/30):v[axis]*(a*System.Math.Cos(a)-System.Math.Sin(a))/(a2*a);
  double dc=a2<1e-8?v[axis]*(-1d/12+a2/180):v[axis]*(a*System.Math.Sin(a)-2*(1-System.Math.Cos(a)))/(a2*a2);
  var k=new double[,]{{0,-v[2],v[1]},{v[2],0,-v[0]},{-v[1],v[0],0}};
  var e=new double[3];e[axis]=1;var dk=new double[,]{{0,-e[2],e[1]},{e[2],0,-e[0]},{-e[1],e[0],0}};
  var kk=De83Mul(k,k);var left=De83Mul(dk,k);var right=De83Mul(k,dk);var result=new double[3,3];
  for(int i=0;i<3;i++)for(int j=0;j<3;j++)result[i,j]=ds*k[i,j]+s*dk[i,j]+dc*kk[i,j]+c*(left[i,j]+right[i,j]);return result;
 }
 public static double[] De100EulerDerivative(double[,] r,double[,] dr){
  double denom=1-r[1,2]*r[1,2];
  if(denom<=0)throw new System.InvalidOperationException("DE100 Euler derivative at singular chart");
  return new[]{-dr[1,2]/System.Math.Sqrt(denom),(r[2,2]*dr[0,2]-r[0,2]*dr[2,2])/(r[0,2]*r[0,2]+r[2,2]*r[2,2]),(r[1,1]*dr[1,0]-r[1,0]*dr[1,1])/(r[1,0]*r[1,0]+r[1,1]*r[1,1])};
 }
 public static double[,] De100RateDerivative(double[] p,int axis){
  var result=new double[6,6];var dr=De100RotationDerivative(p,axis);
  for(int i=0;i<3;i++)for(int j=0;j<3;j++)result[i,j]=dr[i,j];
  double a=p[3],c=p[5],ca=System.Math.Cos(a),sc=System.Math.Sin(c),cc=System.Math.Cos(c);
  if(System.Math.Abs(ca)<.1)throw new System.InvalidOperationException("Body-frame Euler chart near singularity");
  if(axis==3){result[4,3]=sc*System.Math.Sin(a)/(ca*ca);result[4,4]=cc*System.Math.Sin(a)/(ca*ca);result[5,3]=sc/(ca*ca);result[5,4]=cc/(ca*ca);}
  if(axis==5){result[3,3]=-sc;result[3,4]=-cc;result[4,3]=cc/ca;result[4,4]=-sc/ca;result[5,3]=System.Math.Tan(a)*cc;result[5,4]=-System.Math.Tan(a)*sc;}
  return result;
 }
 void De100KxPoseJacobian(double[] z,double[] u,double[,] ja,double[,] jb){
  var p=FinsRovBodyFrame.Pose(z);var d=new double[3];
  for(int i=0;i<3;i++){for(int j=0;j<10;j++)d[i]+=A[4+i,36+j]*z[13+j];for(int j=0;j<6;j++)d[i]+=B[4+i,j]*u[j];d[i]*=Data.de99DeltaScale[3+i];}
  var r=FinsRovBodyFrame.Rotation(p);var e=De83Exp(d);var re=De83Mul(r,e);var local=new double[3,6];
  for(int j=0;j<6;j++){
   var dr=j<3?De83Mul(De100RotationDerivative(p,3+j),e):De83Mul(r,De100ExpDerivative(d,j-3));var v=De100EulerDerivative(re,dr);
   for(int i=0;i<3;i++)local[i,j]=v[i];
  }
  for(int i=0;i<3;i++){
   ja[1+i,1+i]=1;
   for(int j=0;j<10;j++)ja[1+i,13+j]=Data.de99DeltaScale[i]*A[1+i,36+j]/FinsRovLiftedOperator.PoseScale[i];
   for(int j=0;j<6;j++)jb[1+i,j]=Data.de99DeltaScale[i]*B[1+i,j]/FinsRovLiftedOperator.PoseScale[i];
   double scale=FinsRovLiftedOperator.PoseScale[3+i];
   for(int j=0;j<3;j++)ja[4+i,4+j]=local[i,j]*FinsRovLiftedOperator.PoseScale[3+j]/scale;
   for(int j=0;j<10;j++)for(int k=0;k<3;k++)ja[4+i,13+j]+=local[i,3+k]*Data.de99DeltaScale[3+k]*A[4+k,36+j]/scale;
   for(int j=0;j<6;j++)for(int k=0;k<3;k++)jb[4+i,j]+=local[i,3+k]*Data.de99DeltaScale[3+k]*B[4+k,j]/scale;
  }
 }
}
