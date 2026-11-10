using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

// Opt-in matched-state solver diagnosis. Candidate controls never reach the plant.
public sealed partial class FinsRovHybridBasisSmoke
{
    [Serializable] public class PreflightSolve {
        public int budget; public double[] u,gradient;
        public double cost,milliseconds,projectedGradient; public string status;
    }
    [Serializable] public class PreflightObjective {
        public string phase; public int seed; public float t,dt;
        public double[] de83VelocityBias,de83PoseBias,de92Matrix,de92Weights;
        public bool learned; public int size;
        public bool bodyCoordinates,actuatorAware;
        public bool forceDecisions;
        public double[] forceCenter,forceHalf;
        public int decisions;
        public double[] allocationBasis,positiveCurve,negativeCurve,forwardScale,reverseScale;
        public double[] z,a,b,start,referencePose,referenceVelocity,startGradient;
        public double startCost;
        public float[] mass,linearDrag,quadraticDrag,spring,bias,poseCenter,poseScale,velocityScale,
            wrenchLimit,positionWeight,velocityWeight,effortWeight;
        public PreflightSolve[] solves;
    }
    readonly System.Collections.Generic.HashSet<string> preflightCaptured=new System.Collections.Generic.HashSet<string>();
    static double[] Flat(double[,] x) {var v=new double[x.Length];int c=x.GetLength(1);for(int i=0;i<v.Length;i++)v[i]=x[i/c,i%c];return v;}
    void CapturePreflightObjective(float t,bool learned)
    {
        if(Environment.GetEnvironmentVariable("FINSSIM_DUAL_PREFLIGHT")!="1")return;
        if((actuatorMpc && !bodyStudy) || !useSqp)throw new Exception("Preflight requires compatible SQP configuration");
        int target=(int)Math.Round(t/8f)*8;
        if(target>24 || Math.Abs(t-target)>.051f)return;
        string key=phase+"-"+target;
        if(!preflightCaptured.Add(key))return;
        // All diagnostic budgets start from the same zero sequence at the same
        // measured state, independently of the live warm start and chosen action.
        var start=new double[Horizon,DecisionCount];var g=new double[Horizon,DecisionCount];
        if(actuatorAwareMpc)InitializeActuatorAwareWarmStart(t,start);
        var record=new PreflightObjective {phase=phase.ToString(),seed=evaluationSeed,t=t,dt=ControlDt,learned=learned,size=ModelSize,
            z=learned?LiftState(stateP,stateV,historyWrench):PureFossenLift(stateP,stateV),a=Flat(ModelA),b=Flat(ModelB),start=Flat(start),
            mass=mass,linearDrag=linearDrag,quadraticDrag=quadraticDrag,spring=spring,bias=bias,
            poseCenter=FinsRovLiftedOperator.PoseCenter,poseScale=FinsRovLiftedOperator.PoseScale,
            velocityScale=FinsRovLiftedOperator.VelocityScale,wrenchLimit=WrenchLimit,
            positionWeight=PositionWeight,velocityWeight=VelocityWeight,effortWeight=EffortWeight,
            referencePose=new double[Horizon*6],referenceVelocity=new double[Horizon*6]};
        record.bodyCoordinates=bodyStudy && BodyFrameModel;record.actuatorAware=actuatorAwareMpc;
        record.decisions=DecisionCount;
        record.forceDecisions=bodyForceDecisions;
        if(actuatorMpc) {
            record.allocationBasis=new double[48];record.positiveCurve=new double[8];record.negativeCurve=new double[8];record.forwardScale=new double[8];record.reverseScale=new double[8];
            record.forceCenter=new double[8];record.forceHalf=new double[8];
            for(int i=0;i<8;i++) {
                record.positiveCurve[i]=ThrustScale*measuredThrusterCurves[i].positiveCoefficientNPerCommandSquared;
                record.negativeCurve[i]=ThrustScale*measuredThrusterCurves[i].negativeCoefficientNPerCommandSquared;
                record.forwardScale[i]=nominalForwardForceN[i];record.reverseScale[i]=nominalReverseForceN[i];
                record.forceCenter[i]=ForceCenter(i);record.forceHalf[i]=ForceHalf(i);
                for(int a=0;a<6;a++)record.allocationBasis[a*8+i]=optimizationBasis[a,i];
            }
        }
        for(int k=0;k<Horizon;k++)for(int j=0;j<6;j++) {
            float ft=t+ControlDt*(k+1);
            record.referencePose[k*6+j]=Reference(j,ft);record.referenceVelocity[k*6+j]=ReferenceVelocity(j,ft);
        }
        record.de92Matrix=(double[])de92C.Clone();record.de92Weights=new double[8];for(int j=0;j<8;j++)record.de92Weights[j]=De92Weight(j);record.de83VelocityBias=(double[])hybrid.De83VB.Clone();record.de83PoseBias=(double[])hybrid.De83PB.Clone();
        record.startCost=Rollout(t,start,learned,g);record.startGradient=Flat(g);
        var runs=new List<PreflightSolve>();
        foreach(int budget in new[]{32,128,512}) {
            var u=(double[,])start.Clone();var timer=System.Diagnostics.Stopwatch.StartNew();
            var r=SolveBodySqp(u,t,learned,budget);timer.Stop();
            var finalGradient=new double[Horizon,DecisionCount];double cost=Rollout(t,u,learned,finalGradient);
            runs.Add(new PreflightSolve {budget=budget,u=Flat(u),gradient=Flat(finalGradient),cost=cost,
                milliseconds=timer.Elapsed.TotalMilliseconds,projectedGradient=r.ProjectedGradient,status=r.Status});
        }
        record.solves=runs.ToArray();string folder=Path.Combine(artifactDirectory,"preflight_objectives");Directory.CreateDirectory(folder);
        File.WriteAllText(Path.Combine(folder,key+".json"),JsonUtility.ToJson(record));
    }
}
