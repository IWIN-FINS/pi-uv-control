using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

// Opt-in DE21: the calibrated predictor and learned predictor share coordinates.
public sealed partial class FinsRovHybridBasisSmoke
{
    bool bodyStudy;
    bool bodyForceDecisions;
    bool actuatorStatePrediction;
    double actuatorPredictionTau;
    double actuatorPredictionDelay;
    double actuatorPredictionSlew;
    double[] actuatorModelApplied = new double[8];
    double[] actuatorModelCommanded = new double[8];
    readonly List<ActuatorModelRequest> actuatorModelPending = new List<ActuatorModelRequest>();

    sealed class ActuatorModelRequest
    {
        public double DueTime;
        public double[] Force;
        public double[,] Jacobian;
    }

    sealed class ActuatorModelState
    {
        public double[] Applied;
        public double[] Commanded;
        public List<ActuatorModelRequest> Pending;
    }

    void ConfigureActuatorStatePrediction(float tau, float delay, bool instant)
    {
        actuatorStatePrediction = Environment.GetEnvironmentVariable("FINSSIM_BODY_ACTUATOR_STATE_PREDICTION") == "1";
        if (!actuatorStatePrediction) return;
        if (!bodyStudy || !actuatorMpc || !bodyForceDecisions)
            throw new InvalidOperationException("Actuator-state prediction requires body study with direct eight-thruster force decisions");
        if (instant)
        {
            actuatorPredictionTau = actuatorPredictionDelay = actuatorPredictionSlew = 0d;
        }
        else if (tau > 0f || delay > 0f)
        {
            actuatorPredictionTau = Math.Max(0d, tau);
            actuatorPredictionDelay = Math.Max(0d, delay);
            actuatorPredictionSlew = 500d;
        }
        else if (thrusters[0].DynamicsModel == FinsSim.Actuators.Thruster.ActuatorDynamicsModel.FirstOrderDelaySlew)
        {
            actuatorPredictionTau = thrusters[0].ForceTimeConstantSec;
            actuatorPredictionDelay = thrusters[0].CommandDelaySec;
            actuatorPredictionSlew = thrusters[0].MaxForceSlewRateNPerSec;
        }
        else
        {
            actuatorPredictionTau = actuatorPredictionDelay = actuatorPredictionSlew = 0d;
        }
        ResetActuatorModelState();
    }

    void ResetActuatorModelState()
    {
        Array.Clear(actuatorModelApplied, 0, actuatorModelApplied.Length);
        Array.Clear(actuatorModelCommanded, 0, actuatorModelCommanded.Length);
        actuatorModelPending.Clear();
    }

    void AdvanceActuatorModel(double now, double[] target, double[] applied,
        double[] commanded, List<ActuatorModelRequest> pending, double[,] targetJacobian,
        double[,] appliedJacobian, double[,] commandedJacobian)
    {
        int decisionCount = targetJacobian == null ? 0 : targetJacobian.GetLength(1);
        if (actuatorPredictionDelay <= 1e-8d)
        {
            Array.Copy(target, commanded, 8);
            if (commandedJacobian != null)
                Array.Copy(targetJacobian, commandedJacobian, targetJacobian.Length);
        }
        else
        {
            pending.Add(new ActuatorModelRequest {
                DueTime = now + actuatorPredictionDelay,
                Force = (double[])target.Clone(),
                Jacobian = targetJacobian == null ? null : (double[,])targetJacobian.Clone()
            });
            while (pending.Count > 0 && pending[0].DueTime <= now + 1e-6d)
            {
                var due = pending[0]; pending.RemoveAt(0);
                Array.Copy(due.Force, commanded, 8);
                if (commandedJacobian != null)
                {
                    if (due.Jacobian == null) Array.Clear(commandedJacobian, 0, commandedJacobian.Length);
                    else Array.Copy(due.Jacobian, commandedJacobian, due.Jacobian.Length);
                }
            }
        }

        double alpha = actuatorPredictionTau <= 1e-10d
            ? 1d : 1d - Math.Exp(-Time.fixedDeltaTime / actuatorPredictionTau);
        double maxStep = actuatorPredictionSlew * Time.fixedDeltaTime;
        for (int i = 0; i < 8; i++)
        {
            double previous = applied[i];
            double filtered = previous + alpha * (commanded[i] - previous);
            double delta = filtered - previous;
            bool slewLimited = actuatorPredictionSlew > 0d && Math.Abs(delta) > maxStep;
            applied[i] = slewLimited ? previous + Math.Sign(delta) * maxStep : filtered;
            if (appliedJacobian == null) continue;
            for (int j = 0; j < decisionCount; j++)
                appliedJacobian[i, j] = slewLimited
                    ? appliedJacobian[i, j]
                    : (1d - alpha) * appliedJacobian[i, j] + alpha *
                      (commandedJacobian == null ? 0d : commandedJacobian[i, j]);
        }
    }

    void UpdateActuatorModelEstimate()
    {
        var target = new double[8];
        for (int i = 0; i < 8; i++)
            target[i] = ThrustScale * MeasuredThrusterForce(i, motorRequest[i]);
        AdvanceActuatorModel(Time.timeAsDouble, target, actuatorModelApplied,
            actuatorModelCommanded, actuatorModelPending, null, null, null);
    }

    double ActuatorModelForceEstimate(int motor) => actuatorModelApplied[motor];

    void BuildActuatorStatePrediction(double[,] sequence, double[,] predictedWrench,
        double[,,] predictedJacobian)
    {
        int decisionCount = Horizon * DecisionCount;
        var state = new ActuatorModelState {
            Applied = (double[])actuatorModelApplied.Clone(),
            Commanded = (double[])actuatorModelCommanded.Clone(),
            Pending = new List<ActuatorModelRequest>(actuatorModelPending.Count)
        };
        foreach (var item in actuatorModelPending)
            state.Pending.Add(new ActuatorModelRequest { DueTime = item.DueTime,
                Force = (double[])item.Force.Clone(), Jacobian = null });
        var forceJacobian = new double[8, decisionCount];
        var commandJacobian = new double[8, decisionCount];
        int substeps = Mathf.Max(1, Mathf.RoundToInt(ControlDt / Time.fixedDeltaTime));
        if (Math.Abs(substeps * Time.fixedDeltaTime - ControlDt) > 1e-5d)
            throw new InvalidOperationException("ControlDt must be an integer multiple of Unity fixedDeltaTime for actuator prediction");
        var sumForce = new double[8];
        var sumJacobian = new double[8, decisionCount];
        for (int k = 0; k < Horizon; k++)
        {
            Array.Clear(sumForce, 0, sumForce.Length);
            Array.Clear(sumJacobian, 0, sumJacobian.Length);
            for (int sub = 0; sub < substeps; sub++)
            {
                int variable = k * DecisionCount;
                var target = new double[8];
                var targetJacobian = new double[8, decisionCount];
                for (int i = 0; i < 8; i++)
                {
                    target[i] = DecisionForce(i, sequence[k, i]);
                    targetJacobian[i, variable + i] = ForceHalf(i);
                }
                double time = Time.timeAsDouble + (k * substeps + sub) * Time.fixedDeltaTime;
                AdvanceActuatorModel(time, target, state.Applied, state.Commanded,
                    state.Pending, targetJacobian, forceJacobian, commandJacobian);
                for (int i = 0; i < 8; i++)
                {
                    sumForce[i] += state.Applied[i] / substeps;
                    for (int j = 0; j < decisionCount; j++)
                        sumJacobian[i, j] += forceJacobian[i, j] / substeps;
                }
            }
            for (int a = 0; a < 6; a++)
            {
                double value = 0d;
                for (int i = 0; i < 8; i++) value += optimizationBasis[a, i] * sumForce[i];
                predictedWrench[k, a] = value / WrenchLimit[a];
                if (predictedJacobian == null) continue;
                for (int j = 0; j < decisionCount; j++)
                {
                    double derivative = 0d;
                    for (int i = 0; i < 8; i++)
                        derivative += optimizationBasis[a, i] * sumJacobian[i, j];
                    predictedJacobian[k, a, j] = derivative / WrenchLimit[a];
                }
            }
        }
    }
    double ForceCenter(int i)=>.5d*ThrustScale*(measuredThrusterCurves[i].positiveCoefficientNPerCommandSquared-measuredThrusterCurves[i].negativeCoefficientNPerCommandSquared);
    double ForceHalf(int i)=>.5d*ThrustScale*(measuredThrusterCurves[i].positiveCoefficientNPerCommandSquared+measuredThrusterCurves[i].negativeCoefficientNPerCommandSquared);
    double DecisionForce(int i,double x)=>ForceCenter(i)+ForceHalf(i)*x;
    double DecisionMotorRequest(int i,double x)
    {
        double force=DecisionForce(i,x),coefficient=force>=0?measuredThrusterCurves[i].positiveCoefficientNPerCommandSquared:measuredThrusterCurves[i].negativeCoefficientNPerCommandSquared;
        return Math.Sign(force)*Math.Sqrt(Math.Abs(force)/(ThrustScale*coefficient))*(force>=0?nominalForwardForceN[i]:nominalReverseForceN[i]);
    }
    string bodyTask="joint8";
    float bodyNoiseScale=1f;
    static bool BodyChartInvalid(double[] z)=>Math.Abs(FinsRovLiftedOperator.DecodePose(z,3))>=1.3d;
    FinsRovSqpSolver.Result SolveBodySqp(double[,] sequence,float t,bool learned,int budget)
    {
        // Validate before Gauss-Newton, which otherwise re-enters the invalid
        // Euler chart before the solver can report an infeasible initial point.
        De69Verify(t,sequence,learned);
        double cost=Rollout(t,sequence,learned,null);
        double[,] h=null;
        if(FinsRovSqpSolver.ValidCost(cost) && bodyForceDecisions && BodyFrameModel && !actuatorStatePrediction) {
            try { h=BodyGaussNewton(t,sequence,learned); }
            catch(InvalidOperationException ex) when(ex.Message=="Body-frame Euler chart near singularity") { h=null; }
        }
        if(Environment.GetEnvironmentVariable("FINSSIM_DE57_FAULT")=="qp" && t<.05f) {
            h=new double[sequence.Length,sequence.Length];for(int i=0;i<sequence.Length;i++)h[i,i]=-1;
        }
        return FinsRovSqpSolver.Solve(sequence,(u,g)=>Rollout(t,u,learned,g),budget,h,bodyForceDecisions);
    }
    double[,] BodyGaussNewton(float t,double[,] sequence,bool learned)
    {
        int n=De69Enabled(learned)?13+hybrid.Data.latent:ModelSize,m=Horizon*DecisionCount;var h=new double[m,m];var sensitivity=new double[n,m];var z=learned?LiftState(stateP,stateV,historyWrench):PureFossenLift(stateP,stateV);
        if(De69Enabled(learned))z=FinsRovHybridOperator.De69Pack(z);
        for(int k=0;k<Horizon;k++) {
            if(BodyChartInvalid(z))return null;
            var map=new double[6,DecisionCount];var u=MapDecision(sequence,k,map);var ja=new double[n,n];var jb=new double[n,6];
            var next=PredictionStep(z,u,learned,ja,jb);if(BodyChartInvalid(next))return null;
            var ds=new double[n,m];
            for(int i=0;i<n;i++)for(int j=0;j<m;j++)for(int l=0;l<n;l++)ds[i,j]+=ja[i,l]*sensitivity[l,j];
            for(int i=0;i<n;i++)for(int j=0;j<DecisionCount;j++)for(int a=0;a<6;a++)ds[i,k*DecisionCount+j]+=jb[i,a]*map[a,j];
            var rateJac=FinsRovBodyFrame.RatesJacobian(next);var dr=new double[6,m];
            for(int a=0;a<6;a++)for(int j=0;j<m;j++)for(int i=0;i<n;i++)dr[a,j]+=rateJac[a,i]*ds[i,j];
            double terminal=De92Weight(k);
            for(int i=0;i<m;i++)for(int j=0;j<=i;j++) {
                double value=0;
                for(int a=0;a<6;a++)value+=2d*terminal*(PositionWeight[a]*Math.Pow(FinsRovLiftedOperator.PoseScale[a],2)*ds[1+a,i]*ds[1+a,j]+VelocityWeight[a]*dr[a,i]*dr[a,j]);
                h[i,j]+=value;
            }
            for(int i=0;i<DecisionCount;i++)for(int j=0;j<=i;j++) {
                double value=i==j?.001d*Math.Pow(ForceHalf(i),2):0d;
                for(int a=0;a<6;a++)value+=2d*EffortWeight[a]*Math.Pow(WrenchLimit[a],2)*map[a,i]*map[a,j];
                h[k*DecisionCount+i,k*DecisionCount+j]+=value;
            }
            sensitivity=ds;z=next;
        }
        for(int i=0;i<m;i++){h[i,i]+=1e-8;for(int j=0;j<i;j++)h[j,i]=h[i,j];}
        return h;
    }
    float bodyInitialYaw=7f,bodyInitialTiltScale=1f;
    [Serializable] public class BodyEnvironment {public string key,value;}
    [Serializable] public class BodyJob {public string id;public BodyEnvironment[] environment;}
    [Serializable] public class BodyJobs {public BodyJob[] jobs;}
    [Serializable] public class BodyQueueProgress {public int nextJob;}
    bool TryAdvanceBodyQueue()
    {
        string path=Environment.GetEnvironmentVariable("FINSSIM_BODY_JOB_QUEUE");if(string.IsNullOrEmpty(path))return false;
        var plan=JsonUtility.FromJson<BodyJobs>(File.ReadAllText(path));string progress=path+".progress.json";
        int next=File.Exists(progress)?JsonUtility.FromJson<BodyQueueProgress>(File.ReadAllText(progress)).nextJob+1:1;
        File.WriteAllText(progress,JsonUtility.ToJson(new BodyQueueProgress{nextJob=next}));
        if(next>=plan.jobs.Length) {
            File.WriteAllText(path+".completed.json","{\"jobs\":"+next+"}");return false;
        }
        foreach(var setting in plan.jobs[next].environment)Environment.SetEnvironmentVariable(setting.key,setting.value);
        Debug.Log("FINSSIM_BODY_JOB "+next+" "+plan.jobs[next].id);
        UnityEngine.SceneManagement.SceneManager.LoadScene(UnityEngine.SceneManagement.SceneManager.GetActiveScene().name);
        return true;
    }
    void ConfigureBodyTruth()
    {
        float Read(string key,float fallback=0f)=>float.TryParse(Environment.GetEnvironmentVariable(key),NumberStyles.Float,CultureInfo.InvariantCulture,out float value)?value:fallback;
        bodyNoiseScale=Read("FINSSIM_BODY_NOISE_SCALE",1f);
        bodyForceDecisions=Environment.GetEnvironmentVariable("FINSSIM_BODY_FORCE_DECISIONS")=="1";
        if(bodyForceDecisions)Debug.Log("FINSSIM_ACTIVE_SET_QP_CHECK "+FinsRovSqpSolver.VerifyActiveSet());
        bodyInitialYaw=Read("FINSSIM_BODY_INITIAL_YAW",7f);bodyInitialTiltScale=Read("FINSSIM_BODY_INITIAL_TILT_SCALE",1f);
        float tau=Read("FINSSIM_BODY_ACTUATOR_TAU"),delay=Read("FINSSIM_BODY_ACTUATOR_DELAY");
        if(Environment.GetEnvironmentVariable("FINSSIM_BODY_ACTUATOR_INSTANT")=="1")foreach(var motor in thrusters) {
            motor.DynamicsModel=FinsSim.Actuators.Thruster.ActuatorDynamicsModel.None;
            motor.CommandDelaySec=0;motor.ForceTimeConstantSec=0;motor.ResetActuatorState();
        }
        if(bodyNoiseScale<0 || bodyNoiseScale>4 || tau<0 || tau>.5f || delay<0 || delay>.5f)throw new Exception("Body robustness settings out of domain");
        if(tau>0 || delay>0)foreach(var motor in thrusters) {
            motor.DynamicsModel=FinsSim.Actuators.Thruster.ActuatorDynamicsModel.FirstOrderDelaySlew;
            motor.CommandDelaySec=delay;motor.ForceTimeConstantSec=tau;motor.MaxForceSlewRateNPerSec=500f;motor.ResetActuatorState();
        }
        ConfigureActuatorStatePrediction(tau,delay,Environment.GetEnvironmentVariable("FINSSIM_BODY_ACTUATOR_INSTANT")=="1");
        float flow=Read("FINSSIM_BODY_CURRENT_X");
        if(flow!=0) {
            var provider=hydro.waterProviderBehaviour as FinsSim.Hydrodynamics.FlatWaterKinematicsProvider;
            if(provider==null)provider=FindFirstObjectByType<FinsSim.Hydrodynamics.FlatWaterKinematicsProvider>();
            if(provider==null)throw new Exception("Current robustness requires flat water provider");
            provider.currentVelocity=new Vector3(flow,0,.5f*flow);
        }
    }
    float BodyReference(int axis,float t,bool rate)
    {
        if(bodyTask=="hover" || bodyTask=="depth" || bodyTask=="attitude") {
            if(axis==1) {
                float depth=bodyTask=="depth" && t>2f?.08f*(1f-Mathf.Exp(-(t-2f)/.4f)):0f;
                return rate?(bodyTask=="depth"&&t>2f?.2f*Mathf.Exp(-(t-2f)/.4f):0f):-1.2f+depth;
            }
            if(bodyTask=="attitude" && axis>=3) {
                float amplitude=axis==3?.08f:axis==4?.15f:.07f,omega=axis==3?.4f:axis==4?.3f:.35f;
                return rate?amplitude*omega*Mathf.Cos(omega*t):amplitude*Mathf.Sin(omega*t);
            }
            return 0f;
        }
        if(bodyTask!="plane8" && bodyTask!="joint8")throw new Exception("Unknown body study task "+bodyTask);
        float o=2f*Mathf.PI/(EvaluationSeconds*periodScale);
        if(axis==0)return rate?.7f*amplitudeScale*o*Mathf.Cos(o*t):.7f*amplitudeScale*Mathf.Sin(o*t);
        if(axis==2)return rate?.7f*amplitudeScale*o*Mathf.Cos(2f*o*t):.35f*amplitudeScale*Mathf.Sin(2f*o*t);
        if(axis==1)return rate?(bodyTask=="joint8"?.08f*o*Mathf.Cos(o*t):0f):-1.2f+(bodyTask=="joint8"?.08f*Mathf.Sin(o*t):0f);
        float a=axis==4?.10f:bodyTask=="joint8"?(axis==3?.035f:.03f):0f;
        float frequency=axis==4?o:axis==3?.37f:.29f;
        return rate?a*frequency*Mathf.Cos(frequency*t):a*Mathf.Sin(frequency*t);
    }
    StreamWriter bodyInputCsv;
    StreamWriter bodyTimingCsv;
    struct BodyTimingTicket { public long Started; public float PhaseTime; public double ForceDue; }
    readonly Queue<BodyTimingTicket> bodyTimingPending = new Queue<BodyTimingTicket>();
    double bodyTimingLastSample = double.NegativeInfinity;
    void BeginBodyControlTiming()
    {
        if(!bodyStudy || !ViewerEvaluating || Environment.GetEnvironmentVariable("FINSSIM_BODY_TIMING")!="1" || SimTime+.0001d<nextControl)return;
        // FixedUpdate runs every physics tick; only create one ticket per new control sample.
        if(SimTime<=bodyTimingLastSample+.0001d)return;
        bodyTimingLastSample=SimTime;
        double delay=0;foreach(var motor in thrusters)delay=Math.Max(delay,motor.CommandDelaySec);
        bodyTimingPending.Enqueue(new BodyTimingTicket { Started=System.Diagnostics.Stopwatch.GetTimestamp(), PhaseTime=(float)(SimTime-phaseStart), ForceDue=SimTime+delay });
    }
    void FinishBodyControlTiming()
    {
        while(bodyTimingPending.Count>0 && SimTime+.0001d>=bodyTimingPending.Peek().ForceDue) {
            var ticket=bodyTimingPending.Dequeue();
            double elapsed=1000d*(System.Diagnostics.Stopwatch.GetTimestamp()-ticket.Started)/System.Diagnostics.Stopwatch.Frequency;
            if(bodyTimingCsv==null) {
                string path=Path.Combine(artifactDirectory,"body_control_timing.csv");bool append=resumeLearnedArm && File.Exists(path);
                bodyTimingCsv=new StreamWriter(path,append);
                if(!append)bodyTimingCsv.WriteLine("phase,t_s,observation_to_applied_force_ms");
            }
            bodyTimingCsv.WriteLine(string.Join(",",phase.ToString(),RF(ticket.PhaseTime),elapsed.ToString("R",CultureInfo.InvariantCulture)));bodyTimingCsv.Flush();
        }
    }
    readonly double[] estimatedBodySum=new double[6];
    int bodyInputSteps;
    float[,] OptimizationAllocationBasis()
    {
        var basis=GetAllocationBasis();if(!bodyStudy || !BodyFrameModel)return basis;
        var r=FinsRovBodyFrame.Rotation(Array.ConvertAll(stateP,x=>(double)x));var result=new float[6,8];
        for(int block=0;block<6;block+=3)for(int a=0;a<3;a++)for(int i=0;i<8;i++)for(int j=0;j<3;j++)result[block+a,i]+=(float)(r[j,a]*basis[block+j,i]);
        return result;
    }
    static double[] Cross(double[] a,double[] b)=>new[]{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
    void ResetBodyInputInterval(){Array.Clear(estimatedBodySum,0,6);bodyInputSteps=0;}
    void FinishBodyInputInterval()
    {
        for(int a=0;a<6;a++)historyWrench[a]=bodyInputSteps==0?0f:(float)(estimatedBodySum[a]/bodyInputSteps);
        ResetBodyInputInterval();
    }
    void ObserveBodyInput(float[] actualWorld)
    {
        ReadState(auditTruePose,auditTrueVelocity);
        var actual=FinsRovBodyFrame.ToBody(auditTruePose,actualWorld);
        var estimated=new double[6];
        // Known static motor map/geometry only: never read random realized thrust.
        var inverse=Quaternion.Inverse(body.rotation);
        for(int i=0;i<8;i++) {
            double f=actuatorStatePrediction?actuatorModelApplied[i]:
                measuredThrusterCurves[i]==null?ThrustScale*motorRequest[i]:ThrustScale*MeasuredThrusterForce(i,motorRequest[i]);
            var d=inverse*thrusters[i].transform.TransformDirection(thrusters[i].LocalForceDirection.normalized);
            var offset=inverse*(thrusters[i].GetAppliedForcePosition()-body.worldCenterOfMass);
            var torque=Vector3.Cross(offset,d);
            for(int a=0;a<3;a++){estimated[a]+=f*d[a];estimated[a+3]+=f*torque[a];}
        }
        ObserveRateHistory(estimated);
        if(ViewerEvaluating || phase==Phase.Identify) {
            for(int a=0;a<6;a++)estimatedBodySum[a]+=estimated[a];bodyInputSteps++;
        }
        if(bodyInputCsv==null) {
            string path=Path.Combine(artifactDirectory,"body_input_steps.csv");
            bool append=resumeLearnedArm && File.Exists(path);
            bodyInputCsv=new StreamWriter(path,append);
            if(!append)bodyInputCsv.WriteLine("phase,phase_time_s,fixed_time_s,segment,axis,true_body_input,estimated_body_input");
        }
        for(int a=0;a<6;a++)bodyInputCsv.WriteLine(string.Join(",",phase.ToString(),RF((float)(SimTime-phaseStart)),SimTime.ToString("R",CultureInfo.InvariantCulture),identifySegment,a,RF(actual[a]),estimated[a].ToString("R",CultureInfo.InvariantCulture)));
        bodyInputCsv.Flush();
    }

    void ResetBodyActuatorPrediction()
    {
        if(actuatorStatePrediction)ResetActuatorModelState();
    }
    double[] BodyNominalValue(double[] z,double[] worldU)
    {
        double dt=hybrid.Residual!=null?hybrid.Residual.Data.dt:ControlDt;
        var p=FinsRovBodyFrame.Pose(z);var r=FinsRovBodyFrame.Rotation(p);
        var v=new double[3];var w=new double[3];var momentum=new double[3];var angularMomentum=new double[3];
        for(int a=0;a<3;a++){v[a]=FinsRovLiftedOperator.DecodeVelocity(z,a);w[a]=FinsRovLiftedOperator.DecodeVelocity(z,a+3);momentum[a]=mass[a]*v[a];angularMomentum[a]=mass[a+3]*w[a];}
        var cv=Cross(w,momentum);var cw=Cross(v,momentum);var spin=Cross(w,angularMomentum);
        var u=new double[6];
        for(int i=0;i<6;i++)u[i]=worldU[i]*WrenchLimit[i];
        var next=new double[ModelSize];next[0]=1;
        for(int a=0;a<6;a++) {
            double velocity=a<3?v[a]:w[a-3];
            double restoring=a<3?bias[1]*r[1,a]:(a==3?spring[3]*r[1,2]:a==5?-spring[5]*r[1,0]:0d);
            double coriolis=a<3?cv[a]:cw[a-3]+spin[a-3];
            next[7+a]=(velocity+dt*(u[a]-linearDrag[a]*velocity-quadraticDrag[a]*velocity*Math.Abs(velocity)+restoring-coriolis)/mass[a])/FinsRovLiftedOperator.VelocityScale[a];
            next[13+a]=next[7+a]*Math.Abs(next[7+a]);
            if(ModelSize>=42 && !(HybridMode && hybrid.Data.schemaVersion==13))next[36+a]=u[a]/WrenchLimit[a];
        }
        var velocity0=new double[6];var velocity1=new double[6];
        for(int a=0;a<6;a++){velocity0[a]=FinsRovLiftedOperator.DecodeVelocity(z,a);velocity1[a]=FinsRovLiftedOperator.DecodeVelocity(next,a);}
        double[] nextPose;
        if(hybrid!=null&&hybrid.UsesStartPoseAverage)nextPose=FinsRovBodyFrame.StartPoseAverage(p,velocity0,velocity1,dt);
        else if(hybrid!=null&&hybrid.UsesMidpointKinematics)nextPose=FinsRovBodyFrame.MidpointPose(p,velocity0,velocity1,ControlDt);
        else {
            var rates=FinsRovBodyFrame.Rates(nextWithPose(next,z));nextPose=(double[])p.Clone();
            for(int a=0;a<6;a++)nextPose[a]+=ControlDt*rates[a];
        }
        for(int a=0;a<6;a++)next[1+a]=(nextPose[a]-FinsRovLiftedOperator.PoseCenter[a])/FinsRovLiftedOperator.PoseScale[a];
        return next;
    }
    static double[] nextWithPose(double[] next,double[] old){Array.Copy(old,1,next,1,6);return next;}
    static double[,] Skew(double[] v)=>new double[,]{{0,-v[2],v[1]},{v[2],0,-v[0]},{-v[1],v[0],0}};
    double[] BodyNominalStep(double[] z,double[] u,double[,] ja,double[,] jb)
    {
        if(hybrid!=null && hybrid.Unified)return hybrid.UnifiedFossen.Step(z,u,ja,jb);
        double dt=hybrid.Residual!=null?hybrid.Residual.Data.dt:ControlDt;
        var result=BodyNominalValue(z,u);if(ja==null)return result;
        var p=FinsRovBodyFrame.Pose(z);var v=new double[3];var w=new double[3];var momentum=new double[3];var angularMomentum=new double[3];
        for(int a=0;a<3;a++){v[a]=FinsRovLiftedOperator.DecodeVelocity(z,a);w[a]=FinsRovLiftedOperator.DecodeVelocity(z,a+3);momentum[a]=mass[a]*v[a];angularMomentum[a]=mass[a+3]*w[a];}
        var sv=Skew(v);var sw=Skew(w);var sm=Skew(momentum);var sj=Skew(angularMomentum);var dc=new double[6,6];
        for(int i=0;i<3;i++)for(int j=0;j<3;j++) {
            dc[i,j]=sw[i,j]*mass[j];dc[i,j+3]=-sm[i,j];
            dc[i+3,j]=sv[i,j]*mass[j]-sm[i,j];dc[i+3,j+3]=sw[i,j]*mass[j+3]-sj[i,j];
        }
        for(int i=0;i<6;i++) {
            double velocity=FinsRovLiftedOperator.DecodeVelocity(z,i);
            for(int j=0;j<6;j++) {
                double derivative=(i==j?1d-dt*(linearDrag[i]+2d*quadraticDrag[i]*Math.Abs(velocity))/mass[i]:0d)-dt*dc[i,j]/mass[i];
                ja[7+i,7+j]=derivative*FinsRovLiftedOperator.VelocityScale[j]/FinsRovLiftedOperator.VelocityScale[i];
            }
            jb[7+i,i]=dt*WrenchLimit[i]/(mass[i]*FinsRovLiftedOperator.VelocityScale[i]);
            if(ModelSize>=42 && !(HybridMode && hybrid.Data.schemaVersion==13))jb[36+i,i]=1d;
        }
        const double h=1e-6;
        for(int axis=3;axis<6;axis++) {
            double old=p[axis];p[axis]=old+h;var plus=FinsRovBodyFrame.Rotation(p);p[axis]=old-h;var minus=FinsRovBodyFrame.Rotation(p);p[axis]=old;
            for(int i=0;i<6;i++) {
                double dg=i<3?bias[1]*(plus[1,i]-minus[1,i]):i==3?spring[3]*(plus[1,2]-minus[1,2]):i==5?-spring[5]*(plus[1,0]-minus[1,0]):0d;
                ja[7+i,1+axis]=dt*dg/(2*h*mass[i])*FinsRovLiftedOperator.PoseScale[axis]/FinsRovLiftedOperator.VelocityScale[i];
            }
        }
        if(hybrid!=null&&(hybrid.UsesMidpointKinematics||hybrid.UsesStartPoseAverage)) {
            var velocity0=new double[6];var velocity1=new double[6];
            for(int i=0;i<6;i++){velocity0[i]=FinsRovLiftedOperator.DecodeVelocity(z,i);velocity1[i]=FinsRovLiftedOperator.DecodeVelocity(result,i);}
            double[,] poseJacobian,velocity0Jacobian,velocity1Jacobian;
            if(hybrid.UsesStartPoseAverage)FinsRovBodyFrame.StartPoseAverageJacobians(p,velocity0,velocity1,dt,
                out poseJacobian,out velocity0Jacobian,out velocity1Jacobian);
            else FinsRovBodyFrame.MidpointJacobians(p,velocity0,velocity1,ControlDt,
                out poseJacobian,out velocity0Jacobian,out velocity1Jacobian);
            for(int i=0;i<6;i++) {
                for(int j=0;j<ModelSize;j++) {
                    double derivative=0d;
                    for(int l=0;l<6;l++)derivative+=velocity1Jacobian[i,l]*FinsRovLiftedOperator.VelocityScale[l]*ja[7+l,j];
                    if(j>=7&&j<13)derivative+=velocity0Jacobian[i,j-7]*FinsRovLiftedOperator.VelocityScale[j-7];
                    if(j>=1&&j<7)derivative+=poseJacobian[i,j-1]*FinsRovLiftedOperator.PoseScale[j-1];
                    ja[1+i,j]=derivative/FinsRovLiftedOperator.PoseScale[i];
                }
                for(int j=0;j<6;j++) {
                    double derivative=0d;
                    for(int l=0;l<6;l++)derivative+=velocity1Jacobian[i,l]*FinsRovLiftedOperator.VelocityScale[l]*jb[7+l,j];
                    jb[1+i,j]=derivative/FinsRovLiftedOperator.PoseScale[i];
                }
            }
        } else {
            var k=FinsRovBodyFrame.RateMatrix(p);
            for(int i=0;i<6;i++) {
                for(int j=0;j<ModelSize;j++) {
                    ja[1+i,j]=j==1+i?1d:0d;
                    for(int l=0;l<6;l++)ja[1+i,j]+=ControlDt*k[i,l]*FinsRovLiftedOperator.VelocityScale[l]/FinsRovLiftedOperator.PoseScale[i]*ja[7+l,j];
                }
                for(int j=0;j<6;j++)for(int l=0;l<6;l++)jb[1+i,j]+=ControlDt*k[i,l]*FinsRovLiftedOperator.VelocityScale[l]/FinsRovLiftedOperator.PoseScale[i]*jb[7+l,j];
            }
            for(int axis=3;axis<6;axis++) {
                double old=p[axis];p[axis]=old+h;var plus=FinsRovBodyFrame.RateMatrix(p);p[axis]=old-h;var minus=FinsRovBodyFrame.RateMatrix(p);p[axis]=old;
                for(int i=0;i<6;i++)for(int l=0;l<6;l++)ja[1+i,1+axis]+=ControlDt*(plus[i,l]-minus[i,l])/(2*h)*FinsRovLiftedOperator.DecodeVelocity(result,l)*FinsRovLiftedOperator.PoseScale[axis]/FinsRovLiftedOperator.PoseScale[i];
            }
        }
        for(int i=0;i<6;i++) {
            double derivative=2d*Math.Abs(result[7+i]);
            for(int j=0;j<ModelSize;j++)ja[13+i,j]=derivative*ja[7+i,j];
            for(int j=0;j<6;j++)jb[13+i,j]=derivative*jb[7+i,j];
        }
        return result;
    }
}

