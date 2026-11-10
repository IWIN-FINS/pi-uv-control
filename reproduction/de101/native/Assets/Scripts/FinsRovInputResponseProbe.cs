using System;
using System.Globalization;
using System.IO;
using UnityEngine;

// An opt-in experiment only: historical closed-loop paths are unchanged.
// Each +/- arm loads a fresh scene; the plant is never driven by its predictor.
public sealed partial class FinsRovHybridBasisSmoke
{
    [Serializable] public class ResponseProbe
    {
        public string id, state, shape, actuator;
        public int axis, seed;
        public bool directMotor,bodyAxis;
        public int motor;
        public float amplitude, noiseStd, hydroScale, delay, tau, slew;
        public float[] pose, velocity, previous, flowInitial;
    }
    [Serializable] public class ResponseProtocol { public ResponseProbe[] probes; }
    [Serializable] public class ResponseProgress { public int nextArm; }
    [Serializable] public class ResponseNominal { public float[] mass,linearDrag,quadraticDrag,spring,bias; }
    bool responseActive;
    ResponseProtocol responseProtocol;
    ResponseProbe responseProbe;
    int responseArm, responseTick;
    const int ResponseWarmTicks=30, ResponseRunTicks=60;
    string responseRoot;
    StreamWriter responseStateCsv, responseForceCsv;
    StreamWriter preflightActuatorCsv;
    float responseSign;

    bool TryStartInputResponse()
    {
        string root=Environment.GetEnvironmentVariable("FINSSIM_RESPONSE_ROOT");
        if(string.IsNullOrEmpty(root)) return false;
        responseRoot=Path.GetFullPath(root);
        if(Mathf.Abs(Time.fixedDeltaTime-.02f)>1e-6f)throw new Exception("Response requires 20 ms physics step");
        File.WriteAllText(Path.Combine(root,"nominal_parameters.json"),JsonUtility.ToJson(new ResponseNominal{
            mass=mass,linearDrag=linearDrag,quadraticDrag=quadraticDrag,spring=spring,bias=bias}));
        responseProtocol=JsonUtility.FromJson<ResponseProtocol>(File.ReadAllText(Path.Combine(root,"protocol.json")));
        string progress=Path.Combine(root,"progress.json");
        responseArm=File.Exists(progress)?JsonUtility.FromJson<ResponseProgress>(File.ReadAllText(progress)).nextArm:0;
        if(responseArm>=2*responseProtocol.probes.Length) {Application.Quit(0);phase=Phase.Done;return true;}
        responseProbe=responseProtocol.probes[responseArm/2];
        if(Math.Abs(responseProbe.hydroScale-hydroScale)>1e-6) {
            // The existing synthetic load reads this same scalar below.
            hydroScale=responseProbe.hydroScale;
        }
        responseSign=responseArm%2==0?1f:-1f;
        foreach(var motor in thrusters) {
            motor.DynamicsModel=FinsSim.Actuators.Thruster.ActuatorDynamicsModel.FirstOrderDelaySlew;
            motor.CommandDelaySec=responseProbe.delay;
            motor.ForceTimeConstantSec=responseProbe.tau;
            motor.MaxForceSlewRateNPerSec=responseProbe.slew;
            motor.ResetActuatorState();
        }
        string dir=Path.Combine(root,"truth",responseProbe.id,responseArm%2==0?"plus":"minus");
        if(Directory.Exists(dir)) throw new Exception("Refusing response overwrite: "+dir);
        Directory.CreateDirectory(dir);
        OpenCurrentArtifacts(dir,false);
        responseStateCsv=new StreamWriter(Path.Combine(dir,"states.csv"));
        responseStateCsv.WriteLine("step,t_s,p0,p1,p2,p3,p4,p5,v0,v1,v2,v3,v4,v5");
        responseForceCsv=new StreamWriter(Path.Combine(dir,"forces.csv"));
        responseForceCsv.WriteLine("tick,t_s,kind,f0,f1,f2,f3,f4,f5,saturated");
        if(Environment.GetEnvironmentVariable("FINSSIM_DUAL_PREFLIGHT")=="1") {
            preflightActuatorCsv=new StreamWriter(Path.Combine(dir,"actuator_audit.csv"));
            preflightActuatorCsv.Write("tick,fixed_time,frame_time");
            foreach(string field in new[]{"request","target","actual"})for(int i=0;i<8;i++)preflightActuatorCsv.Write(","+field+i);
            for(int a=0;a<6;a++)for(int i=0;i<8;i++)preflightActuatorCsv.Write(",basis"+a+"_"+i);
            preflightActuatorCsv.WriteLine();
        }
        noise=new System.Random(responseProbe.seed);
        responseTick=0; responseActive=true;
        phase=Phase.Identify; // no controller, no tracking, no training
        SetResponseState();
        gameObject.AddComponent<FinsRovHybridActuationObserver>().Owner=this;
        Debug.Log("FINSSIM_RESPONSE_ARM "+responseArm+" "+responseProbe.id+" sign="+responseSign);
        return true;
    }

    void SetResponseState()
    {
        body.isKinematic=true;body.isKinematic=false;
        body.position=new Vector3(responseProbe.pose[0],responseProbe.pose[1],responseProbe.pose[2]);
        body.rotation=Quaternion.Euler(responseProbe.pose[3]*Mathf.Rad2Deg,responseProbe.pose[4]*Mathf.Rad2Deg,responseProbe.pose[5]*Mathf.Rad2Deg);
        body.linearVelocity=new Vector3(responseProbe.velocity[0],responseProbe.velocity[1],responseProbe.velocity[2]);
        body.angularVelocity=new Vector3(responseProbe.velocity[3],responseProbe.velocity[4],responseProbe.velocity[5]);
        Physics.SyncTransforms();body.WakeUp();
    }

    void InputResponseFixedUpdate()
    {
        int tick=responseTick-ResponseWarmTicks;
        if(tick<=0) {
            // Clamp only during artificial actuator preparation. Release once
            // at t=0; all evaluated motion then uses actual Unity/Fossen physics.
            SetResponseState();
            if(tick==0) { hydro.ResetBackendState(); ResetCurrentProcess(responseProbe.seed); }
        }
        StepCurrentModel();
        ReadState(stateP,stateV);
        if(tick>=0 && tick%5==0) {
            var cells=new string[14];cells[0]=(tick/5).ToString();cells[1]=RF(tick*.02);
            for(int a=0;a<6;a++){cells[2+a]=RF(stateP[a]);cells[8+a]=RF(stateV[a]);}
            responseStateCsv.WriteLine(string.Join(",",cells));responseStateCsv.Flush();
        }
        if(tick==ResponseRunTicks) {
            responseStateCsv.Dispose();responseForceCsv.Dispose();preflightActuatorCsv?.Dispose();preflightActuatorCsv=null;
            string dir=Path.Combine(responseRoot,"truth",responseProbe.id,responseArm%2==0?"plus":"minus");
            File.WriteAllText(Path.Combine(dir,"completed.json"),"{\"arm\":"+responseArm+",\"state_samples\":13,\"physics_samples\":60}");
            File.WriteAllText(Path.Combine(responseRoot,"progress.json"),JsonUtility.ToJson(new ResponseProgress{nextArm=responseArm+1}));
            responseActive=false;phase=Phase.Done;
            StopCurrentModel();
            CloseCurrentArtifacts();
            foreach(var motor in thrusters){motor.ResetActuatorState();motor.ApplyForceRequest(0);}
            body.isKinematic=true;
            if(responseArm+1==2*responseProtocol.probes.Length) {
                File.WriteAllText(Path.Combine(responseRoot,"completed.json"),"{\"arms\":"+(responseArm+1)+"}");
                Debug.Log("FINSSIM_RESPONSE_COMPLETE arms="+(responseArm+1));Application.Quit(0);
            } else UnityEngine.SceneManagement.SceneManager.LoadScene(UnityEngine.SceneManagement.SceneManager.GetActiveScene().name);
            return;
        }
        Array.Copy(responseProbe.previous,wrench,6);
        if(tick>=0 && (responseProbe.shape=="held"||tick<10))
            wrench[responseProbe.axis]+=responseSign*responseProbe.amplitude;
        if(responseProbe.bodyAxis) {
            Vector3 f=body.rotation*new Vector3(wrench[0],wrench[1],wrench[2]);
            Vector3 torque=body.rotation*new Vector3(wrench[3],wrench[4],wrench[5]);
            wrench[0]=f.x;wrench[1]=f.y;wrench[2]=f.z;wrench[3]=torque.x;wrench[4]=torque.y;wrench[5]=torque.z;
        }
        if(tick>=0 && hydroScale!=0) {
            Vector3 v=body.linearVelocity,w=body.angularVelocity;
            float phi=stateP[3],theta=stateP[5],depth=stateP[1]+1.2f;
            Vector3 force,torque;
            if(hydroFamily==0) {
                force=new Vector3(-2.2f*Tanh(7*v.x+2*v.z)*Mathf.Abs(v.z)*5,
                    .65f*Mathf.Sin(5*depth)*Tanh(6*v.x),-1.8f*Tanh(6*v.z-2*v.x)*Mathf.Abs(v.x)*5);
                torque=new Vector3(-.18f*Tanh(8*w.x+3*v.z)+.10f*Mathf.Sin(4*theta)*Tanh(8*v.x),
                    -.12f*Tanh(7*w.y)*Mathf.Abs(v.x)*5,-.20f*Tanh(8*w.z-3*v.x)+.10f*Mathf.Sin(4*phi)*Tanh(8*v.z));
            } else if(hydroFamily==1) {
                Vector3 rel=v-new Vector3(.08f,0f,-.04f);
                force=new Vector3(-1.2f*rel.x*Mathf.Abs(rel.x)-.40f*rel.x*Mathf.Abs(rel.z),
                    -.30f*rel.y*Mathf.Abs(rel.y),-1.0f*rel.z*Mathf.Abs(rel.z)-.50f*rel.z*Mathf.Abs(rel.x));
                torque=new Vector3(-.10f*w.x-.04f*rel.z,-.08f*w.y*Mathf.Abs(rel.x),-.12f*w.z-.05f*rel.x);
            } else {
                force=new Vector3(
                    1.40f*Mathf.Sin(5f*depth+1.5f*v.y)*Tanh(7f*v.z)+.45f*Mathf.Sin(3f*phi)*Tanh(4f*v.x),
                    .80f*Mathf.Sin(4f*depth+2f*v.x)*Tanh(6f*v.z),
                    1.55f*Mathf.Sin(5f*theta+1.5f*v.x)*Tanh(6f*v.x)-.35f*Mathf.Sin(3f*phi)*Tanh(4f*v.z));
                torque=new Vector3(.16f*Mathf.Sin(4f*phi+2f*v.z)*Tanh(6f*w.x),
                    .13f*Mathf.Sin(3f*theta+v.x)*Tanh(6f*w.y),
                    .18f*Mathf.Sin(5f*phi+v.z)*Tanh(7f*w.z));
            }
            body.AddForce(hydroScale*force,ForceMode.Force);
            body.AddTorque(hydroScale*torque,ForceMode.Force);
        }
        Allocate(wrench,motorRequest);
        if(bodyStudy && !responseProbe.directMotor)for(int i=0;i<8;i++) {
            var curve=measuredThrusterCurves[i];
            if(curve==null)continue;
            double target=motorRequest[i],coefficient=target>=0?curve.positiveCoefficientNPerCommandSquared:curve.negativeCoefficientNPerCommandSquared;
            double decision=Math.Sign(target)*Math.Sqrt(Math.Abs(target)/(ThrustScale*coefficient));
            motorRequest[i]=(float)(Math.Max(-1,Math.Min(1,decision))*(decision>=0?nominalForwardForceN[i]:nominalReverseForceN[i]));
        }
        if(responseProbe.directMotor) {
            Array.Clear(motorRequest,0,8);
            if(tick>=0 && (responseProbe.shape=="held"||tick<10))motorRequest[responseProbe.motor]=responseSign*responseProbe.amplitude*(responseSign>0?nominalForwardForceN[responseProbe.motor]:nominalReverseForceN[responseProbe.motor]);
        }
        int saturated=0;
        for(int i=0;i<8;i++) {
            // Keep the paired action-response probe on the same actuator truth
            // as DE14: nominal force request -> measured static command/force
            // curve -> common per-thruster force noise. Historical probes with
            // no curve configured retain their original linear force mapping.
            float actuatorForce=measuredThrusterCurves[i]==null
                ? ThrustScale*motorRequest[i]
                : ThrustScale*MeasuredThrusterForce(i,motorRequest[i]);
            float force=actuatorForce+(tick>=0?responseProbe.noiseStd*Gaussian(noise):0);
            thrusters[i].ApplyForceRequest(force);
            if(Mathf.Abs(motorRequest[i])>= (motorRequest[i]>=0?thrusters[i].MaxForwardForceN:-thrusters[i].MaxReverseForceN)-.05f) saturated++;
        }
        lastSaturatedMotors=saturated;
        responseTick++;
    }

    static string RF(double x)=>x.ToString("R",CultureInfo.InvariantCulture);
    void ResponseForceRow(string kind,float[] values,int tick)
    {
        var cells=new string[9];cells[0]=tick.ToString();cells[1]=RF(tick*.02);cells[2]=kind;
        for(int a=0;a<6;a++)cells[3+a]=RF(values[a]);
        responseForceCsv.WriteLine(string.Join(",",cells)+","+lastSaturatedMotors);
    }
    void ObserveInputResponse()
    {
        int tick=responseTick-ResponseWarmTicks-1;
        if(tick<0||tick>=ResponseRunTicks) return;
        Vector3 f=Vector3.zero,t=Vector3.zero;
        foreach(var motor in thrusters){Vector3 part=motor.GetAppliedWorldForce();f+=part;t+=Vector3.Cross(motor.GetAppliedForcePosition()-body.worldCenterOfMass,part);}
        ResponseForceRow("requested",wrench,tick);
        ResponseForceRow("allocated",allocatedWrench,tick);
        ResponseForceRow("applied",new[]{f.x,f.y,f.z,t.x,t.y,t.z},tick);
        if(bodyStudy)ResponseForceRow("applied_body",FinsRovBodyFrame.ToBody(stateP,new[]{f.x,f.y,f.z,t.x,t.y,t.z}),tick);
        var estimate=new float[6];
        var inverse=Quaternion.Inverse(body.rotation);
        for(int i=0;i<8;i++) {
            float force=measuredThrusterCurves[i]==null?ThrustScale*motorRequest[i]:ThrustScale*MeasuredThrusterForce(i,motorRequest[i]);
            var d=inverse*thrusters[i].transform.TransformDirection(thrusters[i].LocalForceDirection.normalized);
            var offset=inverse*(thrusters[i].GetAppliedForcePosition()-body.worldCenterOfMass);
            var torque=Vector3.Cross(offset,d);
            for(int a=0;a<3;a++){estimate[a]+=force*d[a];estimate[a+3]+=force*torque[a];}
        }
        ResponseForceRow("estimated_body",estimate,tick);
        responseForceCsv.Flush();
        if(preflightActuatorCsv!=null) {
            var cells=new System.Collections.Generic.List<string>{tick.ToString(),RF(Time.fixedTimeAsDouble),RF(Time.timeAsDouble)};
            for(int i=0;i<8;i++)cells.Add(RF(motorRequest[i]));
            for(int i=0;i<8;i++)cells.Add(RF(thrusters[i].TargetForceRequest));
            for(int i=0;i<8;i++)cells.Add(RF(thrusters[i].LastForceRequest));
            var basis=GetAllocationBasis();for(int a=0;a<6;a++)for(int i=0;i<8;i++)cells.Add(RF(basis[a,i]));
            preflightActuatorCsv.WriteLine(string.Join(",",cells));preflightActuatorCsv.Flush();
        }
    }
}
