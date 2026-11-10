using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

// Synthetic multirate estimator-output experiment. This is not a sensor EKF.
public sealed partial class FinsRovHybridBasisSmoke
{
    sealed class ObservationSample { public double time;public float[] p,v; }
    readonly List<ObservationSample> observationHistory=new List<ObservationSample>();
    readonly Vector3[] observationDirections=new Vector3[8],observationOffsets=new Vector3[8];
    float[] heldPosition,heldVelocity;
    double heldSourceTime=-1,nextPositionUpdate;
    Quaternion observationRotation;
    bool observationReady,observationChanged;
    string observationMode;
    System.Random observationNoise;
    StreamWriter observationCsv;
    StreamWriter observationNoiseAudit;
    int observationNoiseIndex;
    float AuditedObservationNoise()
    {
        float draw=Gaussian(observationNoise);
        observationNoiseAudit?.WriteLine(string.Join(",",phase.ToString(),observationNoiseIndex++,draw.ToString("R",CultureInfo.InvariantCulture)));
        return draw;
    }
    StreamWriter bodyObservationCsv;
    float[] previousBodyVelocity;
    float previousBodyTime;

    void InitializeObservationProbe()
    {
        bodyObservationCsv?.Dispose();bodyObservationCsv=null;
        observationMode=Environment.GetEnvironmentVariable("FINSSIM_OBSERVATION_CASE");
        if(string.IsNullOrEmpty(observationMode))return;
        if(observationMode!="clean"&&observationMode!="all100"&&observationMode!="multi100"&&observationMode!="multi100_cv"&&observationMode!="pose_outage")throw new Exception("Unknown observation case");
        observationCsv=new StreamWriter(Path.Combine(artifactDirectory,"control_observations.csv"),resumeLearnedArm);
        observationNoiseAudit=new StreamWriter(Path.Combine(artifactDirectory,"observation_noise_audit.csv"),resumeLearnedArm);
        if(!resumeLearnedArm)observationNoiseAudit.WriteLine("phase,index,standard_gaussian_draw");
        if(!resumeLearnedArm)observationCsv.WriteLine("phase,t_s,position_age_s,axis,true_pose,true_velocity,estimated_pose,estimated_velocity");
        Quaternion inv=Quaternion.Inverse(body.rotation);
        for(int i=0;i<8;i++) {
            observationDirections[i]=inv*thrusters[i].transform.TransformDirection(thrusters[i].LocalForceDirection.normalized);
            observationOffsets[i]=inv*(thrusters[i].transform.position-body.worldCenterOfMass);
        }
    }
    void ResetObservationProbe()
    {
        if(string.IsNullOrEmpty(observationMode))return;
        observationHistory.Clear();observationReady=false;observationChanged=false;
        heldPosition=heldVelocity=null;heldSourceTime=-1;nextPositionUpdate=SimTime;
        previousBodyVelocity=null;
        ResetRateSensors();
        observationNoise=new System.Random(evaluationSeed+900000);
        observationNoiseIndex=0;
    }
    void CaptureObservationTruth()
    {
        if(string.IsNullOrEmpty(observationMode))return;
        var p=new float[6];var v=new float[6];ReadState(p,v);
        observationHistory.Add(new ObservationSample{time=SimTime,p=p,v=v});
    }
    ObservationSample ObservationAt(double source)
    {
        int lo=0,hi=observationHistory.Count-1;
        while(lo<hi){int mid=(lo+hi+1)/2;if(observationHistory[mid].time<=source+1e-9)lo=mid;else hi=mid-1;}
        return observationHistory[lo];
    }
    void InitializeObservedControl(float t)
    {
        if(string.IsNullOrEmpty(observationMode))return;
        float[] truthP=(float[])stateP.Clone(),truthV=(float[])stateV.Clone();double age=0;
        if(observationMode!="clean") {
            // Schedule on physical ticks; decimal0.1 minus a float dt can
            // otherwise select the previous20ms sample at a boundary.
            var delayed=ObservationAt(SimTime-5d*Time.fixedDeltaTime);
            if(observationMode=="all100") {
                Array.Copy(delayed.p,stateP,6);Array.Copy(delayed.v,stateV,6);age=SimTime-delayed.time;
            } else {
                bool outage=observationMode=="pose_outage"&&t>=12f&&t<13f;
                if(heldPosition==null || (!outage && SimTime>=nextPositionUpdate-1e-9)) {
                    heldPosition=(float[])delayed.p.Clone();heldVelocity=(float[])delayed.v.Clone();heldSourceTime=delayed.time;
                    for(int a=0;a<3;a++){heldPosition[a]+=(a==1?.005f:.01f)*AuditedObservationNoise();heldVelocity[a]+=.02f*AuditedObservationNoise();}
                    nextPositionUpdate=SimTime+10d*Time.fixedDeltaTime;
                }
                age=SimTime-heldSourceTime;
                for(int a=0;a<3;a++){stateP[a]=heldPosition[a];stateV[a]=heldVelocity[a];
                    if(observationMode=="multi100_cv"||observationMode=="pose_outage")stateP[a]+=(float)age*heldVelocity[a];}
                // Current IMU orientation/rate remain available during position outage.
                ObserveRateImu();
            }
        }
        observationRotation=Quaternion.Euler(stateP[3]*Mathf.Rad2Deg,stateP[4]*Mathf.Rad2Deg,stateP[5]*Mathf.Rad2Deg);
        observationReady=true;observationChanged=true;
        for(int a=0;a<6;a++)observationCsv.WriteLine(string.Join(",",new[]{phase.ToString(),t.ToString("R",CultureInfo.InvariantCulture),age.ToString("R",CultureInfo.InvariantCulture),a.ToString(),
            RF(truthP[a]),RF(truthV[a]),RF(stateP[a]),RF(stateV[a])}));
        observationCsv.Flush();
        if(BodyFrameModel) {
            if(bodyObservationCsv==null) {
                string path=Path.Combine(artifactDirectory,"body_observations.csv");
                bool append=resumeLearnedArm && File.Exists(path);
                bodyObservationCsv=new StreamWriter(path,append);
                if(!append)bodyObservationCsv.WriteLine("phase,t_s,axis,true_body_velocity,estimated_body_velocity,true_body_velocity_derivative,derivative_valid");
            }
            var truthBody=FinsRovBodyFrame.ToBody(truthP,truthV);
            var estimatedBody=FinsRovBodyFrame.ToBody(stateP,stateV);
            bool valid=previousBodyVelocity!=null&&t>previousBodyTime;
            for(int a=0;a<6;a++)bodyObservationCsv.WriteLine(string.Join(",",phase.ToString(),RF(t),a,
                RF(truthBody[a]),RF(estimatedBody[a]),RF(valid?(truthBody[a]-previousBodyVelocity[a])/(t-previousBodyTime):0f),valid?1:0));
            bodyObservationCsv.Flush();previousBodyVelocity=truthBody;previousBodyTime=t;
        }
    }
    void RestoreObservationTruth()
    {
        if(!observationChanged)return;ReadState(stateP,stateV);observationChanged=false;
    }
    float[,] ObservationAllocationBasis()
    {
        if(!observationReady||!ViewerEvaluating||observationMode=="clean")return null;
        var basis=new float[6,8];
        for(int i=0;i<8;i++) {
            Vector3 d=observationRotation*observationDirections[i],t=Vector3.Cross(observationRotation*observationOffsets[i],d);
            basis[0,i]=d.x;basis[1,i]=d.y;basis[2,i]=d.z;basis[3,i]=t.x;basis[4,i]=t.y;basis[5,i]=t.z;
        }
        return basis;
    }
}
