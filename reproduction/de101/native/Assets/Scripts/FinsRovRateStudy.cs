using System;
using System.Globalization;
using System.IO;
using UnityEngine;

// DE48: fixed 100 ms prediction, independently selected control update period.
public sealed partial class FinsRovHybridBasisSmoke
{
    const float ControlDt=.1f;
    float UpdateDt=.1f;
    bool rateDeadlineEnabled,rateCycleActive,rateHeld;
    bool de57SolverHold;
    long rateStarted;
    float ratePhaseTime;
    double ratePeriodMs,rateReadyMs;
    int rateHistorySamples;
    readonly float[] rateOldMotors=new float[8],rateOldRequest=new float[8];
    readonly float[] rateOldWrench=new float[6],rateOldAllocated=new float[6],rateOldHistory=new float[6];
    StreamWriter rateCsv,rateHistoryCsv,rateWarmCsv;
    float[] rateHeldAttitude,rateHeldAngularVelocity;
    double rateNextImu;
    readonly double[,] rateInputRing=new double[5,6];
    int rateInputIndex,rateInputCount;
    static string RateNumber(double v)=>v.ToString("R",CultureInfo.InvariantCulture);

    void ConfigureRateStudy()
    {
        UpdateDt=float.Parse(Environment.GetEnvironmentVariable("FINSSIM_DE48_UPDATE_DT")??"0.1",CultureInfo.InvariantCulture);
        if(Math.Abs(UpdateDt-.1f)>1e-7 && Math.Abs(UpdateDt-.02f)>1e-7)
            throw new ArgumentOutOfRangeException("FINSSIM_DE48_UPDATE_DT");
        if(Math.Abs(Time.fixedDeltaTime-.02f)>1e-7)
            throw new InvalidOperationException("DE48 requires 20ms physics tick");
        rateDeadlineEnabled=Environment.GetEnvironmentVariable("FINSSIM_DE48_DEADLINE_HOLD")=="1" &&
            Environment.GetEnvironmentVariable("FINSSIM_DUAL_PREFLIGHT")!="1";
    }

    void ValidateRateModel()
    {
        if(hybrid==null || !hybrid.Unified || Math.Abs(hybrid.SampleDt-.1d)>1e-7 || learnedHorizon!=8)
            throw new InvalidOperationException("DE51 requires unified schema15..18 / 100ms / H8 model");
        File.WriteAllText(Path.Combine(artifactDirectory,"rate_contract.json"),
            "{\"model_dt\":0.1,\"baseline_model_dt\":0.1,\"learned_period\":"+RateNumber(UpdateDt)+
            ",\"baseline_period\":"+RateNumber(UpdateDt)+",\"learned_horizon\":8,\"baseline_horizon\":8,"+
            "\"prediction_duration_s\":0.8,\"history_duration_s\":0.1,\"deadline_hold\":"+(rateDeadlineEnabled?"true":"false")+"}");
    }

    void ResetRateSensors()
    {
        rateHeldAttitude=null;rateHeldAngularVelocity=null;rateNextImu=SimTime;
        Array.Clear(rateInputRing,0,rateInputRing.Length);rateInputIndex=rateInputCount=0;
    }

    void ObserveRateImu()
    {
        if(rateHeldAttitude==null || SimTime>=rateNextImu-1e-9) {
            rateHeldAttitude=new float[3];rateHeldAngularVelocity=new float[3];
            for(int a=3;a<6;a++) {
                rateHeldAttitude[a-3]=stateP[a]+.3f*Mathf.Deg2Rad*AuditedObservationNoise();
                rateHeldAngularVelocity[a-3]=stateV[a]+.5f*Mathf.Deg2Rad*AuditedObservationNoise();
            }
            rateNextImu=SimTime+5d*Time.fixedDeltaTime;
        }
        for(int a=3;a<6;a++){stateP[a]=rateHeldAttitude[a-3];stateV[a]=rateHeldAngularVelocity[a-3];}
    }

    void ObserveRateHistory(double[] estimated)
    {
        if(!ViewerEvaluating)return;
        for(int a=0;a<6;a++)rateInputRing[rateInputIndex,a]=estimated[a];
        rateInputIndex=(rateInputIndex+1)%5;rateInputCount=Math.Min(5,rateInputCount+1);
    }

    void FinishRateHistory()
    {
        for(int a=0;a<6;a++) {
            double sum=0;for(int k=0;k<5;k++)sum+=rateInputRing[k,a];
            historyWrench[a]=(float)(sum/5d);
        }
        ResetBodyInputInterval();
        if(rateHistoryCsv==null) {
            string path=Path.Combine(artifactDirectory,"rolling_history.csv");bool append=File.Exists(path);
            rateHistoryCsv=new StreamWriter(path,true);
            if(!append)rateHistoryCsv.WriteLine("phase,t_s,available_ticks,history_0,history_1,history_2,history_3,history_4,history_5");
        }
        rateHistoryCsv.Write(phase+","+RateNumber(SimTime-phaseStart)+","+rateInputCount);
        for(int a=0;a<6;a++)rateHistoryCsv.Write(","+RateNumber(historyWrench[a]));
        rateHistoryCsv.WriteLine();rateHistoryCsv.Flush();
    }

    void BuildRateWarmStart(double[,] sequence,float t)
    {
        double fraction=(double)UpdateDt/ControlDt;
        bool audit=Environment.GetEnvironmentVariable("FINSSIM_DUAL_PREFLIGHT")=="1" && t<.31f;
        if(audit && rateWarmCsv==null) {
            string path=Path.Combine(artifactDirectory,"warmstart_alignment.csv");bool append=File.Exists(path);
            rateWarmCsv=new StreamWriter(path,true);
            if(!append)rateWarmCsv.WriteLine("phase,t_s,k,axis,old_k,old_next,new_k,fraction");
        }
        for(int k=0;k<Horizon;k++)for(int a=0;a<DecisionCount;a++) {
            double x=warm[k,a],y=warm[Math.Min(Horizon-1,k+1),a];
            sequence[k,a]=(1d-fraction)*x+fraction*y;
            if(audit)rateWarmCsv.WriteLine(string.Join(",",phase,RateNumber(t),k,a,RateNumber(x),RateNumber(y),RateNumber(sequence[k,a]),RateNumber(fraction)));
        }
        if(audit)rateWarmCsv.Flush();
    }

    double[] PureFossenLift(float[] pose,float[] worldVelocity)
    {
        var z=new double[ModelSize];
        Array.Copy(FinsRovLiftedOperator.Lift(pose,FinsRovBodyFrame.ToBody(pose,worldVelocity)),z,19);
        return z; // Fossen uses physical pose/velocity only, without neural encoding or learned operators.
    }

    void BeginRateCycle()
    {
        if(!bodyStudy || !ViewerEvaluating || SimTime+.0001d<nextControl)return;
        rateStarted=System.Diagnostics.Stopwatch.GetTimestamp();rateCycleActive=true;rateHeld=false;
        ratePhaseTime=(float)(SimTime-phaseStart);ratePeriodMs=1000d*UpdateDt;rateReadyMs=0;
        rateHistorySamples=bodyInputSteps;
        Array.Copy(optimizedMotors,rateOldMotors,8);Array.Copy(motorRequest,rateOldRequest,8);
        Array.Copy(wrench,rateOldWrench,6);Array.Copy(allocatedWrench,rateOldAllocated,6);Array.Copy(bodyHistoryWrench,rateOldHistory,6);
    }

    double RateElapsed()=>1000d*(System.Diagnostics.Stopwatch.GetTimestamp()-rateStarted)/System.Diagnostics.Stopwatch.Frequency;
    void RateDeadlineGate()
    {
        if(!rateCycleActive)return;
        rateReadyMs=RateElapsed();rateHeld=rateDeadlineEnabled && rateReadyMs>ratePeriodMs;
        if(!rateHeld)return;
        Array.Copy(rateOldMotors,optimizedMotors,8);Array.Copy(rateOldRequest,motorRequest,8);
        Array.Copy(rateOldWrench,wrench,6);Array.Copy(rateOldAllocated,allocatedWrench,6);Array.Copy(rateOldHistory,bodyHistoryWrench,6);
    }

    void FinishRateCycle()
    {
        if(!rateCycleActive)return;
        double elapsed=RateElapsed();rateCycleActive=false;
        if(rateCsv==null) {
            string path=Path.Combine(artifactDirectory,"full_cycle_timing.csv");bool append=File.Exists(path);
            rateCsv=new StreamWriter(path,true);
            if(!append)rateCsv.WriteLine("phase,t_s,period_ms,solver_ms,ready_ms,cycle_ms,held_previous,full_overrun,post_dispatch_overrun,history_samples,model_dt,solver_hold");
        }
        rateCsv.WriteLine(string.Join(",",phase,RateNumber(ratePhaseTime),RateNumber(ratePeriodMs),RateNumber(lastSolveMs),
            RateNumber(rateReadyMs),RateNumber(elapsed),rateHeld?"true":"false",elapsed>ratePeriodMs?"true":"false",
            !rateHeld&&elapsed>ratePeriodMs?"true":"false",rateHistorySamples,"0.1",de57SolverHold?"true":"false"));
        rateCsv.Flush();
    }
    void OnDestroy(){rateCsv?.Dispose();rateHistoryCsv?.Dispose();rateWarmCsv?.Dispose();}
}
