using System;
using System.IO;
using UnityEngine;

// DE55 changes only MPC parameters; the truth profile is never perturbed.
public sealed partial class FinsRovHybridBasisSmoke
{
    [Serializable] sealed class ModelParametersAudit
    {
        public string arm,truthBefore,truthAfter,truthEnd;
        public float rigidMass;
        public Vector3 rigidInertia;
        public Quaternion inertiaRotation;
        public double[] sceneEffectiveMass,sceneLinearDrag,sceneQuadraticDrag,probeNextVelocity;
        public FinsRovUnifiedFossen.Parameters predictor;
    }
    ModelParametersAudit modelParametersAudit;

    void CaptureFixedTruth()
    {
        if (mismatchScale!=0f || Environment.GetEnvironmentVariable("FINSSIM_MISMATCH_MODE")!="model_parameters")
            throw new InvalidOperationException("DE55 requires zero truth mismatch and explicit model_parameters mode");
        string arm=Environment.GetEnvironmentVariable("FINSSIM_DE55_ARM");
        if (arm!="K_A" && arm!="K_B" && arm!="C_A" && arm!="C_B")
            throw new ArgumentException("Unknown DE55 arm");
        modelParametersAudit=new ModelParametersAudit {
            arm=arm,truthBefore=JsonUtility.ToJson(hydro.RuntimeProfile),rigidMass=body.mass,
            rigidInertia=body.inertiaTensor,inertiaRotation=body.inertiaTensorRotation,
            sceneEffectiveMass=Array.ConvertAll(mass,value=>(double)value),
            sceneLinearDrag=Array.ConvertAll(linearDrag,value=>(double)value),
            sceneQuadraticDrag=Array.ConvertAll(quadraticDrag,value=>(double)value)
        };
        hydro.ResetBackendState();
    }

    void InstallAndAuditFossenModel()
    {
        if (!hybrid.Unified) throw new InvalidOperationException("DE55 requires unified Fossen runtime");
        var q=hybrid.Data.nominal;
        if (hybrid.Data.predictionType!=modelParametersAudit.arm) throw new Exception("DE55 model arm mismatch");
        var ma=hydro.RuntimeProfile.addedMassDiagonal;
        string prior=Environment.GetEnvironmentVariable("FINSSIM_DE56_PRIOR");
        if(prior!="matched" && prior!="mismatched" && prior!="inertia" && prior!="damping")throw new ArgumentException("DE62 prior");
        bool changeM=prior=="mismatched" || prior=="inertia";
        bool changeD=prior=="mismatched" || prior=="damping";
        // Unity model order u,w,v,p,r,q; native hydrodynamics order u,v,w,p,q,r.
        var delta=new double[]{.525d*ma.u,.30d*ma.w,.375d*ma.v,0d,.45d*ma.r,0d};
        var factors=new double[]{1.825d,1.45d,1.675d,1d,1.525d,1d};
        var linearDelta=new double[]{0d,0d,0d,.27d,0d,.27d};
        var quadraticDelta=new double[]{18d,30d,24d,0d,0d,0d};
        for (int i=0;i<6;i++) {
            double expectedM=modelParametersAudit.sceneEffectiveMass[i]+(changeM?delta[i]:0d);
            double expectedL=modelParametersAudit.sceneLinearDrag[i]*(changeD?factors[i]:1d)+(changeD?linearDelta[i]:0d);
            double expectedQ=modelParametersAudit.sceneQuadraticDrag[i]+(changeD?quadraticDelta[i]:0d);
            if (Math.Abs(q.mass[i]-expectedM)>1e-5 || Math.Abs(q.linearDrag[i]-expectedL)>1e-5 || Math.Abs(q.quadraticDrag[i]-expectedQ)>1e-5)
                throw new Exception("DE55 incorrect predictor parameter axis "+i);
            if (Math.Abs(q.spring[i]-spring[i])>1e-5 || Math.Abs(q.bias[i]-bias[i])>1e-5)
                throw new Exception("DE55 restoring parameters changed");
        }
        // Prediction and model-dependent initial guesses use consistent parameters.
        for (int i=0;i<6;i++) {mass[i]=(float)q.mass[i];linearDrag[i]=(float)q.linearDrag[i];quadraticDrag[i]=(float)q.quadraticDrag[i];}
        modelParametersAudit.predictor=q;
        modelParametersAudit.truthAfter=JsonUtility.ToJson(hydro.RuntimeProfile);
        if (modelParametersAudit.truthBefore!=modelParametersAudit.truthAfter)
            throw new Exception("DE55 truth changed during model load");
        modelParametersAudit.probeNextVelocity=new double[36];
        for (int a=0;a<6;a++) {
            var z=new double[hybrid.Size];z[0]=1;
            // Nonzero speed makes both damping terms observable in the fixed-state probe.
            z[7+a]=.2/FinsRovLiftedOperator.VelocityScale[a];
            var u=new double[6];u[a]=.2;
            var next=BodyNominalStep(z,u,null,null);
            for (int i=0;i<6;i++) modelParametersAudit.probeNextVelocity[a*6+i]=next[7+i]*FinsRovLiftedOperator.VelocityScale[i];
        }
        VerifyFixedTruth(resumeLearnedArm?"learned_start":"baseline_start");
    }

    void VerifyFixedTruth(string stage)
    {
        modelParametersAudit.truthEnd=JsonUtility.ToJson(hydro.RuntimeProfile);
        if (modelParametersAudit.truthBefore!=modelParametersAudit.truthEnd || body.mass!=modelParametersAudit.rigidMass ||
            body.inertiaTensor!=modelParametersAudit.rigidInertia || body.inertiaTensorRotation!=modelParametersAudit.inertiaRotation)
            throw new Exception("DE55 truth changed within episode");
        File.WriteAllText(Path.Combine(artifactDirectory,"model_parameters_"+stage+".json"),JsonUtility.ToJson(modelParametersAudit,true));
    }
}


public sealed partial class FinsRovHybridBasisSmoke {
    void CompleteDe62Collection() {
        phase=Phase.Done;foreach(var motor in thrusters)motor.ApplyForceRequest(0f);OnDisable();
        System.IO.File.WriteAllText(System.IO.Path.Combine(artifactDirectory,"completed.json"),"{\"baseline_only\":true,\"baseline_samples\":"+baselineSamples+",\"compensated_samples\":0}");
        body.isKinematic=true;if(bodyStudy && TryAdvanceBodyQueue())return;UnityEngine.Application.Quit(0);
    }
}
