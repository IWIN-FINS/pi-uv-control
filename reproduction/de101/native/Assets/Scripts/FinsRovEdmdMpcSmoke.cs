using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using FinsSim.Actuators;
using FinsSim.Hydrodynamics;
using UnityEngine;

// Paired calibrated Fossen MPC versus physics-informed, controlled EDMDc MPC.
// Both arms share the same joint six-axis projected-gradient optimizer, cost,
// wrench limits, allocation, truth plant and paired random disturbances.
[DefaultExecutionOrder(-50)]
public sealed class FinsRovEdmdMpcSmoke : MonoBehaviour
{
    enum Phase { Identify, SettleBaseline, BaselineMpc, SettleKoopman, KoopmanMpc, Done }
    const int AxisCount = 6;
    const int Horizon = 12;
    const int GradientIterations = 32;
    const float ControlDt = 0.1f;
    const float IdentifySeconds = 96f;
    const float EvaluationSeconds = 32f;
    const float SettleSeconds = 2f;
    const float NoiseStdN = 0.20f;
    const float ThrustScale = 0.78f;
    const float Period = 32f;
    const int SegmentSeconds = 24;
    static readonly string[] Names =
    {
        "Vertical1", "Vertical2", "Vertical3", "Vertical4",
        "Horizontal1", "Horizontal2", "Horizontal3", "Horizontal4"
    };
    // x, depth(y), z, roll(x), yaw(y), pitch(z); angular states are radians.
    static readonly float[] WrenchLimit = { 15f, 12f, 15f, 2.0f, 2.0f, 2.0f };
    static readonly float[] PositionWeight = { 380f, 550f, 460f, 600f, 280f, 600f };
    static readonly float[] VelocityWeight = { 25f, 45f, 30f, 32f, 20f, 32f };
    static readonly float[] EffortWeight = { 0.012f, 0.016f, 0.012f, 0.045f, 0.045f, 0.045f };

    readonly Thruster[] thrusters = new Thruster[8];
    readonly float[] mass = new float[AxisCount];
    readonly float[] linearDrag = new float[AxisCount];
    readonly float[] quadraticDrag = new float[AxisCount];
    readonly float[] spring = new float[AxisCount];
    readonly float[] bias = new float[AxisCount];
    readonly float[] wrench = new float[AxisCount];
    readonly float[] motorRequest = new float[8];
    readonly double[,] warm = new double[Horizon, 8];
    readonly List<FinsRovLiftedOperator.Transition> transitions =
        new List<FinsRovLiftedOperator.Transition>();
    readonly FinsRovLiftedOperator koopman = new FinsRovLiftedOperator();
    readonly float[] previousP = new float[AxisCount];
    readonly float[] previousV = new float[AxisCount];
    readonly float[] previousWrench = new float[AxisCount];
    readonly float[] stateP = new float[AxisCount];
    readonly float[] stateV = new float[AxisCount];
    readonly float[] predictedInitialCost = new float[AxisCount];
    readonly float[] predictedFinalCost = new float[AxisCount];
    Rigidbody body;
    HydrodynamicsController hydro;
    StreamWriter trialCsv;
    StreamWriter identifyCsv;
    System.Random noise;
    Phase phase;
    string artifactDirectory;
    double phaseStart;
    double nextControl;
    double nextSample;
    double previousSampleTime = -1d;
    double SimTime => batchSeeds!=null ? Time.fixedTimeAsDouble : Time.fixedTime;
    float lastSolveMs;
    float lastAllocationResidual;
    float lastMotorEffort;
    int lastSaturatedMotors;
    int baselineSamples;
    int koopmanSamples;
    int evaluationSeed;
    int identifySegment;
    int trainCount;
    bool modelValid;
    double baselineHorizontalSquared;
    double koopmanHorizontalSquared;
    string diagnosticMode = "full";
    readonly FinsRovHistoryOperator historyOperator = new FinsRovHistoryOperator();
    readonly float[] historyWrench = new float[6];
    bool HistoryMode => diagnosticMode == "history" || diagnosticMode == "invariant";
    int ModelSize => HistoryMode ? FinsRovHistoryOperator.Size : FinsRovLiftedOperator.Lifted;
    double[,] ModelA => HistoryMode ? historyOperator.A : koopman.A;
    double[,] ModelB => HistoryMode ? historyOperator.B : koopman.B;
    float identifyDuration = IdentifySeconds;
    float segmentDuration = SegmentSeconds;
    bool expandedTraining;
    bool actuatorMpc;
    int DecisionCount => actuatorMpc ? 8 : AxisCount;
    float[,] optimizationBasis;
    readonly float[] optimizedMotors = new float[8];
    float periodScale=1f, amplitudeScale=1f, mismatchScale=1f;
    float evaluationDuration=EvaluationSeconds;
    string frozenModelPath;
    string comparisonMethod;
    bool isolatedScenes, resumeLearnedArm;
    int[] batchSeeds;
    int batchIndex;
    string batchRoot;
    double allocationRegularization = 0.04d;
    StreamWriter diagnosticCsv;
    readonly double[] intervalDesired = new double[AxisCount];
    readonly double[] intervalAllocated = new double[AxisCount];
    readonly double[] intervalApplied = new double[AxisCount];
    readonly float[] allocatedWrench = new float[AxisCount];
    double[] diagnosticNominal, diagnosticLearned;
    int intervalSteps;
    float diagnosticStart;

    public string ViewerPhase => phase.ToString();
    public float ViewerPhaseTime => phase == Phase.Done ? evaluationDuration :
        Mathf.Max(0f, (float)(SimTime - phaseStart));
    public float ViewerPhaseDuration => phase == Phase.Identify ? identifyDuration :
        phase == Phase.BaselineMpc || phase == Phase.KoopmanMpc || phase == Phase.Done ?
        evaluationDuration : SettleSeconds;
    public bool ViewerEvaluating => phase == Phase.BaselineMpc || phase == Phase.KoopmanMpc;
    public Vector2 ViewerPosition => new Vector2(body.position.x, body.position.z);
    public Vector2 ViewerReference => new Vector2(Reference(0, ViewerPhaseTime), Reference(2, ViewerPhaseTime));
    public float ViewerDepthError => body.position.y - Reference(1, ViewerPhaseTime);
    public float ViewerTilt => Vector3.Angle(body.rotation * Vector3.up, Vector3.up);
    public float ViewerYawError => Mathf.DeltaAngle(Reference(4, ViewerPhaseTime) * Mathf.Rad2Deg,
        body.rotation.eulerAngles.y);
    public float ViewerBaselineRmse => baselineSamples > 0 ?
        (float)Math.Sqrt(baselineHorizontalSquared / baselineSamples) : 0f;
    public float ViewerKoopmanRmse => koopmanSamples > 0 ?
        (float)Math.Sqrt(koopmanHorizontalSquared / koopmanSamples) : 0f;

    void Start()
    {
        diagnosticMode = Environment.GetEnvironmentVariable("FINSSIM_EDMD_MPC_MODE") ?? "full";
        isolatedScenes=Environment.GetEnvironmentVariable("FINSSIM_EDMD_ISOLATED_SCENES")=="1";
        resumeLearnedArm=isolatedScenes && Environment.GetEnvironmentVariable("FINSSIM_EDMD_RESUME_LEARNED")=="1";
        comparisonMethod = Environment.GetEnvironmentVariable("FINSSIM_EDMD_COMPARISON_METHOD") ?? "koopman";
        if (comparisonMethod != "koopman" && comparisonMethod != "static" && comparisonMethod != "refit")
            throw new ArgumentException("Unknown comparison method: " + comparisonMethod);
        if (diagnosticMode != "full" && diagnosticMode != "translation" &&
            diagnosticMode != "rotation_decoupled" && diagnosticMode != "no_pitch_bias" &&
            diagnosticMode != "no_horizontal_position" && diagnosticMode != "no_cross_input" &&
            diagnosticMode != "history" && diagnosticMode != "invariant")
            throw new ArgumentException("Unknown diagnostic mode: " + diagnosticMode);
        expandedTraining = Environment.GetEnvironmentVariable("FINSSIM_EDMD_MPC_TRAINING") == "tracking";
        if (expandedTraining) { identifyDuration = 192f; segmentDuration = 48f; }
        actuatorMpc = Environment.GetEnvironmentVariable("FINSSIM_EDMD_MPC_ACTUATOR_MPC") == "1";
        frozenModelPath = Environment.GetEnvironmentVariable("FINSSIM_EDMD_MPC_MODEL_PATH");
        string batch=Environment.GetEnvironmentVariable("FINSSIM_EDMD_MPC_BATCH_SEEDS");
        if (!string.IsNullOrEmpty(batch))
        {
            if (string.IsNullOrEmpty(frozenModelPath)) throw new ArgumentException("Batch evaluation requires a frozen model");
            var parts=batch.Split(','); batchSeeds=new int[parts.Length];
            var unique=new System.Collections.Generic.HashSet<int>();
            for (int i=0;i<parts.Length;i++)
            {
                batchSeeds[i]=int.Parse(parts[i],CultureInfo.InvariantCulture);
                if (!unique.Add(batchSeeds[i])) throw new ArgumentException("Duplicate batch seed");
            }
        }
        float ReadScale(string name) => float.TryParse(Environment.GetEnvironmentVariable(name),
            NumberStyles.Float,CultureInfo.InvariantCulture,out float value) ? value : 1f;
        periodScale=ReadScale("FINSSIM_EDMD_MPC_PERIOD_SCALE");
        amplitudeScale=ReadScale("FINSSIM_EDMD_MPC_AMPLITUDE_SCALE");
        mismatchScale=ReadScale("FINSSIM_EDMD_MPC_MISMATCH_SCALE");
        if (periodScale<.5f || periodScale>2f || amplitudeScale<.5f || amplitudeScale>1.5f ||
            mismatchScale<0f || mismatchScale>2f) throw new ArgumentException("Research scenario scale outside predefined range");
        evaluationDuration=EvaluationSeconds*periodScale;
        string regularization = Environment.GetEnvironmentVariable("FINSSIM_EDMD_MPC_ALLOC_REG");
        if (!string.IsNullOrEmpty(regularization))
            allocationRegularization = double.Parse(regularization, CultureInfo.InvariantCulture);
        if (allocationRegularization <= 0d) throw new ArgumentException("Allocation regularization must be positive");
        if (float.TryParse(Environment.GetEnvironmentVariable("FINSSIM_EDMD_MPC_TIME_SCALE"),
            NumberStyles.Float, CultureInfo.InvariantCulture, out float timeScale))
            Time.timeScale = Mathf.Clamp(timeScale, 1f, 10f);
        body = GetComponent<Rigidbody>();
        hydro = GetComponent<HydrodynamicsController>();
        if (body == null || hydro == null || hydro.profile == null)
            throw new InvalidOperationException("MPC smoke requires FinsROV Rigidbody and Fossen profile");
        foreach (var candidate in GetComponentsInChildren<Thruster>(true))
            for (int i = 0; i < Names.Length; i++)
                if (candidate.name == Names[i]) thrusters[i] = candidate;
        foreach (var motor in thrusters)
            if (motor == null) throw new InvalidOperationException("Eight FinsROV thrusters were not resolved");

        var nominal = hydro.profile;
        Vector3 inertia = body.inertiaTensor;
        mass[0] = body.mass + nominal.addedMassDiagonal.u;
        mass[1] = body.mass + nominal.addedMassDiagonal.w;
        mass[2] = body.mass + nominal.addedMassDiagonal.v;
        mass[3] = Mathf.Max(0.05f, inertia.x + nominal.addedMassDiagonal.p);
        mass[4] = Mathf.Max(0.05f, inertia.y + nominal.addedMassDiagonal.r);
        mass[5] = Mathf.Max(0.05f, inertia.z + nominal.addedMassDiagonal.q);
        linearDrag[0] = nominal.linearDamping.u;
        linearDrag[1] = nominal.linearDamping.w;
        linearDrag[2] = nominal.linearDamping.v;
        linearDrag[3] = nominal.linearDamping.p;
        linearDrag[4] = nominal.linearDamping.r;
        linearDrag[5] = nominal.linearDamping.q;
        quadraticDrag[0] = nominal.quadraticDamping.u;
        quadraticDrag[1] = nominal.quadraticDamping.w;
        quadraticDrag[2] = nominal.quadraticDamping.v;
        quadraticDrag[3] = nominal.quadraticDamping.p;
        quadraticDrag[4] = nominal.quadraticDamping.r;
        quadraticDrag[5] = nominal.quadraticDamping.q;
        float buoyancy = nominal.waterDensity * nominal.displacedVolume * nominal.gravityMagnitude;
        bias[1] = buoyancy - body.mass * nominal.gravityMagnitude;
        spring[3] = buoyancy * (nominal.centerOfBuoyancy.y - nominal.centerOfMass.y);
        spring[5] = spring[3];

        // Synthetic plant mismatch exists only in the runtime truth profile.
        var truth = hydro.RuntimeProfile;
        truth.addedMassDiagonal.u *= 1f+.35f*mismatchScale;
        truth.addedMassDiagonal.v *= 1f+.25f*mismatchScale;
        truth.addedMassDiagonal.w *= 1f+.20f*mismatchScale;
        truth.addedMassDiagonal.r *= 1f+.30f*mismatchScale;
        truth.linearDamping.u *= 1f+.55f*mismatchScale;
        truth.linearDamping.v *= 1f+.45f*mismatchScale;
        truth.linearDamping.w *= 1f+.30f*mismatchScale;
        truth.linearDamping.r *= 1f+.35f*mismatchScale;
        truth.linearDamping.p += .18f*mismatchScale;
        truth.linearDamping.q += .18f*mismatchScale;
        truth.quadraticDamping.u += 12f*mismatchScale;
        truth.quadraticDamping.v += 16f*mismatchScale;
        truth.quadraticDamping.w += 20f*mismatchScale;
        hydro.ResetBackendState();

        artifactDirectory = Environment.GetEnvironmentVariable("FINSSIM_EDMD_MPC_DIR");
        if (string.IsNullOrEmpty(artifactDirectory))
            artifactDirectory = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/edmd-mpc-smoke"));
        if (batchSeeds!=null)
        {
            batchRoot=artifactDirectory;
            artifactDirectory=Path.Combine(batchRoot,diagnosticMode+"-seed"+batchSeeds[0]);
        }
        Directory.CreateDirectory(artifactDirectory);
        File.WriteAllText(Path.Combine(artifactDirectory,"comparison_method.txt"), comparisonMethod);
        File.WriteAllText(Path.Combine(artifactDirectory, "diagnostic_config.json"),
            "{\"mode\":\"" + diagnosticMode + "\",\"allocation_regularization\":" +
            allocationRegularization.ToString("R", CultureInfo.InvariantCulture) +
            ",\"fixed_dt\":" + Time.fixedDeltaTime.ToString("R", CultureInfo.InvariantCulture) +
            ",\"expanded_training\":"+(expandedTraining?"true":"false")+
            ",\"actuator_mpc\":"+(actuatorMpc?"true":"false")+
            ",\"period_scale\":"+periodScale.ToString("R",CultureInfo.InvariantCulture)+
            ",\"amplitude_scale\":"+amplitudeScale.ToString("R",CultureInfo.InvariantCulture)+
            ",\"mismatch_scale\":"+mismatchScale.ToString("R",CultureInfo.InvariantCulture)+
            ",\"frozen_model\":"+(!string.IsNullOrEmpty(frozenModelPath)?"true":"false")+"}");
        diagnosticCsv = new StreamWriter(Path.Combine(artifactDirectory, "actuation_prediction.csv"), resumeLearnedArm);
        if (!resumeLearnedArm) diagnosticCsv.WriteLine("phase,t_start,t_end,axis,desired_mean,allocated_mean,applied_mean,next_pose,nominal_pose,learned_pose,next_velocity,nominal_velocity,learned_velocity,physics_steps");
        gameObject.AddComponent<FinsRovEdmdActuationObserver>().Owner = this;
        if (resumeLearnedArm)
            foreach (var line in File.ReadAllLines(Path.Combine(artifactDirectory,"mpc_trials.csv")))
            {
                var cells=line.Split(','); if(cells[0]!="BaselineMpc") continue;
                double error=double.Parse(cells[8],CultureInfo.InvariantCulture);
                baselineHorizontalSquared+=error*error; baselineSamples++;
            }
        trialCsv = new StreamWriter(Path.Combine(artifactDirectory, "mpc_trials.csv"), resumeLearnedArm);
        if (!resumeLearnedArm) trialCsv.WriteLine("phase,t_s,x_m,y_m,z_m,ref_x_m,ref_y_m,ref_z_m,horizontal_error_m,depth_error_m,roll_deg,pitch_deg,yaw_deg,ref_yaw_deg,tilt_deg,attitude_error_deg,fx_n,fy_n,fz_n,tx_nm,ty_nm,tz_nm,motor_effort_n,allocation_residual,solver_ms,predicted_initial_cost,predicted_final_cost,saturated_motors");
        identifyCsv = new StreamWriter(Path.Combine(artifactDirectory, "mpc_identification.csv"), resumeLearnedArm);
        if (!resumeLearnedArm) identifyCsv.WriteLine("segment,t_s,axis,position,velocity,requested_wrench,next_position,next_velocity");
        phase = Phase.Identify;
        phaseStart = SimTime;
        nextControl = SimTime;
        nextSample = SimTime;
        noise = new System.Random(24001);
        if (!int.TryParse(Environment.GetEnvironmentVariable("FINSSIM_EDMD_MPC_EVAL_SEED"),
            NumberStyles.Integer, CultureInfo.InvariantCulture, out evaluationSeed))
            evaluationSeed = 9091;
        if (batchSeeds!=null) evaluationSeed=batchSeeds[0];
        koopman.BuildNominalPrior(mass, linearDrag, quadraticDrag, spring, bias);
        historyOperator.BuildPrior(koopman);
        if (!string.IsNullOrEmpty(frozenModelPath))
        {
            LoadFrozenModel(frozenModelPath);
            phase=resumeLearnedArm?Phase.SettleKoopman:Phase.SettleBaseline;
            VerifyGradients();
            ResetVehicle();
            using (var sha=System.Security.Cryptography.SHA256.Create())
                File.WriteAllText(Path.Combine(artifactDirectory,"frozen_model_sha256.txt"),
                    BitConverter.ToString(sha.ComputeHash(File.ReadAllBytes(frozenModelPath))).Replace("-","").ToLowerInvariant());
        }
        Debug.Log("FINSSIM_EDMD_MPC_START " + artifactDirectory);
        Debug.Log("FINSSIM_EDMD_MPC_EVAL_SEED " + evaluationSeed);
        Debug.Log($"FINSSIM_EDMD_MPC_MODEL mass={string.Join(",", mass)} spring={spring[3]:F3} buoyancyBias={bias[1]:F3}");
    }

    void FixedUpdate()
    {
        if (phase == Phase.Done) return;
        float t = (float)(SimTime - phaseStart);
        float duration = phase == Phase.Identify ? identifyDuration :
            phase == Phase.BaselineMpc || phase == Phase.KoopmanMpc ? evaluationDuration : SettleSeconds;
        // Time.fixedTime is a float; without this tolerance one arm can write
        // a spurious 31.99999 s sample while the other ends at 31.9 s.
        if (t >= duration - 0.0001f)
        {
            AdvancePhase();
            if (phase == Phase.Done) return;
            t = 0f;
        }
        if (phase == Phase.Identify)
        {
            int segment = Mathf.Min(3, Mathf.FloorToInt(t / segmentDuration));
            if (segment > identifySegment)
            {
                if (segment == 3) trainCount = transitions.Count;
                identifySegment = segment;
                ResetVehicle();
                noise = new System.Random(24001 + 271 * segment);
                Debug.Log($"FINSSIM_EDMD_MPC_IDENT_SEGMENT {segment} pairs={transitions.Count}");
            }
        }
        ReadState(stateP, stateV);
        if (SimTime + 0.0001d >= nextControl)
        {
            if (phase == Phase.Identify)
            {
                RecordIdentification(t);
                Excitation(t);
                Array.Copy(stateP, previousP, AxisCount);
                Array.Copy(stateV, previousV, AxisCount);
                Array.Copy(wrench, previousWrench, AxisCount);
                previousSampleTime = SimTime;
            }
            else if (phase == Phase.BaselineMpc || phase == Phase.KoopmanMpc)
            {
                FinishDiagnosticInterval(t);
                Array.Copy(wrench,historyWrench,6);
                SolveMpc(t, phase == Phase.KoopmanMpc);
                var lifted = LiftState(stateP, stateV,historyWrench);
                var input = FinsRovLiftedOperator.Input(wrench);
                diagnosticNominal = NominalStep(lifted, input, null, null);
                diagnosticLearned = PredictionStep(lifted, input, true, null, null);
                diagnosticStart = t;
            }
            else Array.Clear(wrench, 0, AxisCount);
            nextControl = SimTime + ControlDt;
        }
        if (phase == Phase.SettleBaseline || phase == Phase.SettleKoopman)
        {
            foreach (var motor in thrusters) motor.ApplyForceRequest(0f);
            return;
        }
        if (actuatorMpc && (ViewerEvaluating || (phase==Phase.Identify && expandedTraining)))
        {
            Array.Copy(optimizedMotors,motorRequest,8);
            var basis=GetAllocationBasis();
            float error=0f;
            for (int a=0;a<6;a++)
            {
                allocatedWrench[a]=0f;
                for (int i=0;i<8;i++) allocatedWrench[a]+=basis[a,i]*motorRequest[i];
                error+=(allocatedWrench[a]-wrench[a])*(allocatedWrench[a]-wrench[a]);
            }
            lastAllocationResidual=Mathf.Sqrt(error);
        }
        else Allocate(wrench, motorRequest);
        lastMotorEffort = 0f;
        lastSaturatedMotors = 0;
        for (int i = 0; i < thrusters.Length; i++)
        {
            float force = ThrustScale * motorRequest[i] + NoiseStdN * Gaussian(noise);
            thrusters[i].ApplyForceRequest(force);
            lastMotorEffort += Mathf.Abs(force);
            if (Mathf.Abs(motorRequest[i]) >= (motorRequest[i] >= 0f ?
                thrusters[i].MaxForwardForceN : -thrusters[i].MaxReverseForceN) - 0.05f)
                lastSaturatedMotors++;
        }
        if ((phase == Phase.BaselineMpc || phase == Phase.KoopmanMpc) &&
            SimTime + 0.0001d >= nextSample)
        {
            RecordTrial(t);
            nextSample = SimTime + ControlDt;
        }
    }

    void ReadState(float[] p, float[] v)
    {
        Vector3 position = body.position;
        Vector3 velocity = body.linearVelocity;
        Vector3 euler = body.rotation.eulerAngles;
        Vector3 angular = body.angularVelocity;
        p[0] = position.x; p[1] = position.y; p[2] = position.z;
        p[3] = Mathf.DeltaAngle(0f, euler.x) * Mathf.Deg2Rad;
        p[4] = Mathf.DeltaAngle(0f, euler.y) * Mathf.Deg2Rad;
        p[5] = Mathf.DeltaAngle(0f, euler.z) * Mathf.Deg2Rad;
        v[0] = velocity.x; v[1] = velocity.y; v[2] = velocity.z;
        v[3] = angular.x; v[4] = angular.y; v[5] = angular.z;
    }

    float Reference(int axis, float t)
    {
        if (expandedTraining && phase == Phase.Identify)
        {
            float local=t-identifySegment*segmentDuration, shift=identifySegment*.73f;
            if (axis==0) return .50f*Mathf.Sin(.23f*local+shift)+.12f*Mathf.Sin(.71f*local);
            if (axis==1) return -1.2f+.12f*Mathf.Sin(.21f*local+shift);
            if (axis==2) return .28f*Mathf.Cos(.19f*local+shift)+.10f*Mathf.Sin(.47f*local);
            if (axis==3) return .025f*Mathf.Sin(.31f*local);
            if (axis==4) return .12f*Mathf.Sin(.17f*local+shift);
            return .03f*Mathf.Cos(.27f*local+shift);
        }
        float omega = 2f * Mathf.PI / (Period*periodScale);
        if (axis == 0) return 0.60f*amplitudeScale * Mathf.Sin(omega * t);
        if (axis == 1) return -1.20f + 0.08f * Mathf.Sin(omega * t);
        if (axis == 2) return 0.35f*amplitudeScale * Mathf.Sin(2f * omega * t);
        if (axis == 4) return 0.10f * Mathf.Sin(omega * t);
        return 0f;
    }

    float ReferenceVelocity(int axis, float t)
    {
        if (expandedTraining && phase == Phase.Identify)
        {
            float local=t-identifySegment*segmentDuration, shift=identifySegment*.73f;
            if (axis==0) return .115f*Mathf.Cos(.23f*local+shift)+.0852f*Mathf.Cos(.71f*local);
            if (axis==1) return .0252f*Mathf.Cos(.21f*local+shift);
            if (axis==2) return -.0532f*Mathf.Sin(.19f*local+shift)+.047f*Mathf.Cos(.47f*local);
            if (axis==3) return .00775f*Mathf.Cos(.31f*local);
            if (axis==4) return .0204f*Mathf.Cos(.17f*local+shift);
            return -.0081f*Mathf.Sin(.27f*local+shift);
        }
        float omega = 2f * Mathf.PI / (Period*periodScale);
        if (axis == 0) return 0.60f*amplitudeScale * omega * Mathf.Cos(omega * t);
        if (axis == 1) return 0.08f * omega * Mathf.Cos(omega * t);
        if (axis == 2) return 0.70f*amplitudeScale * omega * Mathf.Cos(2f * omega * t);
        if (axis == 4) return 0.10f * omega * Mathf.Cos(omega * t);
        return 0f;
    }

    void Excitation(float t)
    {
        if (expandedTraining)
        {
            Array.Copy(wrench,historyWrench,6);
            SolveMpc(t,false);
            float trackingLocal=t-identifySegment*segmentDuration;
            if (actuatorMpc)
            {
                for (int i=0;i<8;i++) optimizedMotors[i]=Mathf.Clamp(optimizedMotors[i]+.7f*Mathf.Sin((.63f+.17f*i)*trackingLocal+identifySegment*.69f),
                    -Mathf.Abs(thrusters[i].MaxReverseForceN),Mathf.Abs(thrusters[i].MaxForwardForceN));
                var basis=GetAllocationBasis();
                for (int a=0;a<6;a++)
                {
                    wrench[a]=0f;
                    for (int i=0;i<8;i++) wrench[a]+=basis[a,i]*optimizedMotors[i];
                }
                return;
            }
            for (int a=0;a<6;a++)
                wrench[a]=Mathf.Clamp(wrench[a]+(a<3?2f:.40f)*Mathf.Sin((.63f+.21f*a)*trackingLocal+identifySegment*.69f),
                    -WrenchLimit[a],WrenchLimit[a]);
            return;
        }
        float local = t - identifySegment * SegmentSeconds;
        float shift = identifySegment * 0.69f;
        wrench[0] = 6f * Mathf.Sin((0.77f + 0.07f * identifySegment) * local + shift)
            + 2.5f * Mathf.Sin(1.61f * local + 0.35f * shift);
        wrench[1] = 3f * Mathf.Sin((0.83f + 0.05f * identifySegment) * local + 0.7f * shift)
            + 1.3f * Mathf.Sin(1.37f * local);
        wrench[2] = 6f * Mathf.Cos((0.57f + 0.09f * identifySegment) * local + shift)
            + 2.5f * Mathf.Sin(1.29f * local);
        wrench[3] = 0.35f * Mathf.Sin(0.91f * local + shift);
        wrench[4] = 0.35f * Mathf.Cos(0.73f * local + shift);
        wrench[5] = 0.35f * Mathf.Sin(1.09f * local + 0.5f * shift);
    }

    float NominalAcceleration(int axis, float p, float v, float u)
    {
        return (u - linearDrag[axis] * v - quadraticDrag[axis] * v * Mathf.Abs(v)
            - spring[axis] * p + bias[axis]) / mass[axis];
    }

    void RecordIdentification(float t)
    {
        if (previousSampleTime < 0f) return;
        float dt = (float)(SimTime - previousSampleTime);
        if (dt <= 0.001f) return;
        transitions.Add(new FinsRovLiftedOperator.Transition(previousP, previousV,
            previousWrench, stateP, stateV, identifySegment));
        for (int axis = 0; axis < AxisCount; axis++)
        {
            identifyCsv.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "{0},{1:F3},{2},{3:F6},{4:F6},{5:F6},{6:F6},{7:F6}",
                identifySegment, t, axis, previousP[axis], previousV[axis],
                previousWrench[axis], stateP[axis], stateV[axis]));
        }
    }

    void FitOperator()
    {
        if (trainCount <= 0) throw new InvalidOperationException("EDMDc validation segment was not reached");
        if (HistoryMode) historyOperator.Fit(transitions,trainCount,diagnosticMode=="history");
        else koopman.Fit(transitions, trainCount);
        if (diagnosticMode == "rotation_decoupled") koopman.RestrictRotationCoupling();
        if (diagnosticMode == "no_pitch_bias") koopman.RemovePitchAccelerationBias();
        if (diagnosticMode == "no_horizontal_position") koopman.RemoveRotationHorizontalPosition();
        if (diagnosticMode == "no_cross_input") koopman.RemoveRotationCrossInput();
        VerifyGradients();
        modelValid = true;
        string directory = artifactDirectory;
        using (var writer = new StreamWriter(Path.Combine(directory, "edmd_operator.csv"), false))
        {
            writer.WriteLine("output,kind,input,coefficient");
            for (int row = 0; row < ModelSize; row++)
            {
                for (int col = 0; col < ModelSize; col++)
                    writer.WriteLine($"{row},A,{col},{ModelA[row, col].ToString("R", CultureInfo.InvariantCulture)}");
                for (int col = 0; col < AxisCount; col++)
                    writer.WriteLine($"{row},B,{col},{ModelB[row, col].ToString("R", CultureInfo.InvariantCulture)}");
            }
        }
        // Export the nominal lifted approximation and learned matrix residual
        // separately, making the K = K0 + deltaK construction reviewable.
        var nominalOperator=new FinsRovLiftedOperator();
        nominalOperator.BuildNominalPrior(mass,linearDrag,quadraticDrag,spring,bias);
        var augmentedPrior=new FinsRovHistoryOperator();
        augmentedPrior.BuildPrior(nominalOperator);
        using (var priorWriter=new StreamWriter(Path.Combine(directory,"nominal_lifted_operator.csv"),false))
        using (var residualWriter=new StreamWriter(Path.Combine(directory,"edmd_residual_operator.csv"),false))
        {
            priorWriter.WriteLine("output,kind,input,coefficient");
            residualWriter.WriteLine("output,kind,input,coefficient");
            for (int row=0;row<ModelSize;row++)
                foreach (string kind in new[]{"A","B"})
                    for (int col=0;col<(kind=="A"?ModelSize:6);col++)
                    {
                        double prior=HistoryMode ? (kind=="A"?augmentedPrior.A[row,col]:augmentedPrior.B[row,col]) :
                            (kind=="A"?nominalOperator.A[row,col]:nominalOperator.B[row,col]);
                        double learned=kind=="A"?ModelA[row,col]:ModelB[row,col];
                        string prefix=row+","+kind+","+col+",";
                        priorWriter.WriteLine(prefix+prior.ToString("R",CultureInfo.InvariantCulture));
                        residualWriter.WriteLine(prefix+(learned-prior).ToString("R",CultureInfo.InvariantCulture));
                    }
        }
        using (var writer = new StreamWriter(Path.Combine(directory, "edmd_validation.csv"), false))
        {
            writer.WriteLine("horizon_steps,model,normalized_state_rmse,horizontal_position_rmse_m,depth_rmse_m,tilt_axes_rmse_rad,samples");
            foreach (int horizon in new[] { 1, Horizon })
                foreach (bool learned in new[] { false, true })
                {
                    double stateSquared = 0d, horizontalSquared = 0d, depthSquared = 0d;
                    double tiltSquared = 0d;
                    int count = 0;
                    for (int start = trainCount; start + horizon <= transitions.Count; start++)
                    {
                        int segment = transitions[start].Segment;
                        if (transitions[start + horizon - 1].Segment != segment) continue;
                        var previous = start>0 && transitions[start-1].Segment==segment ?
                            transitions[start-1].Wrench : new float[6];
                        double[] z = LiftState(transitions[start].Pose,transitions[start].Velocity,previous);
                        for (int k = 0; k < horizon; k++)
                        {
                            float[] command = transitions[start + k].Wrench;
                            z = PredictionStep(z, FinsRovLiftedOperator.Input(command), learned, null, null);
                        }
                        var actual = transitions[start + horizon - 1];
                        double ex = FinsRovLiftedOperator.DecodePose(z, 0) - actual.NextPose[0];
                        double ey = FinsRovLiftedOperator.DecodePose(z, 1) - actual.NextPose[1];
                        double ez = FinsRovLiftedOperator.DecodePose(z, 2) - actual.NextPose[2];
                        horizontalSquared += ex * ex + ez * ez;
                        depthSquared += ey * ey;
                        for (int axis = 0; axis < AxisCount; axis++)
                        {
                            double ep = (FinsRovLiftedOperator.DecodePose(z, axis) - actual.NextPose[axis]) /
                                FinsRovLiftedOperator.PoseScale[axis];
                            double ev = (FinsRovLiftedOperator.DecodeVelocity(z, axis) - actual.NextVelocity[axis]) /
                                FinsRovLiftedOperator.VelocityScale[axis];
                            stateSquared += ep * ep + ev * ev;
                        }
                        double er = FinsRovLiftedOperator.DecodePose(z, 3) - actual.NextPose[3];
                        double epitch = FinsRovLiftedOperator.DecodePose(z, 5) - actual.NextPose[5];
                        tiltSquared += er * er + epitch * epitch;
                        count++;
                    }
                    if (count == 0) throw new InvalidOperationException("No held-out EDMDc validation windows");
                    double normalized = Math.Sqrt(stateSquared / (count * 12));
                    writer.WriteLine(string.Join(",", new[] {
                        horizon.ToString(), learned ? "edmdc" : "fossen",
                        normalized.ToString("F6", CultureInfo.InvariantCulture),
                        Math.Sqrt(horizontalSquared / count).ToString("F6", CultureInfo.InvariantCulture),
                        Math.Sqrt(depthSquared / count).ToString("F6", CultureInfo.InvariantCulture),
                        Math.Sqrt(tiltSquared / count).ToString("F6", CultureInfo.InvariantCulture),
                        count.ToString(CultureInfo.InvariantCulture)
                    }));
                    Debug.Log($"FINSSIM_EDMD_MPC_VALID horizon={horizon} model={(learned ? "edmdc" : "fossen")} " +
                        $"normalizedRmse={normalized:F5} horizontalRmse={Math.Sqrt(horizontalSquared/count):F5} samples={count}");
                    if (double.IsNaN(normalized) || normalized > 100d) modelValid = false;
                }
        }
        identifyCsv.Flush();
        Debug.Log($"FINSSIM_EDMD_MPC_FIT train={trainCount} validation={transitions.Count-trainCount} valid={modelValid}");
    }

    // Shared six-input projected-gradient MPC. Sequence variables are normalized
    // wrench commands in [-1,1], and both branches use the same objective.
    void SolveMpc(float t, bool useKoopman)
    {
        var timer = System.Diagnostics.Stopwatch.StartNew();
        if (actuatorMpc) optimizationBasis=GetAllocationBasis();
        var sequence = new double[Horizon, DecisionCount];
        for (int k = 0; k < Horizon; k++)
            for (int a = 0; a < DecisionCount; a++)
                sequence[k, a] = warm[Math.Min(Horizon - 1, k + 1), a];
        var gradient = new double[Horizon, DecisionCount];
        double current = Rollout(t, sequence, useKoopman, gradient);
        Array.Clear(predictedInitialCost, 0, AxisCount);
        Array.Clear(predictedFinalCost, 0, AxisCount);
        predictedInitialCost[0] = (float)current;
        for (int iteration = 0; iteration < GradientIterations; iteration++)
        {
            var scale = new double[DecisionCount];
            for (int a = 0; a < DecisionCount; a++)
                for (int k = 0; k < Horizon; k++)
                    scale[a] = Math.Max(scale[a], Math.Abs(gradient[k, a]));
            bool accepted = false;
            double step = 0.4d;
            for (int search = 0; search < 10; search++)
            {
                var candidate = new double[Horizon, DecisionCount];
                double change = 0d;
                for (int k = 0; k < Horizon; k++)
                    for (int a = 0; a < DecisionCount; a++)
                    {
                        candidate[k, a] = Math.Max(-1d, Math.Min(1d,
                            sequence[k, a] - step * gradient[k, a] / Math.Max(1d, scale[a])));
                        change += Math.Abs(candidate[k, a] - sequence[k, a]);
                    }
                if (change < 1e-7) break;
                double next = Rollout(t, candidate, useKoopman, null);
                if (next < current - 1e-7)
                {
                    sequence = candidate;
                    current = Rollout(t, sequence, useKoopman, gradient);
                    accepted = true;
                    break;
                }
                step *= 0.5d;
            }
            if (!accepted) break;
        }
        predictedFinalCost[0] = (float)current;
        for (int k = 0; k < Horizon; k++)
            for (int a = 0; a < DecisionCount; a++) warm[k, a] = sequence[k, a];
        var selected=MapDecision(sequence,0,null);
        for (int a=0;a<6;a++) wrench[a]=(float)(selected[a]*WrenchLimit[a]);
        if (actuatorMpc)
            for (int i=0;i<8;i++) optimizedMotors[i]=(float)(sequence[0,i]*MotorScale(i,sequence[0,i]));
        timer.Stop();
        lastSolveMs = (float)timer.Elapsed.TotalMilliseconds;
    }

    double Rollout(float t, double[,] sequence, bool useKoopman, double[,] gradient)
    {
        int n = ModelSize;
        var states = new double[Horizon + 1][];
        var jacA = gradient == null ? null : new double[Horizon][,];
        var jacB = gradient == null ? null : new double[Horizon][,];
        states[0] = LiftState(stateP,stateV,historyWrench);
        double cost = 0d;
        for (int k = 0; k < Horizon; k++)
        {
            var command = MapDecision(sequence,k,null);
            if (gradient != null)
            {
                jacA[k] = new double[n, n]; jacB[k] = new double[n, AxisCount];
            }
            states[k + 1] = PredictionStep(states[k], command, useKoopman,
                gradient == null ? null : jacA[k], gradient == null ? null : jacB[k]);
            for (int i = 0; i < n; i++)
                if (double.IsNaN(states[k + 1][i]) || double.IsInfinity(states[k + 1][i]) ||
                    Math.Abs(states[k + 1][i]) > 100d)
                {
                    if (gradient != null) Array.Clear(gradient, 0, gradient.Length);
                    return 1e20;
                }
            float futureT = t + ControlDt * (k + 1);
            double terminal = k == Horizon - 1 ? 3d : 1d;
            for (int a = 0; a < AxisCount; a++)
            {
                double ep = FinsRovLiftedOperator.DecodePose(states[k + 1], a) - Reference(a, futureT);
                double ev = FinsRovLiftedOperator.DecodeVelocity(states[k + 1], a) - ReferenceVelocity(a, futureT);
                double u = command[a] * WrenchLimit[a];
                cost += terminal * (PositionWeight[a] * ep * ep + VelocityWeight[a] * ev * ev)
                    + EffortWeight[a] * u * u;
            }
            if (actuatorMpc)
                for (int i=0;i<8;i++) cost+=.0005d*Math.Pow(sequence[k,i]*MotorScale(i,sequence[k,i]),2);
        }
        if (gradient == null) return cost;
        var adjoint = new double[n];
        for (int k = Horizon - 1; k >= 0; k--)
        {
            float futureT = t + ControlDt * (k + 1);
            double terminal = k == Horizon - 1 ? 3d : 1d;
            var nextAdjoint = (double[])adjoint.Clone();
            for (int a = 0; a < AxisCount; a++)
            {
                double ep = FinsRovLiftedOperator.DecodePose(states[k + 1], a) - Reference(a, futureT);
                double ev = FinsRovLiftedOperator.DecodeVelocity(states[k + 1], a) - ReferenceVelocity(a, futureT);
                nextAdjoint[1 + a] += 2d * terminal * PositionWeight[a] * ep * FinsRovLiftedOperator.PoseScale[a];
                nextAdjoint[7 + a] += 2d * terminal * VelocityWeight[a] * ev * FinsRovLiftedOperator.VelocityScale[a];
            }
            var mapping=new double[6,DecisionCount];
            var command=MapDecision(sequence,k,mapping);
            var commandGradient=new double[6];
            for (int a = 0; a < AxisCount; a++)
            {
                double g = 2d * EffortWeight[a] * WrenchLimit[a] * WrenchLimit[a] * command[a];
                for (int i = 0; i < n; i++)
                    g += jacB[k][i, a] * nextAdjoint[i];
                commandGradient[a]=g;
            }
            for (int i=0;i<DecisionCount;i++)
            {
                gradient[k,i]=0d;
                for (int a=0;a<6;a++) gradient[k,i]+=mapping[a,i]*commandGradient[a];
                if (actuatorMpc) gradient[k,i]+=.001d*sequence[k,i]*Math.Pow(MotorScale(i,sequence[k,i]),2);
            }
            var previousAdjoint = new double[n];
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++)
                    previousAdjoint[j] += jacA[k][i, j] * nextAdjoint[i];
            adjoint = previousAdjoint;
        }
        return cost;
    }

    double[] NominalStep(double[] z, double[] command, double[,] jacA, double[,] jacB)
    {
        var next = new double[ModelSize];
        next[0] = 1d;
        if (HistoryMode)
            for (int a=0;a<6;a++) { next[19+a]=command[a]; if (jacB!=null) jacB[19+a,a]=1d; }
        for (int a = 0; a < AxisCount; a++)
        {
            int pi = 1 + a, vi = 7 + a;
            double p = FinsRovLiftedOperator.DecodePose(z, a);
            double v = FinsRovLiftedOperator.DecodeVelocity(z, a);
            double u = command[a] * WrenchLimit[a];
            double acceleration = (u - linearDrag[a] * v - quadraticDrag[a] * v * Math.Abs(v)
                - spring[a] * p + bias[a]) / mass[a];
            double nextV = v + ControlDt * acceleration;
            double nextP = p + ControlDt * nextV;
            next[pi] = (nextP - FinsRovLiftedOperator.PoseCenter[a]) /
                FinsRovLiftedOperator.PoseScale[a];
            next[vi] = nextV / FinsRovLiftedOperator.VelocityScale[a];
            next[13 + a] = next[vi] * Math.Abs(next[vi]);
            if (jacA == null) continue;
            double vp = -ControlDt * spring[a] * FinsRovLiftedOperator.PoseScale[a] /
                (mass[a] * FinsRovLiftedOperator.VelocityScale[a]);
            double vv = 1d - ControlDt * (linearDrag[a] + 2d * quadraticDrag[a] * Math.Abs(v)) / mass[a];
            double vu = ControlDt * WrenchLimit[a] /
                (mass[a] * FinsRovLiftedOperator.VelocityScale[a]);
            double factor = ControlDt * FinsRovLiftedOperator.VelocityScale[a] /
                FinsRovLiftedOperator.PoseScale[a];
            jacA[vi, pi] = vp; jacA[vi, vi] = vv;
            jacA[pi, pi] = 1d + factor * vp;
            jacA[pi, vi] = factor * vv;
            jacB[vi, a] = vu; jacB[pi, a] = factor * vu;
            double qDerivative = 2d * Math.Abs(next[vi]);
            jacA[13 + a, pi] = qDerivative * vp;
            jacA[13 + a, vi] = qDerivative * vv;
            jacB[13 + a, a] = qDerivative * vu;
        }
        return next;
    }

    double[] PredictionStep(double[] z, double[] command, bool learned, double[,] ja, double[,] jb)
    {
        if (!learned) return NominalStep(z, command, ja, jb);
        if (comparisonMethod != "koopman") return ReliftedStep(z, command, ja, jb);
        var next = HistoryMode ? historyOperator.Step(z,command) : koopman.Step(z, command);
        if (ja != null)
        {
            Array.Copy(ModelA, ja, ModelA.Length);
            Array.Copy(ModelB, jb, ModelB.Length);
        }
        if (diagnosticMode == "translation")
        {
            var na = ja == null ? null : new double[FinsRovLiftedOperator.Lifted, FinsRovLiftedOperator.Lifted];
            var nb = jb == null ? null : new double[FinsRovLiftedOperator.Lifted, AxisCount];
            var nominal = NominalStep(z, command, na, nb);
            for (int a = 3; a < AxisCount; a++)
                foreach (int row in new[] { 1 + a, 7 + a, 13 + a })
                {
                    next[row] = nominal[row];
                    if (ja == null) continue;
                    for (int j = 0; j < FinsRovLiftedOperator.Lifted; j++) ja[row, j] = na[row, j];
                    for (int j = 0; j < AxisCount; j++) jb[row, j] = nb[row, j];
                }
        }
        return next;
    }

    // Static residual / refitted Fossen: the velocity map is evaluated on
    // physical state on EVERY predicted step. No independently propagated
    // nonlinear observable is used. The same analytic optimizer is retained.
    double[] ReliftedStep(double[] z, double[] command, double[,] ja, double[,] jb)
    {
        var physical=(double[])z.Clone();
        for (int a=0;a<6;a++) physical[13+a]=z[7+a]*Math.Abs(z[7+a]);
        var next=new double[ModelSize]; next[0]=1d;
        for (int a=0;a<6;a++)
        {
            int vi=7+a,pi=1+a,qi=13+a;
            for (int j=0;j<ModelSize;j++) next[vi]+=ModelA[vi,j]*physical[j];
            for (int j=0;j<6;j++) next[vi]+=ModelB[vi,j]*command[j];
            double factor=ControlDt*FinsRovLiftedOperator.VelocityScale[a]/FinsRovLiftedOperator.PoseScale[a];
            next[pi]=z[pi]+factor*next[vi];
            next[qi]=next[vi]*Math.Abs(next[vi]);
            if (ja!=null)
            {
                for (int j=0;j<ModelSize;j++)
                {
                    double d=j>=13&&j<19?0d:ModelA[vi,j];
                    if (j>=7&&j<13) d+=2d*Math.Abs(z[j])*ModelA[vi,j+6];
                    ja[vi,j]=d; ja[pi,j]=(j==pi?1d:0d)+factor*d;
                    ja[qi,j]=2d*Math.Abs(next[vi])*d;
                }
                for (int j=0;j<6;j++)
                {
                    jb[vi,j]=ModelB[vi,j]; jb[pi,j]=factor*ModelB[vi,j];
                    jb[qi,j]=2d*Math.Abs(next[vi])*ModelB[vi,j];
                }
            }
            if (HistoryMode) {next[19+a]=command[a]; if(jb!=null) jb[19+a,a]=1d;}
        }
        return next;
    }

    double[] LiftState(float[] pose,float[] velocity,float[] previousInput) =>
        HistoryMode ? FinsRovHistoryOperator.Lift(pose,velocity,previousInput) :
        FinsRovLiftedOperator.Lift(pose,velocity);

    double MotorScale(int i,double value) => value>=0 ? Mathf.Abs(thrusters[i].MaxForwardForceN) : Mathf.Abs(thrusters[i].MaxReverseForceN);

    double[] MapDecision(double[,] sequence,int k,double[,] jacobian)
    {
        var command=new double[6];
        if (!actuatorMpc)
        {
            for (int a=0;a<6;a++) { command[a]=sequence[k,a]; if (jacobian!=null) jacobian[a,a]=1d; }
            return command;
        }
        for (int i=0;i<8;i++)
            for (int a=0;a<6;a++)
            {
                double factor=optimizationBasis[a,i]*MotorScale(i,sequence[k,i])/WrenchLimit[a];
                command[a]+=factor*sequence[k,i];
                if (jacobian!=null) jacobian[a,i]=factor;
            }
        return command;
    }

    void LoadFrozenModel(string path)
    {
        var lines=File.ReadAllLines(path);
        if (lines.Length!=1+ModelSize*(ModelSize+6)) throw new InvalidOperationException("Frozen operator size mismatch");
        for (int i=1;i<lines.Length;i++)
        {
            var row=lines[i].Split(',');
            int output=int.Parse(row[0]),input=int.Parse(row[2]);
            double coefficient=double.Parse(row[3],CultureInfo.InvariantCulture);
            if (double.IsNaN(coefficient)||double.IsInfinity(coefficient)) throw new InvalidOperationException("Nonfinite frozen coefficient");
            if (row[1]=="A") ModelA[output,input]=coefficient; else ModelB[output,input]=coefficient;
        }
        Debug.Log("FINSSIM_EDMD_FROZEN_MODEL "+path);
    }

    void VerifyGradients()
    {
        if (actuatorMpc) optimizationBasis=GetAllocationBasis();
        var savedP = (float[])stateP.Clone();
        var savedV = (float[])stateV.Clone();
        for (int a = 0; a < AxisCount; a++)
        {
            stateP[a] = FinsRovLiftedOperator.PoseCenter[a] + 0.03f;
            stateV[a] = 0.012f;
        }
        double worst = 0d;
        foreach (bool learned in new[] { false, true })
        {
            var sequence = new double[Horizon, DecisionCount];
            for (int k = 0; k < Horizon; k++)
                for (int a = 0; a < DecisionCount; a++) sequence[k,a] = 0.02d * Math.Sin(k+a+1);
            var gradient = new double[Horizon, DecisionCount];
            Rollout(3f, sequence, learned, gradient);
            for (int k = 0; k < Horizon; k++)
                for (int a = 0; a < DecisionCount; a++)
                {
                    double old = sequence[k,a], h = 1e-6;
                    sequence[k,a] = old+h;
                    double plus = Rollout(3f, sequence, learned, null);
                    sequence[k,a] = old-h;
                    double minus = Rollout(3f, sequence, learned, null);
                    sequence[k,a] = old;
                    double numerical = (plus-minus)/(2d*h);
                    worst = Math.Max(worst, Math.Abs(numerical-gradient[k,a])/Math.Max(1d,Math.Abs(numerical)));
                }
        }
        Array.Copy(savedP, stateP, AxisCount); Array.Copy(savedV, stateV, AxisCount);
        File.WriteAllText(Path.Combine(artifactDirectory,"gradient_check.txt"), worst.ToString("R",CultureInfo.InvariantCulture));
        if (worst > 1e-4) throw new InvalidOperationException("MPC gradient check failed: " + worst);
        Debug.Log("FINSSIM_EDMD_GRADIENT_CHECK " + worst);
    }

    void Allocate(float[] desired, float[] result)
    {
        var basis=GetAllocationBasis();
        var normal = new double[AxisCount, AxisCount];
        var rhs = new double[AxisCount];
        for (int a = 0; a < AxisCount; a++)
        {
            rhs[a] = desired[a];
            for (int b = 0; b < AxisCount; b++)
            {
                normal[a, b] = a == b ? allocationRegularization : 0d;
                for (int i = 0; i < 8; i++) normal[a, b] += basis[a, i] * basis[b, i];
            }
        }
        double[] multiplier = Solve(normal, rhs);
        for (int i = 0; i < 8; i++)
        {
            double requested = 0d;
            for (int a = 0; a < AxisCount; a++) requested += basis[a, i] * multiplier[a];
            result[i] = Mathf.Clamp((float)requested,
                -Mathf.Abs(thrusters[i].MaxReverseForceN), Mathf.Abs(thrusters[i].MaxForwardForceN));
        }
        float residualSquared = 0f;
        for (int a = 0; a < AxisCount; a++)
        {
            float delivered = 0f;
            for (int i = 0; i < 8; i++) delivered += basis[a, i] * result[i];
            allocatedWrench[a] = delivered;
            residualSquared += (delivered - desired[a]) * (delivered - desired[a]);
        }
        lastAllocationResidual = Mathf.Sqrt(residualSquared);
    }

    float[,] GetAllocationBasis()
    {
        var basis = new float[AxisCount, 8];
        for (int i = 0; i < 8; i++)
        {
            Vector3 direction = thrusters[i].transform.TransformDirection(
                thrusters[i].LocalForceDirection.normalized);
            Vector3 offset = thrusters[i].transform.position - body.worldCenterOfMass;
            Vector3 torque = Vector3.Cross(offset, direction);
            basis[0, i] = direction.x; basis[1, i] = direction.y; basis[2, i] = direction.z;
            basis[3, i] = torque.x; basis[4, i] = torque.y; basis[5, i] = torque.z;
        }
        return basis;
    }

    static double[] Solve(double[,] a, double[] b)
    {
        int n = b.Length;
        for (int column = 0; column < n; column++)
        {
            int pivot = column;
            for (int row = column + 1; row < n; row++)
                if (Math.Abs(a[row, column]) > Math.Abs(a[pivot, column])) pivot = row;
            if (Math.Abs(a[pivot, column]) < 1e-10) throw new InvalidOperationException("Singular wrench/fit system");
            for (int k = column; k < n; k++)
            {
                double swap = a[column, k]; a[column, k] = a[pivot, k]; a[pivot, k] = swap;
            }
            double rhsSwap = b[column]; b[column] = b[pivot]; b[pivot] = rhsSwap;
            double inverse = 1d / a[column, column];
            for (int k = column; k < n; k++) a[column, k] *= inverse;
            b[column] *= inverse;
            for (int row = 0; row < n; row++)
            {
                if (row == column) continue;
                double factor = a[row, column];
                for (int k = column; k < n; k++) a[row, k] -= factor * a[column, k];
                b[row] -= factor * b[column];
            }
        }
        return b;
    }

    static float Gaussian(System.Random random)
    {
        double a = Math.Max(1e-9, random.NextDouble());
        return (float)(Math.Sqrt(-2d * Math.Log(a)) * Math.Cos(2d * Math.PI * random.NextDouble()));
    }

    void RecordTrial(float t)
    {
        float xError = stateP[0] - Reference(0, t);
        float zError = stateP[2] - Reference(2, t);
        float horizontalError = Mathf.Sqrt(xError * xError + zError * zError);
        if (phase == Phase.BaselineMpc)
        {
            baselineHorizontalSquared += horizontalError * horizontalError;
            baselineSamples++;
        }
        else
        {
            koopmanHorizontalSquared += horizontalError * horizontalError;
            koopmanSamples++;
        }
        float yawError = Mathf.DeltaAngle(Reference(4, t) * Mathf.Rad2Deg,
            stateP[4] * Mathf.Rad2Deg);
        float attitudeError = Mathf.Sqrt(
            Mathf.Pow(stateP[3] * Mathf.Rad2Deg, 2f) +
            Mathf.Pow(stateP[5] * Mathf.Rad2Deg, 2f) + yawError * yawError);
        float initialCost = 0f, finalCost = 0f;
        for (int i = 0; i < AxisCount; i++)
        {
            initialCost += predictedInitialCost[i];
            finalCost += predictedFinalCost[i];
        }
        string F(float value) => value.ToString("F5", CultureInfo.InvariantCulture);
        trialCsv.WriteLine(string.Join(",", new[]
        {
            phase.ToString(), F(t), F(stateP[0]), F(stateP[1]), F(stateP[2]),
            F(Reference(0,t)), F(Reference(1,t)), F(Reference(2,t)),
            F(horizontalError), F(stateP[1]-Reference(1,t)),
            F(stateP[3]*Mathf.Rad2Deg), F(stateP[5]*Mathf.Rad2Deg),
            F(stateP[4]*Mathf.Rad2Deg), F(Reference(4,t)*Mathf.Rad2Deg),
            F(Vector3.Angle(body.rotation*Vector3.up,Vector3.up)), F(attitudeError),
            F(wrench[0]), F(wrench[1]), F(wrench[2]), F(wrench[3]), F(wrench[4]), F(wrench[5]),
            F(lastMotorEffort), F(lastAllocationResidual), F(lastSolveMs),
            F(initialCost), F(finalCost), lastSaturatedMotors.ToString(CultureInfo.InvariantCulture)
        }));
        trialCsv.Flush();
    }

    void ResetVehicle()
    {
        Array.Clear(historyWrench,0,6);
        diagnosticNominal = diagnosticLearned = null;
        intervalSteps = 0;
        Array.Clear(intervalDesired, 0, AxisCount);
        Array.Clear(intervalAllocated, 0, AxisCount);
        Array.Clear(intervalApplied, 0, AxisCount);
        foreach (var motor in thrusters) motor.ResetActuatorState();
        // Clear PhysX episode state as well as the visible pose/velocity.
        body.isKinematic = true;
        body.isKinematic = false;
        body.position = new Vector3(0f, -1.1f, 0f);
        body.rotation = Quaternion.Euler(5f, 7f, -4f);
        body.linearVelocity = Vector3.zero;
        body.angularVelocity = Vector3.zero;
        body.Sleep();
        Physics.SyncTransforms();
        hydro.ResetBackendState();
        Array.Clear(wrench, 0, AxisCount);
        Array.Clear(warm, 0, warm.Length);
        nextControl = SimTime;
        nextSample = SimTime;
        previousSampleTime = -1f;
    }

    void AdvancePhase()
    {
        if (phase == Phase.Identify)
        {
            FitOperator();
            phase = Phase.SettleBaseline;
        }
        else if (phase == Phase.SettleBaseline)
        {
            phase = Phase.BaselineMpc;
            noise = new System.Random(evaluationSeed);
        }
        else if (phase == Phase.BaselineMpc)
        {
            if (isolatedScenes)
            {
                ReloadIsolatedScene(true);
                return;
            }
            phase = Phase.SettleKoopman;
        }
        else if (phase == Phase.SettleKoopman)
        {
            phase = Phase.KoopmanMpc;
            noise = new System.Random(evaluationSeed);
        }
        else
        {
            phase = Phase.Done;
            foreach (var motor in thrusters) motor.ApplyForceRequest(0f);
            double baseline = Math.Sqrt(baselineHorizontalSquared / Math.Max(1, baselineSamples));
            double koopman = Math.Sqrt(koopmanHorizontalSquared / Math.Max(1, koopmanSamples));
            Debug.Log($"FINSSIM_EDMD_MPC_RESULT baselineRmseM={baseline:F4} koopmanRmseM={koopman:F4} " +
                $"improvementPct={100d*(baseline-koopman)/Math.Max(1e-9,baseline):F2} " +
                $"samplesBaseline={baselineSamples} samplesKoopman={koopmanSamples}");
            trialCsv?.Dispose(); identifyCsv?.Dispose(); diagnosticCsv?.Dispose();
            File.WriteAllText(Path.Combine(artifactDirectory,"completed.json"),
                "{\"seed\":"+evaluationSeed+",\"baseline_samples\":"+baselineSamples+",\"compensated_samples\":"+koopmanSamples+"}");
            if (batchSeeds!=null && batchIndex+1<batchSeeds.Length)
            {
                StartNextFrozenTrial();
                return;
            }
            body.isKinematic = true;
            if (Environment.GetEnvironmentVariable("FINSSIM_EDMD_MPC_AUTOQUIT") == "1")
                Application.Quit(0);
            return;
        }
        ResetVehicle();
        phaseStart = SimTime;
        Debug.Log("FINSSIM_EDMD_MPC_PHASE " + phase);
    }

    void OnDisable()
    {
        trialCsv?.Dispose(); identifyCsv?.Dispose(); diagnosticCsv?.Dispose();
    }

    void StartNextFrozenTrial()
    {
        if (isolatedScenes)
        {
            var remaining=new int[batchSeeds.Length-batchIndex-1];
            Array.Copy(batchSeeds,batchIndex+1,remaining,0,remaining.Length);
            Environment.SetEnvironmentVariable("FINSSIM_EDMD_MPC_BATCH_SEEDS",string.Join(",",remaining));
            ReloadIsolatedScene(false);
            return;
        }
        string previousDirectory=artifactDirectory;
        evaluationSeed=batchSeeds[++batchIndex];
        artifactDirectory=Path.Combine(batchRoot,diagnosticMode+"-seed"+evaluationSeed);
        if (Directory.Exists(artifactDirectory)) throw new IOException("Refusing to overwrite frozen trial "+artifactDirectory);
        Directory.CreateDirectory(artifactDirectory);
        File.Copy(Path.Combine(previousDirectory,"diagnostic_config.json"),Path.Combine(artifactDirectory,"diagnostic_config.json"));
        File.Copy(Path.Combine(previousDirectory,"comparison_method.txt"),Path.Combine(artifactDirectory,"comparison_method.txt"));
        File.Copy(Path.Combine(previousDirectory,"frozen_model_sha256.txt"),Path.Combine(artifactDirectory,"frozen_model_sha256.txt"));
        trialCsv=new StreamWriter(Path.Combine(artifactDirectory,"mpc_trials.csv"),false);
        trialCsv.WriteLine(File.ReadAllLines(Path.Combine(previousDirectory,"mpc_trials.csv"))[0]);
        diagnosticCsv=new StreamWriter(Path.Combine(artifactDirectory,"actuation_prediction.csv"),false);
        diagnosticCsv.WriteLine(File.ReadAllLines(Path.Combine(previousDirectory,"actuation_prediction.csv"))[0]);
        identifyCsv=new StreamWriter(Path.Combine(artifactDirectory,"mpc_identification.csv"),false);
        identifyCsv.WriteLine("segment,t_s,axis,position,velocity,requested_wrench,next_position,next_velocity");
        baselineSamples=koopmanSamples=0;
        baselineHorizontalSquared=koopmanHorizontalSquared=0;
        phase=Phase.SettleBaseline;
        body.isKinematic=false;
        ResetVehicle();
        phaseStart=SimTime;
        VerifyGradients();
        Debug.Log("FINSSIM_EDMD_FROZEN_BATCH_SEED "+evaluationSeed);
    }

    void ReloadIsolatedScene(bool learnedArm)
    {
        trialCsv?.Dispose(); identifyCsv?.Dispose(); diagnosticCsv?.Dispose();
        phase=Phase.Done;
        Environment.SetEnvironmentVariable("FINSSIM_EDMD_RESUME_LEARNED",learnedArm?"1":"0");
        Environment.SetEnvironmentVariable("FINSSIM_EDMD_MPC_DIR",batchRoot??artifactDirectory);
        Debug.Log("FINSSIM_EDMD_ISOLATED_RELOAD learned="+learnedArm);
        UnityEngine.SceneManagement.SceneManager.LoadScene(UnityEngine.SceneManagement.SceneManager.GetActiveScene().name);
    }

    // Called after all Thruster.FixedUpdate methods and before physics integrates.
    public void ObserveAppliedStep()
    {
        if (diagnosticNominal == null || !ViewerEvaluating) return;
        Vector3 force = Vector3.zero, torque = Vector3.zero;
        foreach (var motor in thrusters)
        {
            var f = motor.GetAppliedWorldForce();
            force += f;
            torque += Vector3.Cross(motor.GetAppliedForcePosition() - body.worldCenterOfMass, f);
        }
        var actual = new[] { force.x, force.y, force.z, torque.x, torque.y, torque.z };
        for (int a = 0; a < AxisCount; a++)
        {
            intervalDesired[a] += wrench[a];
            intervalAllocated[a] += allocatedWrench[a];
            intervalApplied[a] += actual[a];
        }
        intervalSteps++;
    }

    void FinishDiagnosticInterval(float t)
    {
        if (diagnosticNominal == null || intervalSteps == 0) return;
        for (int a = 0; a < AxisCount; a++)
            diagnosticCsv.WriteLine(string.Join(",", new[] {
                phase.ToString(), diagnosticStart.ToString("F6", CultureInfo.InvariantCulture),
                t.ToString("F6", CultureInfo.InvariantCulture), a.ToString(),
                (intervalDesired[a]/intervalSteps).ToString("R", CultureInfo.InvariantCulture),
                (intervalAllocated[a]/intervalSteps).ToString("R", CultureInfo.InvariantCulture),
                (intervalApplied[a]/intervalSteps).ToString("R", CultureInfo.InvariantCulture),
                stateP[a].ToString("R", CultureInfo.InvariantCulture),
                FinsRovLiftedOperator.DecodePose(diagnosticNominal,a).ToString("R", CultureInfo.InvariantCulture),
                FinsRovLiftedOperator.DecodePose(diagnosticLearned,a).ToString("R", CultureInfo.InvariantCulture),
                stateV[a].ToString("R", CultureInfo.InvariantCulture),
                FinsRovLiftedOperator.DecodeVelocity(diagnosticNominal,a).ToString("R", CultureInfo.InvariantCulture),
                FinsRovLiftedOperator.DecodeVelocity(diagnosticLearned,a).ToString("R", CultureInfo.InvariantCulture),
                intervalSteps.ToString() }));
        diagnosticCsv.Flush();
        intervalSteps = 0;
        Array.Clear(intervalDesired, 0, AxisCount);
        Array.Clear(intervalAllocated, 0, AxisCount);
        Array.Clear(intervalApplied, 0, AxisCount);
    }
}
