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
public sealed partial class FinsRovHybridBasisSmoke : MonoBehaviour
{
    enum Phase { Identify, SettleBaseline, BaselineMpc, SettleKoopman, KoopmanMpc, Done }
    [Serializable] sealed class ThrusterCurveEntry
    {
        public string unityThruster;
        public string sourceChannel;
        public string sourceMotorId;
        public double negativeCoefficientNPerCommandSquared;
        public double positiveCoefficientNPerCommandSquared;
    }
    [Serializable] sealed class ThrusterCurveConfig
    {
        public int schemaVersion;
        public string model;
        public string input;
        public string output;
        public bool duplicateHorizontalCurvesToVertical;
        public ThrusterCurveEntry[] thrusters;
    }
    const int AxisCount = 6;
    const int MaxHorizon = 12;
    int Horizon = MaxHorizon;
    int learnedHorizon = MaxHorizon;
    const int GradientIterations = 32;
    // DE47 ControlDt is supplied by FinsRovRateStudy.
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
    readonly float[] nominalForwardForceN = new float[8];
    readonly float[] nominalReverseForceN = new float[8];
    readonly ThrusterCurveEntry[] measuredThrusterCurves = new ThrusterCurveEntry[8];
    string thrusterCurvePath;
    string thrusterCurveSha256;
    readonly float[] mass = new float[AxisCount];
    readonly float[] linearDrag = new float[AxisCount];
    readonly float[] quadraticDrag = new float[AxisCount];
    readonly float[] spring = new float[AxisCount];
    readonly float[] bias = new float[AxisCount];
    readonly float[] wrench = new float[AxisCount];
    readonly float[] motorRequest = new float[8];
    readonly double[,] warm = new double[MaxHorizon, 8];
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
    StreamWriter actuatorActionCsv;
    System.Random noise;
    Phase phase;
    string artifactDirectory;
    double phaseStart;
    double nextControl;
    double nextSample;
    double previousSampleTime = -1d;
    double SimTime => Time.fixedTimeAsDouble;
    float lastSolveMs;
    bool useSqp;
    int sqpIterationLimit = GradientIterations;
    FinsRovSqpSolver.Result lastSqp;
    bool lastWarmStartRecovered;
    StreamWriter solverCsv;
    float lastAllocationResidual;
    float lastMotorEffort;
    int lastSaturatedMotors;
    int baselineSamples;
    int koopmanSamples;
    int baselineMidpointChartRejects;
    int koopmanMidpointChartRejects;
    int evaluationSeed;
    int identifySegment;
    int trainCount;
    bool modelValid;
    bool studyInitialized;
    double baselineHorizontalSquared;
    double koopmanHorizontalSquared;
    string diagnosticMode = "full";
    readonly FinsRovHistoryOperator historyOperator = new FinsRovHistoryOperator();
    readonly float[] historyWrench = new float[6];
    readonly float[] bodyHistoryWrench = new float[6];
    bool BodyFrameModel => HybridMode && hybrid != null && hybrid.BodyVelocity;
    void UpdateModelHistory()
    {
        if(bodyStudy && BodyFrameModel) { FinishRateHistory();return; }
        Array.Copy(BodyFrameModel?bodyHistoryWrench:wrench,historyWrench,6);
    }
    bool HistoryMode => diagnosticMode == "history" || diagnosticMode == "invariant";
    readonly FinsRovHybridOperator hybrid=new FinsRovHybridOperator();
    bool HybridMode=>hybrid.Data!=null;
    float hydroScale;
    int hydroFamily;
    int trainingSeed=24001;
    static float Tanh(float x)=>(float)Math.Tanh(x);
    int ModelSize => HybridMode ? hybrid.Size : HistoryMode ? FinsRovHistoryOperator.Size : FinsRovLiftedOperator.Lifted;
    double[,] ModelA => HybridMode ? hybrid.A : HistoryMode ? historyOperator.A : koopman.A;
    double[,] ModelB => HybridMode ? hybrid.B : HistoryMode ? historyOperator.B : koopman.B;
    float identifyDuration = IdentifySeconds;
    float segmentDuration = SegmentSeconds;
    bool expandedTraining;
    bool actuatorMpc;
    bool actuatorAwareMpc;
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
    readonly float[] auditTruePose = new float[AxisCount];
    readonly float[] auditTrueVelocity = new float[AxisCount];
    int actuatorAuditRows;
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
    public Vector2 ViewerReferenceAt(float t) => new Vector2(Reference(0, t), Reference(2, t));
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
        bodyStudy=Environment.GetEnvironmentVariable("FINSSIM_BODY_STUDY")=="1";
        ConfigureRateStudy();
        string requestedHorizon=Environment.GetEnvironmentVariable("FINSSIM_DE45_MPC_HORIZON");
        if(!string.IsNullOrEmpty(requestedHorizon)) {
            if(!int.TryParse(requestedHorizon,out learnedHorizon) || (learnedHorizon!=8 && learnedHorizon!=10 && learnedHorizon!=12))
                throw new ArgumentOutOfRangeException("FINSSIM_DE45_MPC_HORIZON");
        }
        string solver=Environment.GetEnvironmentVariable("FINSSIM_HYBRID_MPC_SOLVER")??"projected_gradient";
        if(solver!="sqp" && solver!="projected_gradient")throw new InvalidOperationException("Unknown MPC solver "+solver);
        useSqp=solver=="sqp";
        if(useSqp && int.TryParse(Environment.GetEnvironmentVariable("FINSSIM_HYBRID_SQP_ITERATIONS"),out int sqpLimit)) {
            if(sqpLimit!=32 && sqpLimit!=128 && sqpLimit!=512) throw new ArgumentOutOfRangeException("FINSSIM_HYBRID_SQP_ITERATIONS");
            sqpIterationLimit=sqpLimit;
        }
        diagnosticMode = Environment.GetEnvironmentVariable("FINSSIM_EDMD_MPC_MODE") ?? "full";
        isolatedScenes=Environment.GetEnvironmentVariable("FINSSIM_EDMD_ISOLATED_SCENES")=="1";
        resumeLearnedArm=isolatedScenes && Environment.GetEnvironmentVariable("FINSSIM_EDMD_RESUME_LEARNED")=="1";
        comparisonMethod = Environment.GetEnvironmentVariable("FINSSIM_EDMD_COMPARISON_METHOD") ?? "koopman";
        float.TryParse(Environment.GetEnvironmentVariable("FINSSIM_HYBRID_HYDRO_SCALE"), NumberStyles.Float,CultureInfo.InvariantCulture,out hydroScale);
        if(hydroScale<0 || hydroScale>2) throw new Exception("Hydro scale out of range");
        if(!int.TryParse(Environment.GetEnvironmentVariable("FINSSIM_HYBRID_HYDRO_FAMILY"),NumberStyles.Integer,CultureInfo.InvariantCulture,out hydroFamily))hydroFamily=0;
        if(hydroFamily<0 || hydroFamily>2)throw new ArgumentOutOfRangeException("FINSSIM_HYBRID_HYDRO_FAMILY");
        if(int.TryParse(Environment.GetEnvironmentVariable("FINSSIM_HYBRID_TRAIN_SEED"),out int ts)) trainingSeed=ts;
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
        actuatorAwareMpc = Environment.GetEnvironmentVariable("FINSSIM_HYBRID_ACTUATOR_AWARE_MPC") == "1";
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
        float bodySeconds=0f;
        if(bodyStudy && float.TryParse(Environment.GetEnvironmentVariable("FINSSIM_BODY_EVAL_SECONDS"),NumberStyles.Float,CultureInfo.InvariantCulture,out bodySeconds)) {
            if(bodySeconds<12.8f || bodySeconds>80f)throw new ArgumentOutOfRangeException("FINSSIM_BODY_EVAL_SECONDS");
            evaluationDuration=bodySeconds;
        }
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
        InitializeCurrentModel();
        foreach (var candidate in GetComponentsInChildren<Thruster>(true))
            for (int i = 0; i < Names.Length; i++)
                if (candidate.name == Names[i]) thrusters[i] = candidate;
        foreach (var motor in thrusters)
            if (motor == null) throw new InvalidOperationException("Eight FinsROV thrusters were not resolved");
        LoadMeasuredThrusterCurves();
        if (actuatorAwareMpc && (!actuatorMpc || measuredThrusterCurves[0] == null))
            throw new InvalidOperationException("Actuator-aware MPC requires actuator-MPC mode and a measured thruster curve");

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
        if(bodyStudy)bodyTask=Environment.GetEnvironmentVariable("FINSSIM_BODY_TASK")??"joint8";

        CaptureFixedTruth();
        if(bodyStudy)ConfigureBodyTruth();
        if (TryStartInputResponse()) return;
        artifactDirectory = Environment.GetEnvironmentVariable("FINSSIM_EDMD_MPC_DIR");
        if (string.IsNullOrEmpty(artifactDirectory))
            artifactDirectory = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/edmd-mpc-smoke"));
        if (batchSeeds!=null)
        {
            batchRoot=artifactDirectory;
            artifactDirectory=Path.Combine(batchRoot,diagnosticMode+"-seed"+batchSeeds[0]);
        }
        Directory.CreateDirectory(artifactDirectory);
        OpenCurrentArtifacts(artifactDirectory, resumeLearnedArm);
        if(bodyStudy)File.WriteAllText(Path.Combine(artifactDirectory,"body_nominal_parameters.json"),JsonUtility.ToJson(new ResponseNominal{mass=mass,linearDrag=linearDrag,quadraticDrag=quadraticDrag,spring=spring,bias=bias}));
        InitializeObservationProbe();
        File.WriteAllText(Path.Combine(artifactDirectory,"comparison_method.txt"), comparisonMethod);
        File.WriteAllText(Path.Combine(artifactDirectory,"mpc_solver.txt"),useSqp?"sqp,max_outer_iterations="+sqpIterationLimit:"projected_gradient");
        File.WriteAllText(Path.Combine(artifactDirectory, "diagnostic_config.json"),
            "{\"mode\":\"" + diagnosticMode + "\",\"allocation_regularization\":" +
            allocationRegularization.ToString("R", CultureInfo.InvariantCulture) +
            ",\"fixed_dt\":" + Time.fixedDeltaTime.ToString("R", CultureInfo.InvariantCulture) +
            ",\"expanded_training\":"+(expandedTraining?"true":"false")+
            ",\"actuator_mpc\":"+(actuatorMpc?"true":"false")+
            ",\"actuator_aware_mpc\":"+(actuatorAwareMpc?"true":"false")+
            ",\"actuator_state_prediction\":"+(actuatorStatePrediction?"true":"false")+
            ",\"period_scale\":"+periodScale.ToString("R",CultureInfo.InvariantCulture)+
            ",\"amplitude_scale\":"+amplitudeScale.ToString("R",CultureInfo.InvariantCulture)+
            ",\"mismatch_scale\":"+mismatchScale.ToString("R",CultureInfo.InvariantCulture)+
            ",\"measured_thruster_curve\":"+(measuredThrusterCurves[0]!=null?"true":"false")+
            ",\"frozen_model\":"+(!string.IsNullOrEmpty(frozenModelPath)?"true":"false")+"}");
        if (measuredThrusterCurves[0] != null)
        {
            File.Copy(thrusterCurvePath, Path.Combine(artifactDirectory, "thruster_curve_config.json"), true);
            File.WriteAllText(Path.Combine(artifactDirectory, "thruster_curve_sha256.txt"), thrusterCurveSha256);
        }
        diagnosticCsv = new StreamWriter(Path.Combine(artifactDirectory, "actuation_prediction.csv"), resumeLearnedArm);
        if (!resumeLearnedArm) diagnosticCsv.WriteLine("phase,t_start,t_end,axis,desired_mean,allocated_mean,applied_mean,next_pose,nominal_pose,learned_pose,next_velocity,nominal_velocity,learned_velocity,physics_steps");
        gameObject.AddComponent<FinsRovHybridActuationObserver>().Owner = this;
        if (resumeLearnedArm)
            foreach (var line in File.ReadAllLines(Path.Combine(artifactDirectory,"mpc_trials.csv")))
            {
                var cells=line.Split(','); if(cells[0]!="BaselineMpc") continue;
                double error=double.Parse(cells[8],CultureInfo.InvariantCulture);
                baselineHorizontalSquared+=error*error; baselineSamples++;
            }
        trialCsv = new StreamWriter(Path.Combine(artifactDirectory, "mpc_trials.csv"), resumeLearnedArm);
        if (!resumeLearnedArm) trialCsv.WriteLine("phase,t_s,x_m,y_m,z_m,ref_x_m,ref_y_m,ref_z_m,horizontal_error_m,depth_error_m,roll_deg,pitch_deg,yaw_deg,ref_yaw_deg,tilt_deg,attitude_error_deg,fx_n,fy_n,fz_n,tx_nm,ty_nm,tz_nm,motor_effort_n,allocation_residual,solver_ms,predicted_initial_cost,predicted_final_cost,saturated_motors");
        string actionAuditPath=Path.Combine(artifactDirectory,"motor_actions.csv");
        if(resumeLearnedArm&&!File.Exists(actionAuditPath))throw new FileNotFoundException("Baseline motor action audit is missing before learned-arm append",actionAuditPath);
        actuatorActionCsv=new StreamWriter(actionAuditPath,resumeLearnedArm);
        if(!resumeLearnedArm)WriteMotorActionHeader();
        identifyCsv = new StreamWriter(Path.Combine(artifactDirectory, "mpc_identification.csv"), resumeLearnedArm);
        if (!resumeLearnedArm) identifyCsv.WriteLine("segment,t_s,axis,position,velocity,requested_wrench,next_position,next_velocity");
        phase = Phase.Identify;
        phaseStart = SimTime;
        nextControl = SimTime;
        nextSample = SimTime;
        noise = new System.Random(trainingSeed);
        File.WriteAllText(Path.Combine(artifactDirectory,"hybrid_scenario.json"),
            "{\"hydro_scale\":"+hydroScale.ToString("R",CultureInfo.InvariantCulture)+",\"hydro_family\":"+hydroFamily+",\"training_seed\":"+trainingSeed+"}");
        if (!int.TryParse(Environment.GetEnvironmentVariable("FINSSIM_EDMD_MPC_EVAL_SEED"),
            NumberStyles.Integer, CultureInfo.InvariantCulture, out evaluationSeed))
            evaluationSeed = 9091;
        if (batchSeeds!=null) evaluationSeed=batchSeeds[0];
        ResetCurrentProcess(trainingSeed);
        koopman.BuildNominalPrior(mass, linearDrag, quadraticDrag, spring, bias);
        historyOperator.BuildPrior(koopman);
        if (!string.IsNullOrEmpty(frozenModelPath))
        {
            LoadFrozenModel(frozenModelPath);
            phase=Phase.SettleKoopman; // DE83 single learned-only episode
            VerifyGradients();
            ResetVehicle();
            using (var sha=System.Security.Cryptography.SHA256.Create())
                File.WriteAllText(Path.Combine(artifactDirectory,"frozen_model_sha256.txt"),
                    BitConverter.ToString(sha.ComputeHash(File.ReadAllBytes(frozenModelPath))).Replace("-","").ToLowerInvariant());
        }
        Debug.Log("FINSSIM_EDMD_MPC_START " + artifactDirectory);
        Debug.Log("FINSSIM_EDMD_MPC_EVAL_SEED " + evaluationSeed);
        Debug.Log($"FINSSIM_EDMD_MPC_MODEL mass={string.Join(",", mass)} spring={spring[3]:F3} buoyancyBias={bias[1]:F3}");
        if(Environment.GetEnvironmentVariable("FINSSIM_RESIDUAL_INTERFACE_ONLY")=="1") {
            if(hybrid.Residual==null)throw new InvalidOperationException("Residual interface validation requires schema 7");
            File.WriteAllText(Path.Combine(artifactDirectory,"residual_interface_passed.json"),"{\"schema\":7,\"status\":\"passed\"}");
            phase=Phase.Done;
            if(!TryAdvanceBodyQueue())Application.Quit(0);
            return;
        }
        studyInitialized=true;
    }

    void FixedUpdate()
    {
        if (responseActive) { InputResponseFixedUpdate(); return; }
        // A failed frozen-model initialization must not enter Identify by default.
        if (!studyInitialized) return;
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
        StepCurrentModel();
        if (phase == Phase.Identify)
        {
            int segment = Mathf.Min(3, Mathf.FloorToInt(t / segmentDuration));
            if (segment > identifySegment)
            {
                if (segment == 3) trainCount = transitions.Count;
                identifySegment = segment;
                ResetVehicle();
                noise = new System.Random(trainingSeed + 271 * segment);
                Debug.Log($"FINSSIM_EDMD_MPC_IDENT_SEGMENT {segment} pairs={transitions.Count}");
            }
        }
        BeginRateCycle();
        CaptureObservationTruth();
        ReadState(stateP, stateV);
        // Unmodelled, state-dependent fluid load, never exposed as a controller input.
        // This is a synthetic learnability test, not a validated CFD model.
        if(hydroScale!=0 && phase!=Phase.SettleBaseline && phase!=Phase.SettleKoopman)
        {
            Vector3 v=body.linearVelocity, w=body.angularVelocity;
            float phi=stateP[3],theta=stateP[5],depth=stateP[1]+1.2f;
            Vector3 force,torque;
            if(hydroFamily==0)
            {
                force=new Vector3(
                    -2.2f*Tanh(7f*v.x+2f*v.z)*Mathf.Abs(v.z)*5f,
                    .65f*Mathf.Sin(5f*depth)*Tanh(6f*v.x),
                    -1.8f*Tanh(6f*v.z-2f*v.x)*Mathf.Abs(v.x)*5f);
                torque=new Vector3(
                    -.18f*Tanh(8f*w.x+3f*v.z)+.10f*Mathf.Sin(4f*theta)*Tanh(8f*v.x),
                    -.12f*Tanh(7f*w.y)*Mathf.Abs(v.x)*5f,
                    -.20f*Tanh(8f*w.z-3f*v.x)+.10f*Mathf.Sin(4f*phi)*Tanh(8f*v.z));
            }
            else if(hydroFamily==1)
            {
                // Current-relative crossflow drag: a distinct, physically
                // interpretable family with velocity offsets and cross terms.
                Vector3 rel=v-new Vector3(.08f,0f,-.04f);
                force=new Vector3(
                    -1.2f*rel.x*Mathf.Abs(rel.x)-.40f*rel.x*Mathf.Abs(rel.z),
                    -.30f*rel.y*Mathf.Abs(rel.y),
                    -1.0f*rel.z*Mathf.Abs(rel.z)-.50f*rel.z*Mathf.Abs(rel.x));
                torque=new Vector3(-.10f*w.x-.04f*rel.z,
                    -.08f*w.y*Mathf.Abs(rel.x),-.12f*w.z-.05f*rel.x);
            }
            else
            {
                // Saturating periodic vortex-like load, held out as a more
                // nonlinear depth/attitude/velocity-dependent disturbance.
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
                // DE48 disables diagnostics comparing different time intervals.
                InitializeObservedControl(t);
                UpdateModelHistory();
                De84AlignObservation(t);
                try { De83Observe(t,phase==Phase.KoopmanMpc); }
                catch(System.InvalidOperationException error) {
                    // Preserve a failed episode; never turn domain failure into a valid hold or fallback.
                    System.IO.File.WriteAllText(System.IO.Path.Combine(artifactDirectory,"de100_terminal_failure.json"),UnityEngine.JsonUtility.ToJson(new De100Failure{t=t,reason=error.Message,site="De83Observe",pose=(float[])stateP.Clone(),velocity=(float[])stateV.Clone()}));
                    RestoreObservationTruth();
                    foreach(var motor in thrusters)motor.ApplyForceRequest(0f);
                    enabled=false;UnityEngine.Application.Quit(20);return;
                }
                SolveMpc(t, phase == Phase.KoopmanMpc);
                diagnosticNominal=diagnosticLearned=null;
            }
            else Array.Clear(wrench, 0, AxisCount);
            nextControl = SimTime + UpdateDt;
        }
        RestoreObservationTruth();
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
                for (int i=0;i<8;i++)
                {
                    float allocatedMotorForce=actuatorAwareMpc
                        ? ThrustScale*MeasuredThrusterForce(i,motorRequest[i])
                        : motorRequest[i];
                    allocatedWrench[a]+=basis[a,i]*allocatedMotorForce;
                }
                error+=(allocatedWrench[a]-wrench[a])*(allocatedWrench[a]-wrench[a]);
            }
            lastAllocationResidual=Mathf.Sqrt(error);
        }
        else Allocate(wrench, motorRequest);
        RateDeadlineGate();
        lastMotorEffort = 0f;
        lastSaturatedMotors = 0;
        for (int i = 0; i < thrusters.Length; i++)
        {
            float force = (measuredThrusterCurves[i] == null
                ? ThrustScale * motorRequest[i]
                : ThrustScale * MeasuredThrusterForce(i, motorRequest[i])) + NoiseStdN * bodyNoiseScale * Gaussian(noise);
            thrusters[i].ApplyForceRequest(force);
            lastMotorEffort += Mathf.Abs(force);
            if (Mathf.Abs(motorRequest[i]) >= (motorRequest[i] >= 0f ?
                nominalForwardForceN[i] : nominalReverseForceN[i]) - 0.05f)
                lastSaturatedMotors++;
        }
        if ((phase == Phase.BaselineMpc || phase == Phase.KoopmanMpc) &&
            SimTime + 0.0001d >= nextSample)
        {
            RecordTrial(t);
            nextSample = SimTime + UpdateDt;
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
        if(bodyStudy && phase!=Phase.Identify)return BodyReference(axis,t,false);
        if (expandedTraining && phase == Phase.Identify)
        {
            float local=t-identifySegment*segmentDuration, shift=identifySegment*.73f+.17f*(trainingSeed%11);
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
        if(bodyStudy && phase!=Phase.Identify)return BodyReference(axis,t,true);
        if (expandedTraining && phase == Phase.Identify)
        {
            float local=t-identifySegment*segmentDuration, shift=identifySegment*.73f+.17f*(trainingSeed%11);
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
            UpdateModelHistory();
            SolveMpc(t,false);
            float trackingLocal=t-identifySegment*segmentDuration;
            if (actuatorMpc)
            {
                for (int i=0;i<8;i++) optimizedMotors[i]=Mathf.Clamp(optimizedMotors[i]+.7f*Mathf.Sin((.63f+.17f*i)*trackingLocal+identifySegment*.69f),
                    -nominalReverseForceN[i],nominalForwardForceN[i]);
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
        de57SolverHold=false;
        Horizon=learnedHorizon;
        var timer = System.Diagnostics.Stopwatch.StartNew();
        if (actuatorMpc) optimizationBasis=OptimizationAllocationBasis();
        var sequence = new double[Horizon, DecisionCount];
        BuildRateWarmStart(sequence,t);
        if (actuatorAwareMpc)
        {
            bool hasWarmStart=false;
            for(int k=0;k<Horizon && !hasWarmStart;k++)
                for(int a=0;a<DecisionCount;a++)
                    if(Math.Abs(sequence[k,a])>1e-7d){hasWarmStart=true;break;}
            if(!hasWarmStart) InitializeActuatorAwareWarmStart(t,sequence);
        }
        var gradient = new double[Horizon, DecisionCount];
        lastWarmStartRecovered=false;
        double current = Rollout(t, sequence, useKoopman, gradient);
        if(useSqp && !FinsRovSqpSolver.ValidEvaluation(current,gradient))
        {
            // The actuator-aware feedforward warm start can be infeasible for
            // a weak calibrated predictor over the full horizon. A zero-command
            // neutral-force trajectory is a safe feasible seed near hover; with
            // asymmetric motor maps, neutral need not be decision value zero. If
            // it fails, backtrack the original sequence toward neutral before SQP.
            var original=(double[,])sequence.Clone();
            Array.Clear(sequence,0,sequence.Length);
            if(bodyForceDecisions)
                for(int k=0;k<Horizon;k++)for(int i=0;i<DecisionCount;i++)
                    sequence[k,i]=Math.Max(-1d,Math.Min(1d,-ForceCenter(i)/ForceHalf(i)));
            current=Rollout(t,sequence,useKoopman,gradient);
            if(!FinsRovSqpSolver.ValidEvaluation(current,gradient))
            {
                for(int attempt=1;attempt<=8;attempt++)
                {
                    double scale=Math.Pow(.5d,attempt);
                    var candidate=new double[Horizon,DecisionCount];
                    for(int k=0;k<Horizon;k++)for(int a=0;a<DecisionCount;a++)
                        candidate[k,a]=sequence[k,a]+scale*(original[k,a]-sequence[k,a]);
                    double cost=Rollout(t,candidate,useKoopman,null);
                    if(FinsRovSqpSolver.ValidCost(cost))
                    {
                        Array.Copy(candidate,sequence,candidate.Length);
                        current=Rollout(t,sequence,useKoopman,gradient);
                        if(FinsRovSqpSolver.ValidEvaluation(current,gradient))break;
                    }
                }
            }
            lastWarmStartRecovered=FinsRovSqpSolver.ValidEvaluation(current,gradient);
        }
        Array.Clear(predictedInitialCost, 0, AxisCount);
        Array.Clear(predictedFinalCost, 0, AxisCount);
        predictedInitialCost[0] = (float)current;
        if(useSqp) {
            lastSqp=SolveBodySqp(sequence,t,useKoopman,sqpIterationLimit);
            current=lastSqp.Cost;
        }
        for (int iteration = 0; !useSqp && iteration < GradientIterations; iteration++)
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
        if(bodyStudy && BodyFrameModel) {
            var r=FinsRovBodyFrame.Rotation(Array.ConvertAll(stateP,x=>(double)x));var bodyWrench=(float[])wrench.Clone();
            for(int block=0;block<6;block+=3)for(int i=0;i<3;i++){wrench[block+i]=0f;for(int j=0;j<3;j++)wrench[block+i]+=(float)(r[i,j]*bodyWrench[block+j]);}
        }
        if(BodyFrameModel)Array.Copy(FinsRovBodyFrame.ToBody(stateP,wrench),bodyHistoryWrench,6);
        if (actuatorMpc)
            for (int i=0;i<8;i++) optimizedMotors[i]=(float)(bodyForceDecisions?DecisionMotorRequest(i,sequence[0,i]):sequence[0,i]*MotorScale(i,sequence[0,i]));
        if(useSqp && (!lastSqp.Valid || !FinsRovSqpSolver.ValidCost(current))) {
            // Identical bounded last-applied-command hold for all five methods.
            // No hidden Fossen controller is substituted for a failed K/C model.
            de57SolverHold=true;Array.Clear(warm,0,warm.Length);
            Array.Copy(rateOldRequest,optimizedMotors,8);Array.Copy(rateOldRequest,motorRequest,8);
            Array.Copy(rateOldWrench,wrench,6);Array.Copy(rateOldAllocated,allocatedWrench,6);Array.Copy(rateOldHistory,bodyHistoryWrench,6);
        }
        timer.Stop();
        lastSolveMs = (float)timer.Elapsed.TotalMilliseconds;
        CapturePreflightObjective(t,useKoopman);
        if(useSqp) {
            if(solverCsv==null) {
                solverCsv=new StreamWriter(Path.Combine(artifactDirectory,"sqp_diagnostics.csv"),true);
                if(solverCsv.BaseStream.Length==0)solverCsv.WriteLine("phase,t_s,iterations,qp_sweeps,evaluations,projected_gradient,qp_residual,status,initial_cost,final_cost,warmstart_recovered,solver_ms,horizon_steps,decision_variables,valid_solution,invalid_evaluations,hessian_repairs,qp_retries,solver_hold");
            }
            solverCsv.WriteLine(string.Join(",",phase.ToString(),t.ToString("R",CultureInfo.InvariantCulture),lastSqp.Iterations,lastSqp.QpSweeps,lastSqp.Evaluations,
                lastSqp.ProjectedGradient.ToString("R",CultureInfo.InvariantCulture),lastSqp.QpResidual.ToString("R",CultureInfo.InvariantCulture),
                lastSqp.Status,predictedInitialCost[0].ToString("R",CultureInfo.InvariantCulture),predictedFinalCost[0].ToString("R",CultureInfo.InvariantCulture),
                lastWarmStartRecovered?"true":"false",lastSolveMs.ToString("R",CultureInfo.InvariantCulture),Horizon,Horizon*DecisionCount,
                lastSqp.Valid?"true":"false",lastSqp.InvalidEvaluations,lastSqp.HessianRepairs,lastSqp.QpRetries,de57SolverHold?"true":"false"));
            solverCsv.Flush();
        }
    }

    double Rollout(float t, double[,] sequence, bool useKoopman, double[,] gradient)
    {
        // Acceptance tests only; never enabled by the frozen formal protocol.
        if(Environment.GetEnvironmentVariable("FINSSIM_DE57_FAULT")=="invalid" && t>.05f && t<.15f) {
            if(gradient!=null)Array.Clear(gradient,0,gradient.Length);return 1e20;
        }
        int n = De69Enabled(useKoopman)?13+hybrid.Data.latent:ModelSize;
        var states = new double[Horizon + 1][];
        var jacA = gradient == null ? null : new double[Horizon][,];
        var jacB = gradient == null ? null : new double[Horizon][,];
        var actuatorWrench = actuatorStatePrediction ? new double[Horizon, AxisCount] : null;
        var actuatorJacobian = actuatorStatePrediction && gradient != null
            ? new double[Horizon, AxisCount, Horizon * DecisionCount] : null;
        if (actuatorStatePrediction) BuildActuatorStatePrediction(sequence, actuatorWrench, actuatorJacobian);
        states[0] = !useKoopman && bodyStudy ? PureFossenLift(stateP,stateV) : LiftState(stateP,stateV,historyWrench);
        if(De69Enabled(useKoopman))states[0]=FinsRovHybridOperator.De69Pack(states[0]);
        bool traceInvalid=gradient!=null && t==0f && Environment.GetEnvironmentVariable("FINSSIM_MPC_TRACE_INVALID")=="1";
        double cost = 0d;
        for (int k = 0; k < Horizon; k++)
        {
            if(bodyStudy && BodyFrameModel && BodyChartInvalid(states[k])) {
                if(traceInvalid)Debug.Log($"FINSSIM_ROLLOUT_INVALID reason=current_chart step={k} learned={useKoopman} pose={string.Join(",",Array.ConvertAll(FinsRovBodyFrame.Pose(states[k]),x=>x.ToString("R",CultureInfo.InvariantCulture)))}");
                if(gradient!=null)Array.Clear(gradient,0,gradient.Length);return 1e20;
            }
            var command = MapDecision(sequence,k,null);
            double[] predictionInput = command;
            if (actuatorStatePrediction)
            {
                predictionInput = new double[AxisCount];
                for (int a=0;a<AxisCount;a++) predictionInput[a]=actuatorWrench[k,a];
            }
            if (gradient != null)
            {
                jacA[k] = new double[n, n]; jacB[k] = new double[n, AxisCount];
            }
            try
            {
                states[k + 1] = PredictionStep(states[k], predictionInput, useKoopman,
                    gradient == null ? null : jacA[k], gradient == null ? null : jacB[k]);
            }
            catch (InvalidOperationException ex) when (
                bodyStudy && BodyFrameModel &&
                ex.Message == "Body-frame Euler chart near singularity")
            {
                // SQP may probe a trial sequence whose midpoint pose leaves the
                // Euler chart even when both endpoint states passed the guards.
                // Reject that candidate as infeasible; do not let a line-search
                // probe abort the Unity episode or corrupt collected samples.
                if (traceInvalid)
                    Debug.Log($"FINSSIM_ROLLOUT_REJECTED reason=midpoint_chart step={k + 1} learned={useKoopman}");
                if(useKoopman)koopmanMidpointChartRejects++;
                else baselineMidpointChartRejects++;
                if (gradient != null) Array.Clear(gradient, 0, gradient.Length);
                return 1e20;
            }
            if(bodyStudy && BodyFrameModel && BodyChartInvalid(states[k+1])) {
                if(traceInvalid)Debug.Log($"FINSSIM_ROLLOUT_INVALID reason=next_chart step={k+1} learned={useKoopman} pose={string.Join(",",Array.ConvertAll(FinsRovBodyFrame.Pose(states[k+1]),x=>x.ToString("R",CultureInfo.InvariantCulture)))} command={string.Join(",",Array.ConvertAll(command,x=>x.ToString("R",CultureInfo.InvariantCulture)))}");
                if(gradient!=null)Array.Clear(gradient,0,gradient.Length);return 1e20;
            }
            for (int i = 0; i < n; i++)
            {
                // The calibrated Fossen step does not consume lifted v|v|,
                // cross-product, or history slots. Large values in those
                // bookkeeping coordinates must not invalidate an otherwise
                // finite physical trajectory; the learned operator still
                // uses and therefore bounds every lifted coordinate.
                bool relevantState=i>=1 && (i<13 || useKoopman);
                if (relevantState && (double.IsNaN(states[k + 1][i]) || double.IsInfinity(states[k + 1][i]) ||
                    Math.Abs(states[k + 1][i]) > (!De69Enabled(useKoopman) && hybrid!=null && hybrid.Unified && i>=13 && i<34?10000d:100d)))
                {
                    if(traceInvalid)Debug.Log($"FINSSIM_ROLLOUT_INVALID reason=state_bound step={k+1} learned={useKoopman} index={i} value={states[k+1][i].ToString("R",CultureInfo.InvariantCulture)}");
                    if (gradient != null) Array.Clear(gradient, 0, gradient.Length);
                    return 1e20;
                }
            }
            float futureT = t + ControlDt * (k + 1);
            double terminal = De92Weight(k);
            var bodyRates=BodyFrameModel && (useKoopman || bodyStudy) ? FinsRovBodyFrame.Rates(states[k+1]) : null;
            for (int a = 0; a < AxisCount; a++)
            {
                double ep = FinsRovLiftedOperator.DecodePose(states[k + 1], a) - Reference(a, futureT);
                double ev = (bodyRates==null?FinsRovLiftedOperator.DecodeVelocity(states[k + 1], a):bodyRates[a]) - ReferenceVelocity(a, futureT);
                double u = command[a] * WrenchLimit[a];
                cost += terminal * (PositionWeight[a] * ep * ep + VelocityWeight[a] * ev * ev)
                    + EffortWeight[a] * u * u;
            }
            if (actuatorMpc)
                for (int i=0;i<8;i++) cost+=.0005d*Math.Pow(bodyForceDecisions?DecisionForce(i,sequence[k,i]):sequence[k,i]*MotorScale(i,sequence[k,i]),2);
        }
        if (gradient == null) return cost;
        Array.Clear(gradient, 0, gradient.Length);
        var adjoint = new double[n];
        for (int k = Horizon - 1; k >= 0; k--)
        {
            float futureT = t + ControlDt * (k + 1);
            double terminal = De92Weight(k);
            var nextAdjoint = (double[])adjoint.Clone();
            var bodyRates=BodyFrameModel && (useKoopman || bodyStudy) ? FinsRovBodyFrame.Rates(states[k+1]) : null;
            var bodyRateJacobian=bodyRates==null?null:FinsRovBodyFrame.RatesJacobian(states[k+1]);
            for (int a = 0; a < AxisCount; a++)
            {
                double ep = FinsRovLiftedOperator.DecodePose(states[k + 1], a) - Reference(a, futureT);
                double ev = (bodyRates==null?FinsRovLiftedOperator.DecodeVelocity(states[k + 1], a):bodyRates[a]) - ReferenceVelocity(a, futureT);
                nextAdjoint[1 + a] += 2d * terminal * PositionWeight[a] * ep * FinsRovLiftedOperator.PoseScale[a];
                if(bodyRateJacobian==null)nextAdjoint[7 + a] += 2d * terminal * VelocityWeight[a] * ev * FinsRovLiftedOperator.VelocityScale[a];
                else for(int j=0;j<n;j++)nextAdjoint[j]+=2d*terminal*VelocityWeight[a]*ev*bodyRateJacobian[a,j];
            }
            var mapping=new double[6,DecisionCount];
            var command=MapDecision(sequence,k,mapping);
            var inputGradient=new double[6];
            var effortGradient=new double[6];
            for (int a = 0; a < AxisCount; a++)
            {
                effortGradient[a] = 2d * EffortWeight[a] * WrenchLimit[a] * WrenchLimit[a] * command[a];
                for (int i = 0; i < n; i++)
                    inputGradient[a] += jacB[k][i, a] * nextAdjoint[i];
            }
            for (int i=0;i<DecisionCount;i++)
            {
                for (int a=0;a<6;a++) gradient[k,i]+=mapping[a,i]*effortGradient[a];
                if (actuatorMpc) gradient[k,i]+=bodyForceDecisions?.001d*DecisionForce(i,sequence[k,i])*ForceHalf(i):.001d*sequence[k,i]*Math.Pow(MotorScale(i,sequence[k,i]),2);
            }
            if (actuatorStatePrediction)
            {
                for (int decision = 0; decision < (k + 1) * DecisionCount; decision++)
                {
                    int decisionStep = decision / DecisionCount;
                    int decisionMotor = decision % DecisionCount;
                    double g = 0d;
                    for (int a = 0; a < AxisCount; a++)
                        g += actuatorJacobian[k, a, decision] * inputGradient[a];
                    gradient[decisionStep, decisionMotor] += g;
                }
            }
            else
            {
                for (int i=0;i<DecisionCount;i++)
                    for (int a=0;a<AxisCount;a++) gradient[k,i]+=mapping[a,i]*inputGradient[a];
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
        if(bodyStudy && BodyFrameModel)return BodyNominalStep(z,command,jacA,jacB);
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
        if(De69Enabled(learned) && z.Length==13+hybrid.Data.latent)return De92Prediction(z,command,ja,jb);
        if (!learned) return NominalStep(z, command, ja, jb);
        if(HybridMode && comparisonMethod=="koopman") return bodyStudy && BodyFrameModel?hybrid.StepBody(z,command,ja,jb):hybrid.StepWorld(z,command,ja,jb);
        if(HybridMode && comparisonMethod=="static") return PhysicalReliftStep(z,command,ja,jb);
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
        HybridMode ? hybrid.LiftWorld(pose,velocity,previousInput) : HistoryMode ? FinsRovHistoryOperator.Lift(pose,velocity,previousInput) :
        FinsRovLiftedOperator.Lift(pose,velocity);

    double MotorScale(int i,double value) => value>=0 ? nominalForwardForceN[i] : nominalReverseForceN[i];

    void LoadMeasuredThrusterCurves()
    {
        for (int i=0;i<8;i++)
        {
            nominalForwardForceN[i]=Mathf.Abs(thrusters[i].MaxForwardForceN);
            nominalReverseForceN[i]=Mathf.Abs(thrusters[i].MaxReverseForceN);
        }
        thrusterCurvePath=Environment.GetEnvironmentVariable("FINSSIM_EDMD_THRUSTER_CURVE_PATH");
        if (string.IsNullOrEmpty(thrusterCurvePath)) return;
        if (!File.Exists(thrusterCurvePath)) throw new FileNotFoundException("Measured thruster curve config was not found",thrusterCurvePath);
        var config=JsonUtility.FromJson<ThrusterCurveConfig>(File.ReadAllText(thrusterCurvePath));
        if (config==null || config.schemaVersion!=1 ||
            config.model!="signed_piecewise_quadratic_static" ||
            config.input!="normalized_target_command" ||
            config.output!="axial_force_n" ||
            !config.duplicateHorizontalCurvesToVertical ||
            config.thrusters==null || config.thrusters.Length!=8)
            throw new InvalidDataException("Unsupported measured thruster curve profile");
        for (int i=0;i<Names.Length;i++)
        {
            ThrusterCurveEntry found=null;
            foreach (var entry in config.thrusters)
                if (entry!=null && entry.unityThruster==Names[i])
                {
                    if (found!=null) throw new InvalidDataException("Duplicate curve entry for "+Names[i]);
                    found=entry;
                }
            if (found==null || string.IsNullOrEmpty(found.sourceMotorId) ||
                string.IsNullOrEmpty(found.sourceChannel) ||
                found.negativeCoefficientNPerCommandSquared<=0d ||
                found.positiveCoefficientNPerCommandSquared<=0d)
                throw new InvalidDataException("Missing or invalid curve entry for "+Names[i]);
            measuredThrusterCurves[i]=found;
            // Keep the measured physical force limits on the Thruster itself so
            // ApplyForceRequest does not clip the measured curve to the old
            // synthetic limits. MPC allocation continues to use the saved
            // nominal limits above.
            thrusters[i].MaxForwardForceN=(float)found.positiveCoefficientNPerCommandSquared;
            thrusters[i].MaxReverseForceN=-(float)found.negativeCoefficientNPerCommandSquared;
        }
        for (int i=0;i<4;i++)
        {
            var vertical=measuredThrusterCurves[i];
            var horizontal=measuredThrusterCurves[i+4];
            if (vertical.sourceMotorId!=horizontal.sourceMotorId ||
                vertical.sourceChannel!=horizontal.sourceChannel ||
                Math.Abs(vertical.negativeCoefficientNPerCommandSquared-horizontal.negativeCoefficientNPerCommandSquared)>1e-10d ||
                Math.Abs(vertical.positiveCoefficientNPerCommandSquared-horizontal.positiveCoefficientNPerCommandSquared)>1e-10d)
                throw new InvalidDataException("Vertical curve must duplicate its same-quadrant horizontal measurement: "+Names[i]);
        }
        using (var sha=System.Security.Cryptography.SHA256.Create())
            thrusterCurveSha256=BitConverter.ToString(sha.ComputeHash(File.ReadAllBytes(thrusterCurvePath))).Replace("-","").ToLowerInvariant();
        Debug.Log("FINSSIM_MEASURED_THRUSTER_CURVE "+thrusterCurvePath+" sha256="+thrusterCurveSha256);
    }

    float MeasuredThrusterForce(int i,float nominalForceRequestN)
    {
        float nominalLimit=nominalForceRequestN>=0f?nominalForwardForceN[i]:nominalReverseForceN[i];
        float command=Mathf.Clamp(nominalForceRequestN/Mathf.Max(nominalLimit,1e-6f),-1f,1f);
        var curve=measuredThrusterCurves[i];
        double coefficient=command>=0f?curve.positiveCoefficientNPerCommandSquared:curve.negativeCoefficientNPerCommandSquared;
        return (float)(coefficient*command*Math.Abs(command));
    }

    double[] MapDecision(double[,] sequence,int k,double[,] jacobian)
    {
        var command=new double[6];
        if (!actuatorMpc)
        {
            for (int a=0;a<6;a++) { command[a]=sequence[k,a]; if (jacobian!=null) jacobian[a,a]=1d; }
            return command;
        }
        for (int i=0;i<8;i++)
        {
            double motorForce, derivative;
            if(bodyForceDecisions) {
                motorForce=DecisionForce(i,sequence[k,i]);derivative=ForceHalf(i);
            }
            else if (actuatorAwareMpc && measuredThrusterCurves[i] != null)
            {
                double decision=Math.Max(-1d,Math.Min(1d,sequence[k,i]));
                var curve=measuredThrusterCurves[i];
                double coefficient=decision>=0d?curve.positiveCoefficientNPerCommandSquared:
                    curve.negativeCoefficientNPerCommandSquared;
                motorForce=ThrustScale*coefficient*decision*Math.Abs(decision);
                // At a feasible box boundary use the inward derivative.
                derivative=(Math.Abs(sequence[k,i])>1d || (!bodyStudy && Math.Abs(sequence[k,i])>=1d))?0d:
                    ThrustScale*2d*coefficient*Math.Abs(decision);
            }
            else
            {
                double scale=MotorScale(i,sequence[k,i]);
                motorForce=scale*sequence[k,i];
                derivative=scale;
            }
            for (int a=0;a<6;a++)
            {
                command[a]+=optimizationBasis[a,i]*motorForce/WrenchLimit[a];
                if (jacobian!=null)
                    jacobian[a,i]=optimizationBasis[a,i]*derivative/WrenchLimit[a];
            }
        }
        return command;
    }

    void InitializeActuatorAwareWarmStart(float t,double[,] sequence)
    {
        var basis=optimizationBasis??GetAllocationBasis();
        var normal=new double[AxisCount,AxisCount];
        for(int a=0;a<AxisCount;a++)
        {
            normal[a,a]=allocationRegularization;
            for(int b=0;b<AxisCount;b++)
                for(int i=0;i<8;i++) normal[a,b]+=basis[a,i]*basis[b,i];
        }
        for(int k=0;k<Horizon;k++)
        {
            float futureT=t+ControlDt*(k+1);
            var requested=new double[AxisCount];
            for(int a=0;a<AxisCount;a++)
            {
                double reference=Reference(a,futureT);
                double positionError=reference-stateP[a];
                if(a>=3) positionError=Mathf.DeltaAngle(stateP[a]*Mathf.Rad2Deg,(float)reference*Mathf.Rad2Deg)*Mathf.Deg2Rad;
                double velocityError=ReferenceVelocity(a,futureT)-stateV[a];
                double acceleration=2.5d*positionError+2d*velocityError;
                requested[a]=mass[a]*acceleration+linearDrag[a]*stateV[a]+
                    quadraticDrag[a]*stateV[a]*Math.Abs(stateV[a])+spring[a]*stateP[a]-bias[a];
                requested[a]=Math.Max(-WrenchLimit[a],Math.Min(WrenchLimit[a],requested[a]));
            }
            if(bodyStudy && BodyFrameModel)requested=FinsRovBodyFrame.ToBody(Array.ConvertAll(stateP,x=>(double)x),requested);
            double[] multiplier=Solve(normal,requested);
            for(int i=0;i<8;i++)
            {
                double force=0d;
                for(int a=0;a<AxisCount;a++) force+=basis[a,i]*multiplier[a];
                var curve=measuredThrusterCurves[i];
                double coefficient=force>=0d?curve.positiveCoefficientNPerCommandSquared:
                    curve.negativeCoefficientNPerCommandSquared;
                double maxForce=ThrustScale*coefficient;
                force=Math.Max(-.8d*maxForce,Math.Min(.8d*maxForce,force));
                if(bodyForceDecisions) {sequence[k,i]=Math.Max(-1d,Math.Min(1d,(force-ForceCenter(i))/ForceHalf(i)));continue;}
                double decision=Math.Sign(force)*Math.Sqrt(Math.Abs(force)/Math.Max(1e-9d,maxForce));
                sequence[k,i]=Math.Max(-.8d,Math.Min(.8d,decision));
            }
        }
    }

    void LoadFrozenModel(string path)
    {
        if(Path.GetExtension(path)==".json") {
            hybrid.Load(path);
            InstallAndAuditFossenModel();
            ValidateRateModel();
            if(hybrid.Residual!=null) {
                if(!bodyStudy||!bodyForceDecisions)throw new InvalidOperationException("Explicit residual requires native body study and affine thruster forces");
                hybrid.Residual.Bind(BodyNominalStep,mass,linearDrag,quadraticDrag,spring,bias,path);
            }
            return;
        }
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
        if (actuatorMpc) optimizationBasis=OptimizationAllocationBasis();
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
            Horizon=learnedHorizon;
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
        Horizon=learnedHorizon;
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
                -nominalReverseForceN[i], nominalForwardForceN[i]);
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
        var observedBasis=ObservationAllocationBasis();
        if(observedBasis!=null)return observedBasis;
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
        if(bodyStudy)attitudeError=Quaternion.Angle(Quaternion.Euler(Reference(3,t)*Mathf.Rad2Deg,Reference(4,t)*Mathf.Rad2Deg,Reference(5,t)*Mathf.Rad2Deg),body.rotation);
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
        De83Reset();
        De84ResetObservation();
        ResetBodyInputInterval();
        Array.Clear(historyWrench,0,6);
        Array.Clear(bodyHistoryWrench,0,6);
        diagnosticNominal = diagnosticLearned = null;
        intervalSteps = 0;
        Array.Clear(intervalDesired, 0, AxisCount);
        Array.Clear(intervalAllocated, 0, AxisCount);
        Array.Clear(intervalApplied, 0, AxisCount);
        foreach (var motor in thrusters) motor.ResetActuatorState();
        ResetBodyActuatorPrediction();
        // Clear PhysX episode state as well as the visible pose/velocity.
        body.isKinematic = true;
        body.isKinematic = false;
        body.position = new Vector3(0f, -1.1f, 0f);
        body.rotation = bodyStudy?Quaternion.Euler(5f*bodyInitialTiltScale,bodyInitialYaw,-4f*bodyInitialTiltScale):Quaternion.Euler(5f,7f,-4f);
        body.linearVelocity = Vector3.zero;
        body.angularVelocity = Vector3.zero;
        body.Sleep();
        Physics.SyncTransforms();
        hydro.ResetBackendState();
        ResetObservationProbe();
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
            if(Environment.GetEnvironmentVariable("FINSSIM_HYBRID_IDENTIFY_ONLY")=="1")
            {
                trialCsv?.Dispose(); identifyCsv?.Dispose(); diagnosticCsv?.Dispose(); CloseMotorActionAudit();
            phase=Phase.Done;
            StopCurrentModel();
            CloseCurrentArtifacts();
                File.WriteAllText(Path.Combine(artifactDirectory,"identified.json"),"{\"complete\":true}");
                Application.Quit(0); return;
            }
            phase = Phase.SettleBaseline;
        }
        else if (phase == Phase.SettleBaseline)
        {
            phase = Phase.BaselineMpc;
            noise = new System.Random(evaluationSeed);
            ResetCurrentProcess(evaluationSeed);
        }
        else if (phase == Phase.BaselineMpc)
        {
            VerifyFixedTruth("baseline_end");
            if(Environment.GetEnvironmentVariable("FINSSIM_DE62_COLLECTION")=="1"){CompleteDe62Collection();return;}
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
            ResetCurrentProcess(evaluationSeed);
        }
        else
        {
            VerifyFixedTruth("learned_end");
            phase = Phase.Done;
            StopCurrentModel();
            CloseCurrentArtifacts();
            foreach (var motor in thrusters) motor.ApplyForceRequest(0f);
            double baseline = Math.Sqrt(baselineHorizontalSquared / Math.Max(1, baselineSamples));
            double koopman = Math.Sqrt(koopmanHorizontalSquared / Math.Max(1, koopmanSamples));
            Debug.Log($"FINSSIM_EDMD_MPC_RESULT baselineRmseM={baseline:F4} koopmanRmseM={koopman:F4} " +
                $"improvementPct={100d*(baseline-koopman)/Math.Max(1e-9,baseline):F2} " +
                $"samplesBaseline={baselineSamples} samplesKoopman={koopmanSamples}");
            Debug.Log($"FINSSIM_MIDPOINT_CHART_REJECTIONS baseline={baselineMidpointChartRejects} learned={koopmanMidpointChartRejects}");
            trialCsv?.Dispose(); identifyCsv?.Dispose(); diagnosticCsv?.Dispose(); CloseMotorActionAudit();
            solverCsv?.Dispose();solverCsv=null;
            File.WriteAllText(Path.Combine(artifactDirectory,"completed.json"),
                "{\"seed\":"+evaluationSeed+",\"baseline_samples\":"+baselineSamples+",\"compensated_samples\":"+koopmanSamples+
                ",\"baseline_midpoint_chart_rejects\":"+baselineMidpointChartRejects+
                ",\"learned_midpoint_chart_rejects\":"+koopmanMidpointChartRejects+"}");
            if (batchSeeds!=null && batchIndex+1<batchSeeds.Length)
            {
                StartNextFrozenTrial();
                return;
            }
            body.isKinematic = true;
            if(bodyStudy && TryAdvanceBodyQueue())return;
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
        de82PhysicsCsv?.Dispose();de82PhysicsCsv=null;
        StopCurrentModel();
        CloseCurrentArtifacts();
        bodyTimingCsv?.Dispose();bodyTimingCsv=null;
        bodyInputCsv?.Dispose();bodyInputCsv=null;
        observationCsv?.Dispose();
        observationNoiseAudit?.Dispose();
        bodyObservationCsv?.Dispose();bodyObservationCsv=null;
        solverCsv?.Dispose();solverCsv=null;
        trialCsv?.Dispose(); identifyCsv?.Dispose(); diagnosticCsv?.Dispose(); CloseMotorActionAudit();
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
        solverCsv?.Dispose();solverCsv=null;
        CloseCurrentArtifacts();
        evaluationSeed=batchSeeds[++batchIndex];
        artifactDirectory=Path.Combine(batchRoot,diagnosticMode+"-seed"+evaluationSeed);
        if (Directory.Exists(artifactDirectory)) throw new IOException("Refusing to overwrite frozen trial "+artifactDirectory);
        Directory.CreateDirectory(artifactDirectory);
        OpenCurrentArtifacts(artifactDirectory, false);
        File.Copy(Path.Combine(previousDirectory,"diagnostic_config.json"),Path.Combine(artifactDirectory,"diagnostic_config.json"));
        File.Copy(Path.Combine(previousDirectory,"comparison_method.txt"),Path.Combine(artifactDirectory,"comparison_method.txt"));
        File.Copy(Path.Combine(previousDirectory,"mpc_solver.txt"),Path.Combine(artifactDirectory,"mpc_solver.txt"));
        File.Copy(Path.Combine(previousDirectory,"frozen_model_sha256.txt"),Path.Combine(artifactDirectory,"frozen_model_sha256.txt"));
        trialCsv=new StreamWriter(Path.Combine(artifactDirectory,"mpc_trials.csv"),false);
        trialCsv.WriteLine(File.ReadAllLines(Path.Combine(previousDirectory,"mpc_trials.csv"))[0]);
        diagnosticCsv=new StreamWriter(Path.Combine(artifactDirectory,"actuation_prediction.csv"),false);
        diagnosticCsv.WriteLine(File.ReadAllLines(Path.Combine(previousDirectory,"actuation_prediction.csv"))[0]);
        identifyCsv=new StreamWriter(Path.Combine(artifactDirectory,"mpc_identification.csv"),false);
        identifyCsv.WriteLine("segment,t_s,axis,position,velocity,requested_wrench,next_position,next_velocity");
        actuatorActionCsv=new StreamWriter(Path.Combine(artifactDirectory,"motor_actions.csv"),false);
        WriteMotorActionHeader();
        baselineSamples=koopmanSamples=0;
        baselineHorizontalSquared=koopmanHorizontalSquared=0;
        baselineMidpointChartRejects=koopmanMidpointChartRejects=0;
        phase=Phase.SettleBaseline;
        body.isKinematic=false;
        ResetVehicle();
        phaseStart=SimTime;
        VerifyGradients();
        Debug.Log("FINSSIM_EDMD_FROZEN_BATCH_SEED "+evaluationSeed);
    }

    void ReloadIsolatedScene(bool learnedArm)
    {
        solverCsv?.Dispose();solverCsv=null;
        trialCsv?.Dispose(); identifyCsv?.Dispose(); diagnosticCsv?.Dispose(); CloseMotorActionAudit();
        StopCurrentModel();
        CloseCurrentArtifacts();
        phase=Phase.Done;
        Environment.SetEnvironmentVariable("FINSSIM_EDMD_RESUME_LEARNED",learnedArm?"1":"0");
        Environment.SetEnvironmentVariable("FINSSIM_EDMD_MPC_DIR",batchRoot??artifactDirectory);
        Debug.Log("FINSSIM_EDMD_ISOLATED_RELOAD learned="+learnedArm);
        UnityEngine.SceneManagement.SceneManager.LoadScene(UnityEngine.SceneManagement.SceneManager.GetActiveScene().name);
    }

    // Called after all Thruster.FixedUpdate methods and before physics integrates.
    public void ObserveAppliedStep()
    {
        if (responseActive) { ObserveInputResponse(); return; }
        if (phase != Phase.Identify && !ViewerEvaluating) return;
        Vector3 force = Vector3.zero, torque = Vector3.zero;
        foreach (var motor in thrusters)
        {
            var f = motor.GetAppliedWorldForce();
            force += f;
            torque += Vector3.Cross(motor.GetAppliedForcePosition() - body.worldCenterOfMass, f);
        }
        // DE47 full-cycle timing ends after input estimation and audit work.
        var actual = new[] { force.x, force.y, force.z, torque.x, torque.y, torque.z };
        if(bodyStudy && actuatorStatePrediction)UpdateActuatorModelEstimate();
        de82PhysicsTime=(float)(SimTime-phaseStart);
        WriteMotorActionRow(actual);
        if(bodyStudy)ObserveBodyInput(actual);
        FinishRateCycle();
        if (diagnosticNominal == null || !ViewerEvaluating) return;
        for (int a = 0; a < AxisCount; a++)
        {
            intervalDesired[a] += wrench[a];
            intervalAllocated[a] += allocatedWrench[a];
            intervalApplied[a] += actual[a];
        }
        intervalSteps++;
    }

    void WriteMotorActionHeader()
    {
        if (actuatorActionCsv == null) return;
        var columns = new List<string> {
            "phase", "phase_time_s", "fixed_time_s", "training_seed", "evaluation_seed", "identify_segment"
        };
        for (int a=0;a<AxisCount;a++) columns.Add("pose_"+a);
        for (int a=0;a<AxisCount;a++) columns.Add("velocity_"+a);
        for (int a=0;a<AxisCount;a++) columns.Add("reference_pose_"+a);
        for (int a=0;a<AxisCount;a++) columns.Add("requested_wrench_"+a);
        for (int a=0;a<AxisCount;a++) columns.Add("allocated_wrench_"+a);
        for (int a=0;a<AxisCount;a++) columns.Add("applied_wrench_"+a);
        for (int i=0;i<8;i++) columns.Add("decision_motor_"+i);
        for (int i=0;i<8;i++) columns.Add("nominal_motor_request_n_"+i);
        for (int i=0;i<8;i++) columns.Add("target_force_request_n_"+i);
        for (int i=0;i<8;i++) columns.Add("last_force_request_n_"+i);
        for (int i=0;i<8;i++) for (int a=0;a<3;a++) columns.Add("motor_"+i+"_world_force_"+a+"_n");
        if(actuatorStatePrediction)for(int i=0;i<8;i++)columns.Add("actuator_model_estimate_n_"+i);
        columns.Add("saturated_motors");
        actuatorActionCsv.WriteLine(string.Join(",", columns.ToArray()));
        actuatorActionCsv.Flush();
    }

    void WriteMotorActionRow(float[] appliedWrench)
    {
        if (actuatorActionCsv == null) return;
        float phaseTime=(float)(SimTime-phaseStart);
        ReadState(auditTruePose,auditTrueVelocity);
        var cells=new List<string>(100) {
            phase.ToString(), phaseTime.ToString("R",CultureInfo.InvariantCulture),
            SimTime.ToString("R",CultureInfo.InvariantCulture), trainingSeed.ToString(),
            evaluationSeed.ToString(), identifySegment.ToString()
        };
        for(int a=0;a<AxisCount;a++) cells.Add(auditTruePose[a].ToString("R",CultureInfo.InvariantCulture));
        for(int a=0;a<AxisCount;a++) cells.Add(auditTrueVelocity[a].ToString("R",CultureInfo.InvariantCulture));
        for(int a=0;a<AxisCount;a++) cells.Add(Reference(a,phaseTime).ToString("R",CultureInfo.InvariantCulture));
        for(int a=0;a<AxisCount;a++) cells.Add(wrench[a].ToString("R",CultureInfo.InvariantCulture));
        for(int a=0;a<AxisCount;a++) cells.Add(allocatedWrench[a].ToString("R",CultureInfo.InvariantCulture));
        for(int a=0;a<AxisCount;a++) cells.Add(appliedWrench[a].ToString("R",CultureInfo.InvariantCulture));
        bool optimized=actuatorMpc && (ViewerEvaluating || (phase==Phase.Identify && expandedTraining));
        for(int i=0;i<8;i++) cells.Add((optimized?optimizedMotors[i]:motorRequest[i]).ToString("R",CultureInfo.InvariantCulture));
        for(int i=0;i<8;i++) cells.Add(motorRequest[i].ToString("R",CultureInfo.InvariantCulture));
        for(int i=0;i<8;i++) cells.Add(thrusters[i].TargetForceRequest.ToString("R",CultureInfo.InvariantCulture));
        for(int i=0;i<8;i++) cells.Add(thrusters[i].LastForceRequest.ToString("R",CultureInfo.InvariantCulture));
        for(int i=0;i<8;i++)
        {
            Vector3 f=thrusters[i].GetAppliedWorldForce();
            cells.Add(f.x.ToString("R",CultureInfo.InvariantCulture));
            cells.Add(f.y.ToString("R",CultureInfo.InvariantCulture));
            cells.Add(f.z.ToString("R",CultureInfo.InvariantCulture));
        }
        if(actuatorStatePrediction)for(int i=0;i<8;i++)cells.Add(ActuatorModelForceEstimate(i).ToString("R",CultureInfo.InvariantCulture));
        cells.Add(lastSaturatedMotors.ToString());
        actuatorActionCsv.WriteLine(string.Join(",",cells.ToArray()));
        if(++actuatorAuditRows%5==0) actuatorActionCsv.Flush();
    }

    void CloseMotorActionAudit()
    {
        if(actuatorActionCsv==null) return;
        actuatorActionCsv.Flush();
        actuatorActionCsv.Dispose();
        actuatorActionCsv=null;
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

public sealed partial class FinsRovHybridBasisSmoke {
 bool de69Legacy,de69Checked;
 bool De69Enabled(bool learned)=>learned && !de69Legacy && hybrid!=null && hybrid.Data.de68Compact && System.Environment.GetEnvironmentVariable("FINSSIM_DE69_COMPACT")=="1";
 [System.Serializable] public class De69CheckRecord {public bool passed;public int latentDimension=16,physicalDimension=12,bufferDimension=29,decisionDimension=64;public double stateError,jacobianError,inputJacobianError,costError,gradientError,hessianError;}
 void De69Verify(float t,double[,] sequence,bool learned){
  if(de69Checked || !De69Enabled(learned) || System.Environment.GetEnvironmentVariable("FINSSIM_DUAL_PREFLIGHT")!="1")return;
  de69Checked=true;int d=hybrid.Data.latent,n=13+d,large=36+d;var report=new De69CheckRecord();report.latentDimension=d;report.bufferDimension=n;De81EncodingTiming();var old=LiftState(stateP,stateV,historyWrench);var small=FinsRovHybridOperator.De69Pack(old);
  for(int k=0;k<Horizon;k++){
   var command=MapDecision(sequence,k,null);var a=new double[large,large];var b=new double[large,6];var ca=new double[n,n];var cb=new double[n,6];
   old=hybrid.StepBody(old,command,a,b);small=hybrid.De69Step(small,command,ca,cb);var packed=FinsRovHybridOperator.De69Pack(old);
   for(int i=0;i<n;i++){
    int oi=i<13?i:i+23;report.stateError=System.Math.Max(report.stateError,System.Math.Abs(packed[i]-small[i]));
    for(int j=0;j<n;j++){int oj=j<13?j:j+23;report.jacobianError=System.Math.Max(report.jacobianError,System.Math.Abs(a[oi,oj]-ca[i,j]));}
    for(int j=0;j<6;j++)report.inputJacobianError=System.Math.Max(report.inputJacobianError,System.Math.Abs(b[oi,j]-cb[i,j]));
   }
  }
  var go=new double[Horizon,DecisionCount];var gc=new double[Horizon,DecisionCount];
  de69Legacy=true;double co=Rollout(t,sequence,learned,go);var ho=BodyGaussNewton(t,sequence,learned);
  de69Legacy=false;double cc=Rollout(t,sequence,learned,gc);var hc=BodyGaussNewton(t,sequence,learned);
  report.costError=System.Math.Abs(co-cc)/System.Math.Max(1,System.Math.Abs(co));
  for(int i=0;i<Horizon;i++)for(int j=0;j<DecisionCount;j++)report.gradientError=System.Math.Max(report.gradientError,System.Math.Abs(go[i,j]-gc[i,j])/System.Math.Max(1,System.Math.Abs(go[i,j])));
  if(ho==null || hc==null)throw new System.Exception("DE69 equivalence invalid GN start");
  for(int i=0;i<ho.GetLength(0);i++)for(int j=0;j<ho.GetLength(1);j++)report.hessianError=System.Math.Max(report.hessianError,System.Math.Abs(ho[i,j]-hc[i,j])/System.Math.Max(1,System.Math.Abs(ho[i,j])));
  report.passed=report.stateError<1e-10&&report.jacobianError<1e-10&&report.inputJacobianError<1e-10&&report.costError<1e-10&&report.gradientError<1e-9&&report.hessianError<1e-9;
  System.IO.File.WriteAllText(System.IO.Path.Combine(artifactDirectory,"compact_equivalence.json"),UnityEngine.JsonUtility.ToJson(report,true));
  if(!report.passed)throw new System.Exception("DE69 equivalence failed");
  UnityEngine.Debug.Log("FINSSIM_DE69_COMPACT_EQUIVALENT "+report.stateError+" "+report.gradientError+" "+report.hessianError);
 }
}

public sealed partial class FinsRovHybridBasisSmoke {
 void De81EncodingTiming(){
  var values=new double[101];for(int i=0;i<20;i++)hybrid.LiftWorld(stateP,stateV,historyWrench);
  for(int i=0;i<values.Length;i++){var sw=System.Diagnostics.Stopwatch.StartNew();for(int j=0;j<100;j++)hybrid.LiftWorld(stateP,stateV,historyWrench);sw.Stop();values[i]=sw.Elapsed.TotalMilliseconds/100;}
  System.Array.Sort(values);
  System.IO.File.WriteAllText(System.IO.Path.Combine(artifactDirectory,"encoder_timing.json"),"{\"full_lift_ms_median\":"+values[50].ToString("R",System.Globalization.CultureInfo.InvariantCulture)+",\"full_lift_ms_p95\":"+values[95].ToString("R",System.Globalization.CultureInfo.InvariantCulture)+",\"includes_physical_and_encoding\":true}");
 }
}

public sealed partial class FinsRovHybridBasisSmoke {
 double[] de83Previous;float de83PreviousTime;
 readonly System.Collections.Generic.List<double[]> de83VErrors=new System.Collections.Generic.List<double[]>();
 readonly System.Collections.Generic.List<double[]> de83PErrors=new System.Collections.Generic.List<double[]>();
 string De83Online=>System.Environment.GetEnvironmentVariable("FINSSIM_DE83_ONLINE")??"none";
 [System.Serializable] public class De83Record {
  public float t,previousTime;public string mode;public int samples;public bool acceptedV,acceptedP;
  public double milliseconds,fitVBefore,fitVAfter,fitPBefore,fitPAfter;
  public double[] previous,input,observed,baseNext,errorV,errorP,velocityBias,poseBias;
 }
 void De83Reset(){
  De92Reset();de85Variance=null;de83Previous=null;de83VErrors.Clear();de83PErrors.Clear();de82Truth=null;
  if(GetComponent<De82PhysicsAudit>()==null)gameObject.AddComponent<De82PhysicsAudit>();
  if(hybrid!=null){System.Array.Clear(hybrid.De83VB,0,6);System.Array.Clear(hybrid.De83PB,0,6);}
 }
 static double De83Norm(double[] a){double s=0;foreach(double x in a)s+=x*x;return System.Math.Sqrt(s);}
 static double De83Fit(System.Collections.Generic.List<double[]> errors,double[] b){double value=0;for(int j=0;j<errors.Count;j++){double w=System.Math.Pow(.98,errors.Count-1-j);for(int i=0;i<6;i++){double e=errors[j][i]-b[i];value+=w*e*e;}}return value;}
 static bool De83Update(System.Collections.Generic.List<double[]> errors,double[] bias,double gain,double bound,double step,out double before,out double after){
  before=De83Fit(errors,bias);after=before;if(errors.Count<20)return false;
  double sum=0;var target=new double[6];for(int j=0;j<errors.Count;j++){double w=System.Math.Pow(.98,errors.Count-1-j);sum+=w;for(int i=0;i<6;i++)target[i]+=w*errors[j][i];}
  var candidate=new double[6];for(int i=0;i<6;i++){target[i]=(target[i]+.5*bias[i])/(sum+1.5);candidate[i]=bias[i]+gain*(target[i]-bias[i]);}
  double norm=De83Norm(candidate);if(norm>bound)for(int i=0;i<6;i++)candidate[i]*=bound/norm;
  var diff=new double[6];for(int i=0;i<6;i++)diff[i]=candidate[i]-bias[i];norm=De83Norm(diff);if(norm>step)for(int i=0;i<6;i++)candidate[i]=bias[i]+diff[i]*step/norm;
  double fit=De83Fit(errors,candidate);if(fit>before+1e-14)return false;System.Array.Copy(candidate,bias,6);after=fit;return true;
 }
 void De83Observe(float t,bool learned){
  if(!learned)return;var watch=System.Diagnostics.Stopwatch.StartNew();var current=FinsRovHybridOperator.De69Pack(LiftState(stateP,stateV,historyWrench));
  if(de83Previous==null){de83Previous=current;de83PreviousTime=t;return;}
  if(System.Math.Abs(t-de83PreviousTime-.1)>.002)throw new System.Exception("DE83 causal interval");
  var u=new double[6];for(int i=0;i<6;i++)u[i]=historyWrench[i]/WrenchLimit[i];
  var oldV=hybrid.De83VB;var oldP=hybrid.De83PB;hybrid.De83VB=new double[6];hybrid.De83PB=new double[6];double[] baseline;
  try{baseline=hybrid.De69Step(de83Previous,u);}finally{hybrid.De83VB=oldV;hybrid.De83PB=oldP;}
  var ev=new double[6];var ep=new double[6];var bp=FinsRovBodyFrame.Pose(baseline);var cp=FinsRovBodyFrame.Pose(current);var angle=FinsRovHybridOperator.De83RotationError(bp,cp);
  for(int i=0;i<6;i++){ev[i]=current[7+i]-baseline[7+i];ep[i]=(i<3?cp[i]-bp[i]:angle[i-3])/hybrid.Data.de83PoseBiasScale[i];}
  de83VErrors.Add(ev);de83PErrors.Add(ep);if(de83VErrors.Count>40){de83VErrors.RemoveAt(0);de83PErrors.RemoveAt(0);}
  double vb=De83Fit(de83VErrors,oldV),va=vb,pb=De83Fit(de83PErrors,oldP),pa=pb;bool av=false,ap=false;
  if(De83Online=="velocity" || De83Online=="both")av=De83Update(de83VErrors,hybrid.De83VB,.2,.25,.01,out vb,out va);
  if(De83Online=="estimator"){av=De85Update(ev,hybrid.De83VB);va=De83Fit(de83VErrors,hybrid.De83VB);}
  if(De83Online=="de92")av=De92Update(t,ev,de83Previous,out vb,out va);
  if(De83Online=="both")ap=De83Update(de83PErrors,hybrid.De83PB,.05,1d,.05,out pb,out pa);
  watch.Stop();var rec=new De83Record{t=t,previousTime=de83PreviousTime,mode=De83Online,samples=de83VErrors.Count,acceptedV=av,acceptedP=ap,milliseconds=watch.Elapsed.TotalMilliseconds,fitVBefore=vb,fitVAfter=va,fitPBefore=pb,fitPAfter=pa,
   previous=de83Previous,input=u,observed=current,baseNext=baseline,errorV=ev,errorP=ep,velocityBias=hybrid.De83VB,poseBias=hybrid.De83PB};
  System.IO.File.AppendAllText(System.IO.Path.Combine(artifactDirectory,"online_adaptation.jsonl"),UnityEngine.JsonUtility.ToJson(rec)+"\n");de83Previous=current;de83PreviousTime=t;
 }
}

public sealed partial class FinsRovHybridBasisSmoke {
 float de82PhysicsTime;
 System.IO.StreamWriter de82PhysicsCsv;
 string de82Truth;

 string de90FlowPath="";double[,] de90Flow;
 UnityEngine.Vector3 De82FlowAt(double t,int seed){
  string path=System.Environment.GetEnvironmentVariable("FINSSIM_DE90_FLOW_FILE");
  if(path!=de90FlowPath){var lines=System.IO.File.ReadAllLines(path);if(lines.Length!=4000)throw new System.Exception("DE90 flow length");de90Flow=new double[4000,3];for(int i=0;i<4000;i++){var cells=lines[i].Split(',');for(int j=0;j<3;j++)de90Flow[i,j]=double.Parse(cells[j],System.Globalization.CultureInfo.InvariantCulture);}de90FlowPath=path;}
  int k=System.Math.Max(0,System.Math.Min(3999,(int)System.Math.Round(t/.02)));
  return new UnityEngine.Vector3((float)de90Flow[k,0],(float)de90Flow[k,1],(float)de90Flow[k,2]);
 }
 public void De82AuditPhysics(){
  if(phase!=Phase.KoopmanMpc)return;
  var p=hydro.RuntimeProfile;string truth=UnityEngine.JsonUtility.ToJson(p);
  if(de82Truth==null){de82Truth=truth;System.IO.File.WriteAllText(System.IO.Path.Combine(artifactDirectory,"physical_profile_fixed.json"),truth);}
  if(truth!=de82Truth)throw new System.Exception("DE82 true hydrodynamic parameters changed");
  FinsSim.Hydrodynamics.Fossen6DofDiagnostics d;
  if(!hydro.TryGetFossenDiagnostics(out d))throw new System.Exception("DE82 expected Fossen backend");
  if(de82PhysicsCsv==null){
   de82PhysicsCsv=new System.IO.StreamWriter(System.IO.Path.Combine(artifactDirectory,"current_physics.csv"),false);
   var header=new System.Collections.Generic.List<string>{"t_s","flow_x","flow_y","flow_z"};
   foreach(string name in new[]{"ld","qd","fd","v","damping","added_acc","coriolis"})for(int i=0;i<6;i++)header.Add(name+"_"+i);
   de82PhysicsCsv.WriteLine(string.Join(",",header));
  }
  var flow=hydro.LastWaterSample.FlowVelocity;
  var values=new System.Collections.Generic.List<string>{de82PhysicsTime.ToString("R",System.Globalization.CultureInfo.InvariantCulture),flow.x.ToString("R",System.Globalization.CultureInfo.InvariantCulture),flow.y.ToString("R",System.Globalization.CultureInfo.InvariantCulture),flow.z.ToString("R",System.Globalization.CultureInfo.InvariantCulture)};
  foreach(var v in new[]{p.linearDamping,p.quadraticDamping,p.forwardSpeedDamping,d.RelativeVelocity,d.DampingBodyWrench,d.AddedMassAccelerationBodyWrench,d.AddedMassCoriolisBodyWrench})for(int i=0;i<6;i++)values.Add(v[i].ToString("R",System.Globalization.CultureInfo.InvariantCulture));
  de82PhysicsCsv.WriteLine(string.Join(",",values));if(rateCycleActive)de82PhysicsCsv.Flush();
 }
}
[UnityEngine.DefaultExecutionOrder(30)]
public sealed class De82PhysicsAudit:UnityEngine.MonoBehaviour {
 void FixedUpdate(){var host=GetComponent<FinsRovHybridBasisSmoke>();if(host!=null)host.De82AuditPhysics();}
}

// DE84 causal timestamp alignment. Append to the isolated DE83 smoke source.
// Hooks: De84ResetObservation() at episode reset; De84AlignObservation(t)
// after InitializeObservedControl/UpdateModelHistory, before De83Observe/MPC.
public sealed partial class FinsRovHybridBasisSmoke {
 sealed class De84SensorPoint {
  public double time;public double[] attitude,angularWorld,input;public int ticks;
 }
 [System.Serializable] public class De84PropagationStep {
  public double start,end;public int availableTicks;public double[] input,state;
 }
 [System.Serializable] public class De84AlignmentRecord {
  public int index;public string mode,status;public bool newMeasurement,applied;
  public double t,now,sourceTime,arrivalTime,sourceAge,sourceImuTime,milliseconds;
  public double[] beforePose,beforeWorldVelocity,measuredPosition,measuredWorldVelocity;
  public double[] currentInput,sourcePose,sourceWorldVelocity,initial,predicted,afterPose,afterWorldVelocity;
  public int currentInputTicks;public De84PropagationStep[] steps;
 }
 readonly System.Collections.Generic.List<De84SensorPoint> de84SensorHistory=new System.Collections.Generic.List<De84SensorPoint>();
 double de84LastSource=double.NaN,de84ArrivalTime;int de84ObservationIndex;
 string De84ObservationMode=>System.Environment.GetEnvironmentVariable("FINSSIM_DE84_OBSERVATION")??"legacy";
 void De84ResetObservation(){de84SensorHistory.Clear();de84LastSource=double.NaN;de84ArrivalTime=0;de84ObservationIndex=0;}
 static double[] De84Doubles(float[] a)=>System.Array.ConvertAll(a,x=>(double)x);
 static double[] De84Three(float[] a){return new[]{(double)a[0],(double)a[1],(double)a[2]};}
 De84SensorPoint De84At(double time){
  foreach(var q in de84SensorHistory)if(System.Math.Abs(q.time-time)<2e-5)return q;
  return null;
 }
 void De84AlignObservation(float t){
  var watch=System.Diagnostics.Stopwatch.StartNew();string mode=De84ObservationMode;
  if(mode!="legacy" && mode!="aligned")throw new System.Exception("DE84 observation mode");
  if(observationMode!="multi100_cv" || heldPosition==null || heldVelocity==null || !observationReady)
   throw new System.Exception("DE84 requires arrived multi100_cv measurement");
  if(hybrid.Data.de83Mode!="A" || hybrid.Data.latent!=10 || hybrid.Data.encoderHistoryInput)
   throw new System.Exception("DE84 requires frozen G1 A/no-history model");
  double now=SimTime;var input=new double[6];for(int i=0;i<6;i++)input[i]=historyWrench[i]/WrenchLimit[i];
  var point=new De84SensorPoint{time=now,attitude=new[]{(double)stateP[3],stateP[4],stateP[5]},
   angularWorld=new[]{(double)stateV[3],stateV[4],stateV[5]},input=input,ticks=rateInputCount};
  if(de84SensorHistory.Count>0 && System.Math.Abs(now-de84SensorHistory[de84SensorHistory.Count-1].time-.1)>2e-5)
   throw new System.Exception("DE84 observation control interval");
  de84SensorHistory.Add(point);
  bool fresh=double.IsNaN(de84LastSource)||System.Math.Abs(heldSourceTime-de84LastSource)>2e-5;
  de85Fresh=fresh;
  if(fresh){de84LastSource=heldSourceTime;de84ArrivalTime=now;}
  var rec=new De84AlignmentRecord{index=de84ObservationIndex++,mode=mode,status="legacy",newMeasurement=fresh,applied=false,
   t=t,now=now,sourceTime=heldSourceTime,arrivalTime=de84ArrivalTime,sourceAge=now-heldSourceTime,sourceImuTime=-1,
   beforePose=De84Doubles(stateP),beforeWorldVelocity=De84Doubles(stateV),
   measuredPosition=De84Three(heldPosition),measuredWorldVelocity=De84Three(heldVelocity),
   currentInput=(double[])input.Clone(),currentInputTicks=rateInputCount,
   initial=new double[0],predicted=new double[0],sourcePose=new double[0],sourceWorldVelocity=new double[0],steps=new De84PropagationStep[0]};
  if(rec.sourceAge< -2e-5 || rec.sourceAge>.201)throw new System.Exception("DE84 unexpected measurement age");
  var source=De84At(heldSourceTime);int count=(int)System.Math.Round(rec.sourceAge/.1);
  bool ready=source!=null && System.Math.Abs(rec.sourceAge-count*.1)<2e-5;
  var spans=new System.Collections.Generic.List<De84SensorPoint>();
  if(ready)for(int j=1;j<=count;j++){
   var endpoint=De84At(heldSourceTime+j*.1);if(endpoint==null || endpoint.ticks!=5){ready=false;break;}spans.Add(endpoint);
  }
  if(!ready){
   rec.status="warmup_missing_history";
   // Never fall back to delayed.p's noiseless angles or a truth velocity.
   if(t>=.3f)throw new System.Exception("DE84 causal source/input history unavailable after warmup");
  } else {
   rec.sourceImuTime=source.time;var p=new float[6];var v=new float[6];
   for(int i=0;i<3;i++){p[i]=heldPosition[i];v[i]=heldVelocity[i];p[3+i]=(float)source.attitude[i];v[3+i]=(float)source.angularWorld[i];}
   rec.sourcePose=De84Doubles(p);rec.sourceWorldVelocity=De84Doubles(v);
   if(mode=="aligned"){
    var oldV=hybrid.De83VB;var oldP=hybrid.De83PB;
    hybrid.De83VB=new double[6];hybrid.De83PB=new double[6];
    try{
     var z=FinsRovHybridOperator.De69Pack(hybrid.LiftWorld(p,v,new float[6]));rec.initial=(double[])z.Clone();
     var steps=new System.Collections.Generic.List<De84PropagationStep>();double start=heldSourceTime;
     foreach(var endpoint in spans){
      z=hybrid.De69Step(z,endpoint.input);
      steps.Add(new De84PropagationStep{start=start,end=endpoint.time,availableTicks=endpoint.ticks,input=(double[])endpoint.input.Clone(),state=(double[])z.Clone()});start=endpoint.time;
     }
     rec.steps=steps.ToArray();rec.predicted=(double[])z.Clone();
     var pp=FinsRovBodyFrame.Pose(z);var rp=FinsRovBodyFrame.Rotation(pp);var ri=FinsRovBodyFrame.Rotation(rec.beforePose);
     // Both velocities are at COM. Keep predicted COM position while replacing
     // predicted orientation with the current arrived IMU orientation.
     for(int i=0;i<3;i++){
      double position=pp[i],velocity=0;
      for(int j=0;j<3;j++){position+=(rp[i,j]-ri[i,j])*hybrid.Data.de83Center[j];velocity+=rp[i,j]*z[7+j]*FinsRovLiftedOperator.VelocityScale[j];}
      stateP[i]=(float)position;stateV[i]=(float)velocity;
     }
     rec.applied=true;rec.status="aligned";
    }finally{hybrid.De83VB=oldV;hybrid.De83PB=oldP;}
   }
  }
  rec.afterPose=De84Doubles(stateP);rec.afterWorldVelocity=De84Doubles(stateV);
  // Orientation/rate and observationRotation are deliberately unchanged.
  watch.Stop();rec.milliseconds=watch.Elapsed.TotalMilliseconds;
  System.IO.File.AppendAllText(System.IO.Path.Combine(artifactDirectory,"observation_alignment.jsonl"),UnityEngine.JsonUtility.ToJson(rec)+"\n");
  if(de84SensorHistory.Count>16)de84SensorHistory.RemoveAt(0);
 }
}

public sealed partial class FinsRovHybridBasisSmoke {
 bool de85Fresh; double[] de85Variance;
 readonly double[] de85R=new double[]{0.031689448958163745,0.090208639863389437,0.034074704719025843,0.033199089528250729,0.020321989586419197,0.01838267299769571};
 bool De85Update(double[] y,double[] mean){
  if(de85Variance==null)de85Variance=(double[])de85R.Clone();
  var prior=new double[6];var gain=new double[6];var delta=new double[6];var candidate=new double[6];
  for(int i=0;i<6;i++){prior[i]=de85Variance[i]+.01*de85R[i];gain[i]=(i>=3||de85Fresh)?prior[i]/(prior[i]+de85R[i]):0;delta[i]=gain[i]*(y[i]-mean[i]);}
  double factor=System.Math.Min(1d,.01/System.Math.Max(De83Norm(delta),1e-300));
  for(int i=0;i<6;i++)candidate[i]=mean[i]+factor*delta[i];
  double scale=System.Math.Min(1d,.25/System.Math.Max(De83Norm(candidate),1e-300));bool clipped=false;
  for(int i=0;i<6;i++){candidate[i]*=scale;if(System.Math.Abs(candidate[i]-mean[i]-delta[i])>1e-14)clipped=true;}
  for(int i=0;i<6;i++){de85Variance[i]=(clipped && System.Environment.GetEnvironmentVariable("FINSSIM_DE86_COVARIANCE")!="1")?prior[i]:(1-gain[i])*prior[i];mean[i]=candidate[i];}return true;
 }
}

public sealed partial class FinsRovHybridBasisSmoke {
 double[] de92C=new double[60],de92Weights;int de92Tick;
 readonly System.Collections.Generic.List<double[]> de92Z=new System.Collections.Generic.List<double[]>(),de92Y=new System.Collections.Generic.List<double[]>();
 string De92Mode=>System.Environment.GetEnvironmentVariable("FINSSIM_DE92_MODE")??"fixed";
 void De92Reset(){System.Array.Clear(de92C,0,60);de92Weights=null;de92Tick=0;de92Z.Clear();de92Y.Clear();}
 double De92Weight(int index){
  if(de92Weights==null){de92Weights=new double[8];string mode=System.Environment.GetEnvironmentVariable("FINSSIM_DE92_COST")??"legacy";double sum=0;
   for(int j=0;j<8;j++){double x=mode=="legacy"?(j==7?3d:1d):mode=="flat"?1d:mode=="exp"?System.Math.Pow(.25,j/7d):mode=="linear"?1d-.75*j/7d:double.NaN;de92Weights[j]=x;sum+=x;}
   if(double.IsNaN(sum))throw new System.Exception("DE92 cost mode");if(mode!="legacy")for(int j=0;j<8;j++)de92Weights[j]*=10d/sum;
  }return de92Weights[index];
 }
 static double[,] De92Solve(double[,] a,double[,] b){
  int d=a.GetLength(0);var m=(double[,])a.Clone();var x=(double[,])b.Clone();
  for(int k=0;k<d;k++){int p=k;for(int i=k+1;i<d;i++)if(System.Math.Abs(m[i,k])>System.Math.Abs(m[p,k]))p=i;
   if(System.Math.Abs(m[p,k])<1e-12)throw new System.Exception("DE92 singular ridge");
   for(int j=0;j<d;j++){double v=m[k,j];m[k,j]=m[p,j];m[p,j]=v;}for(int j=0;j<6;j++){double v=x[k,j];x[k,j]=x[p,j];x[p,j]=v;}
   double divisor=m[k,k];for(int j=k;j<d;j++)m[k,j]/=divisor;for(int j=0;j<6;j++)x[k,j]/=divisor;
   for(int i=0;i<d;i++)if(i!=k){double f=m[i,k];for(int j=k;j<d;j++)m[i,j]-=f*m[k,j];for(int j=0;j<6;j++)x[i,j]-=f*x[k,j];}
  }return x;
 }
 double De92Fit(double[,] parameters,double[] weights,bool joint){
  double value=0;for(int n=0;n<de92Y.Count;n++)for(int i=0;i<6;i++){double e=de92Y[n][i]-parameters[i,0];if(joint)for(int j=0;j<10;j++)e-=parameters[i,1+j]*de92Z[n][j];value+=weights[n]*e*e;}return value;
 }
 [System.Serializable] public class De92Record {public double t,before,after,weightSum;public int tick,samples,window;public double forget;public bool accepted;public double[] matrix,bias,center,features,innovation;}
 bool De92Update(float t,double[] y,double[] previous,out double before,out double after){
  bool joint=De92Mode=="joint";int d=joint?11:1;
  int window=int.Parse(System.Environment.GetEnvironmentVariable("FINSSIM_DE92_WINDOW"));double forget=double.Parse(System.Environment.GetEnvironmentVariable("FINSSIM_DE92_FORGET"),System.Globalization.CultureInfo.InvariantCulture);
  var z=new double[10];System.Array.Copy(previous,13,z,0,10);de92Z.Add(z);de92Y.Add((double[])y.Clone());de92Tick++;
  if(de92Z.Count>window){de92Z.RemoveAt(0);de92Y.RemoveAt(0);}int count=de92Y.Count;
  double reference=0,sum=0;for(int j=0;j<System.Math.Min(de92Tick,40);j++)reference+=System.Math.Pow(.98,j);
  var weights=new double[count];for(int n=0;n<count;n++){weights[n]=System.Math.Pow(forget,count-1-n);sum+=weights[n];}
  for(int n=0;n<count;n++)weights[n]*=reference/sum;
  var mu=new double[10];if(joint)for(int n=0;n<count;n++)for(int j=0;j<10;j++)mu[j]+=weights[n]*de92Z[n][j]/reference;
  var old=new double[6,d];for(int i=0;i<6;i++){old[i,0]=hybrid.De83VB[i];if(joint)for(int j=0;j<10;j++)old[i,1+j]=de92C[i*10+j];}
  before=De92Fit(old,weights,joint);after=before;bool accepted=false;
  if(de92Tick>=20){
   var gram=new double[d,d];var rhs=new double[d,6];for(int j=0;j<d;j++){gram[j,j]=j==0?1.5:4.5;for(int i=0;i<6;i++){rhs[j,i]=.5*old[i,j];if(joint && j==0)for(int l=0;l<10;l++)rhs[j,i]+=.5*de92C[i*10+l]*mu[l];}}
   for(int n=0;n<count;n++){var x=new double[d];x[0]=1;if(joint)for(int j=0;j<10;j++)x[j+1]=de92Z[n][j]-mu[j];
    for(int j=0;j<d;j++){for(int l=0;l<d;l++)gram[j,l]+=weights[n]*x[j]*x[l];for(int i=0;i<6;i++)rhs[j,i]+=weights[n]*x[j]*de92Y[n][i];}}
   var sol=De92Solve(gram,rhs);var target=new double[6,d];double norm=0;
   for(int i=0;i<6;i++){for(int j=0;j<d;j++)target[i,j]=sol[j,i];if(joint)for(int j=0;j<10;j++)target[i,0]-=target[i,j+1]*mu[j];}
   if(!joint)for(int i=0;i<6;i++)target[i,0]=old[i,0]+.2*(target[i,0]-old[i,0]);
   for(int i=0;i<6;i++)for(int j=0;j<d;j++)norm+=target[i,j]*target[i,j];double f=System.Math.Min(1,.25/System.Math.Max(System.Math.Sqrt(norm),1e-30));
   var delta=new double[6,d];norm=0;for(int i=0;i<6;i++)for(int j=0;j<d;j++){delta[i,j]=(joint?.2:1d)*(f*target[i,j]-old[i,j]);norm+=delta[i,j]*delta[i,j];}
   f=System.Math.Min(1,.01/System.Math.Max(System.Math.Sqrt(norm),1e-30));var proposal=new double[6,d];for(int i=0;i<6;i++)for(int j=0;j<d;j++)proposal[i,j]=old[i,j]+f*delta[i,j];
   double score=De92Fit(proposal,weights,joint);if(score<=before+1e-14){accepted=true;after=score;for(int i=0;i<6;i++){hybrid.De83VB[i]=proposal[i,0];if(joint)for(int j=0;j<10;j++)de92C[i*10+j]=proposal[i,j+1];}}
  }
  var record=new De92Record{t=t,tick=de92Tick,samples=count,window=window,forget=forget,before=before,after=after,weightSum=reference,accepted=accepted,matrix=de92C,bias=hybrid.De83VB,center=mu,features=z,innovation=y};
  System.IO.File.AppendAllText(System.IO.Path.Combine(artifactDirectory,"window_update.jsonl"),UnityEngine.JsonUtility.ToJson(record)+"\n");return accepted;
 }
 double[] De92Prediction(double[] z,double[] u,double[,] ja,double[,] jb){
  if(De92Mode!="joint")return hybrid.De69Step(z,u,ja,jb);
  var before=new double[60];for(int i=0;i<6;i++)for(int j=0;j<10;j++){int k=i*10+j;before[k]=hybrid.A[7+i,36+j];hybrid.A[7+i,36+j]+=de92C[k];}
  try{return hybrid.De69Step(z,u,ja,jb);}finally{for(int i=0;i<6;i++)for(int j=0;j<10;j++)hybrid.A[7+i,36+j]=before[i*10+j];}
 }
}

public sealed partial class FinsRovHybridBasisSmoke { [System.Serializable] sealed class De100Failure { public float t;public string reason,site;public float[] pose,velocity; } }
