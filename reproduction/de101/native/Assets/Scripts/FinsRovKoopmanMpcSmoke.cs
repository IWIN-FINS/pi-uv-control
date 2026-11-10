using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using FinsSim.Actuators;
using FinsSim.Hydrodynamics;
using UnityEngine;

// Paired six-axis receding-horizon experiment. Both arms use the exact same
// finite-horizon optimizer, costs, wrench bounds and eight-thruster allocator.
// The Koopman arm alone adds a frozen, identified acceleration residual to
// the model rolled out inside MPC; no extra force is applied outside MPC.
[DefaultExecutionOrder(-50)]
public sealed class FinsRovKoopmanMpcSmoke : MonoBehaviour
{
    enum Phase { Identify, SettleBaseline, BaselineMpc, SettleKoopman, KoopmanMpc, Done }
    const int AxisCount = 6;
    const int LiftCount = 4;
    const int Horizon = 12;
    const int GradientIterations = 28;
    const float ControlDt = 0.1f;
    const float IdentifySeconds = 24f;
    const float EvaluationSeconds = 32f;
    const float SettleSeconds = 2f;
    const float NoiseStdN = 0.20f;
    const float ThrustScale = 0.78f;
    const float Period = 32f;
    const float ValidationRatioLimit = 0.20f;
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
    readonly float[,] warm = new float[AxisCount, Horizon];
    readonly float[,] fitted = new float[AxisCount, LiftCount];
    readonly bool[] residualAllowed = new bool[AxisCount];
    readonly List<float[]> identifyFeature = new List<float[]>();
    readonly List<float[]> identifyTarget = new List<float[]>();
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
    float phaseStart;
    float nextControl;
    float nextSample;
    float previousSampleTime = -1f;
    float lastSolveMs;
    float lastAllocationResidual;
    float lastMotorEffort;
    int lastSaturatedMotors;
    int baselineSamples;
    int koopmanSamples;
    int evaluationSeed;
    double baselineHorizontalSquared;
    double koopmanHorizontalSquared;

    public string ViewerPhase => phase.ToString();
    public float ViewerPhaseTime => phase == Phase.Done ? EvaluationSeconds :
        Mathf.Max(0f, Time.fixedTime - phaseStart);
    public float ViewerPhaseDuration => phase == Phase.Identify ? IdentifySeconds :
        phase == Phase.BaselineMpc || phase == Phase.KoopmanMpc || phase == Phase.Done ?
        EvaluationSeconds : SettleSeconds;
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
        truth.addedMassDiagonal.u *= 1.35f;
        truth.addedMassDiagonal.v *= 1.25f;
        truth.addedMassDiagonal.w *= 1.20f;
        truth.addedMassDiagonal.r *= 1.30f;
        truth.linearDamping.u *= 1.55f;
        truth.linearDamping.v *= 1.45f;
        truth.linearDamping.w *= 1.30f;
        truth.linearDamping.r *= 1.35f;
        truth.linearDamping.p += 0.18f;
        truth.linearDamping.q += 0.18f;
        truth.quadraticDamping.u += 12f;
        truth.quadraticDamping.v += 16f;
        truth.quadraticDamping.w += 20f;
        hydro.ResetBackendState();

        artifactDirectory = Environment.GetEnvironmentVariable("FINSSIM_KOOPMAN_MPC_DIR");
        if (string.IsNullOrEmpty(artifactDirectory))
            artifactDirectory = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/koopman-mpc-smoke"));
        Directory.CreateDirectory(artifactDirectory);
        trialCsv = new StreamWriter(Path.Combine(artifactDirectory, "mpc_trials.csv"), false);
        trialCsv.WriteLine("phase,t_s,x_m,y_m,z_m,ref_x_m,ref_y_m,ref_z_m,horizontal_error_m,depth_error_m,roll_deg,pitch_deg,yaw_deg,ref_yaw_deg,tilt_deg,attitude_error_deg,fx_n,fy_n,fz_n,tx_nm,ty_nm,tz_nm,motor_effort_n,allocation_residual,solver_ms,predicted_initial_cost,predicted_final_cost,saturated_motors");
        identifyCsv = new StreamWriter(Path.Combine(artifactDirectory, "mpc_identification.csv"), false);
        identifyCsv.WriteLine("axis,t_s,position,velocity,requested_wrench,residual_acceleration");
        phase = Phase.Identify;
        phaseStart = Time.fixedTime;
        nextControl = Time.fixedTime;
        nextSample = Time.fixedTime;
        noise = new System.Random(24001);
        if (!int.TryParse(Environment.GetEnvironmentVariable("FINSSIM_KOOPMAN_MPC_EVAL_SEED"),
            NumberStyles.Integer, CultureInfo.InvariantCulture, out evaluationSeed))
            evaluationSeed = 9091;
        Debug.Log("FINSSIM_KOOPMAN_MPC_START " + artifactDirectory);
        Debug.Log("FINSSIM_KOOPMAN_MPC_EVAL_SEED " + evaluationSeed);
        Debug.Log($"FINSSIM_KOOPMAN_MPC_MODEL mass={string.Join(",", mass)} spring={spring[3]:F3} buoyancyBias={bias[1]:F3}");
    }

    void FixedUpdate()
    {
        if (phase == Phase.Done) return;
        float t = Time.fixedTime - phaseStart;
        float duration = phase == Phase.Identify ? IdentifySeconds :
            phase == Phase.BaselineMpc || phase == Phase.KoopmanMpc ? EvaluationSeconds : SettleSeconds;
        if (t >= duration)
        {
            AdvancePhase();
            if (phase == Phase.Done) return;
            t = 0f;
        }
        ReadState(stateP, stateV);
        if (Time.fixedTime + 0.0001f >= nextControl)
        {
            if (phase == Phase.Identify)
            {
                RecordIdentification(t);
                Excitation(t);
                Array.Copy(stateP, previousP, AxisCount);
                Array.Copy(stateV, previousV, AxisCount);
                Array.Copy(wrench, previousWrench, AxisCount);
                previousSampleTime = Time.fixedTime;
            }
            else if (phase == Phase.BaselineMpc || phase == Phase.KoopmanMpc)
                SolveMpc(t, phase == Phase.KoopmanMpc);
            else Array.Clear(wrench, 0, AxisCount);
            nextControl = Time.fixedTime + ControlDt;
        }
        if (phase == Phase.SettleBaseline || phase == Phase.SettleKoopman)
        {
            foreach (var motor in thrusters) motor.ApplyForceRequest(0f);
            return;
        }
        Allocate(wrench, motorRequest);
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
            Time.fixedTime + 0.0001f >= nextSample)
        {
            RecordTrial(t);
            nextSample = Time.fixedTime + ControlDt;
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

    static float Reference(int axis, float t)
    {
        float omega = 2f * Mathf.PI / Period;
        if (axis == 0) return 0.60f * Mathf.Sin(omega * t);
        if (axis == 1) return -1.20f + 0.08f * Mathf.Sin(omega * t);
        if (axis == 2) return 0.35f * Mathf.Sin(2f * omega * t);
        if (axis == 4) return 0.10f * Mathf.Sin(omega * t);
        return 0f;
    }

    static float ReferenceVelocity(int axis, float t)
    {
        float omega = 2f * Mathf.PI / Period;
        if (axis == 0) return 0.60f * omega * Mathf.Cos(omega * t);
        if (axis == 1) return 0.08f * omega * Mathf.Cos(omega * t);
        if (axis == 2) return 0.70f * omega * Mathf.Cos(2f * omega * t);
        if (axis == 4) return 0.10f * omega * Mathf.Cos(omega * t);
        return 0f;
    }

    void Excitation(float t)
    {
        wrench[0] = 4f * Mathf.Sin(0.77f * t) + 2f * Mathf.Sin(1.61f * t);
        wrench[1] = 2.5f * Mathf.Sin(0.83f * t) + 1.0f * Mathf.Sin(1.37f * t);
        wrench[2] = 4f * Mathf.Cos(0.57f * t) + 2f * Mathf.Sin(1.29f * t);
        wrench[3] = 0.45f * Mathf.Sin(0.91f * t);
        wrench[4] = 0.45f * Mathf.Cos(0.73f * t);
        wrench[5] = 0.45f * Mathf.Sin(1.09f * t);
    }

    float NominalAcceleration(int axis, float p, float v, float u)
    {
        return (u - linearDrag[axis] * v - quadraticDrag[axis] * v * Mathf.Abs(v)
            - spring[axis] * p + bias[axis]) / mass[axis];
    }

    static float[] Lift(float v, float u)
    {
        return new[] { 1f, 5f * v, 0.2f * u, 25f * v * Mathf.Abs(v) };
    }

    void RecordIdentification(float t)
    {
        if (previousSampleTime < 0f) return;
        float dt = Time.fixedTime - previousSampleTime;
        if (dt <= 0.001f) return;
        var phi = new float[AxisCount * LiftCount];
        var residual = new float[AxisCount];
        for (int axis = 0; axis < AxisCount; axis++)
        {
            float observed = (stateV[axis] - previousV[axis]) / dt;
            residual[axis] = observed - NominalAcceleration(axis,
                previousP[axis], previousV[axis], previousWrench[axis]);
            float[] lifted = Lift(previousV[axis], previousWrench[axis]);
            Array.Copy(lifted, 0, phi, axis * LiftCount, LiftCount);
            identifyCsv.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "{0},{1:F3},{2:F6},{3:F6},{4:F6},{5:F6}", axis, t,
                previousP[axis], previousV[axis], previousWrench[axis], residual[axis]));
        }
        identifyFeature.Add(phi);
        identifyTarget.Add(residual);
    }

    void FitResidual()
    {
        if (identifyFeature.Count < 160)
            throw new InvalidOperationException("Insufficient six-axis identification samples");
        int trainCount = identifyFeature.Count * 4 / 5;
        for (int axis = 0; axis < AxisCount; axis++)
        {
            var gram = new double[LiftCount, LiftCount];
            var rhs = new double[LiftCount];
            int offset = axis * LiftCount;
            for (int i = 0; i < trainCount; i++)
                for (int a = 0; a < LiftCount; a++)
                {
                    double value = identifyFeature[i][offset + a];
                    rhs[a] += value * identifyTarget[i][axis];
                    for (int b = 0; b < LiftCount; b++)
                        gram[a, b] += value * identifyFeature[i][offset + b];
                }
            for (int a = 0; a < LiftCount; a++) gram[a, a] += a == 0 ? 0.001 : 0.03;
            double[] weights = Solve(gram, rhs);
            for (int a = 0; a < LiftCount; a++) fitted[axis, a] = (float)weights[a];
            double zeroError = 0d, fittedError = 0d;
            for (int i = trainCount; i < identifyFeature.Count; i++)
            {
                double predicted = 0d;
                for (int a = 0; a < LiftCount; a++)
                    predicted += weights[a] * identifyFeature[i][offset + a];
                double target = identifyTarget[i][axis];
                zeroError += target * target;
                fittedError += (target - predicted) * (target - predicted);
            }
            double ratio = fittedError / Math.Max(1e-12, zeroError);
            // A residual that fails its disjoint identification holdout is not
            // admitted into MPC. This gate is fixed before either test arm.
            residualAllowed[axis] = ratio <= ValidationRatioLimit;
            Debug.Log($"FINSSIM_KOOPMAN_MPC_FIT axis={axis} train={trainCount} " +
                $"holdout={identifyFeature.Count - trainCount} mseRatio={ratio:F4} " +
                $"enabled={residualAllowed[axis]}");
        }
        identifyCsv.Flush();
    }

    float Residual(int axis, float v, float u, out float derivativeV, out float derivativeU)
    {
        float raw = fitted[axis, 0] + 5f * fitted[axis, 1] * v +
            0.2f * fitted[axis, 2] * u + 25f * fitted[axis, 3] * v * Mathf.Abs(v);
        float limit = axis < 3 ? 0.40f : 1.5f;
        if (raw < -limit || raw > limit)
        {
            derivativeV = 0f; derivativeU = 0f;
            return Mathf.Clamp(raw, -limit, limit);
        }
        derivativeV = 5f * fitted[axis, 1] + 50f * fitted[axis, 3] * Mathf.Abs(v);
        derivativeU = 0.2f * fitted[axis, 2];
        return raw;
    }

    void SolveMpc(float t, bool useResidual)
    {
        var timer = System.Diagnostics.Stopwatch.StartNew();
        for (int axis = 0; axis < AxisCount; axis++)
        {
            var sequence = new float[Horizon];
            for (int k = 0; k < Horizon - 1; k++) sequence[k] = warm[axis, k + 1];
            sequence[Horizon - 1] = warm[axis, Horizon - 1];
            var gradient = new float[Horizon];
            float currentCost = Rollout(axis, t, stateP[axis], stateV[axis],
                sequence, useResidual, gradient);
            predictedInitialCost[axis] = currentCost;
            float nominalStep = axis < 3 ? 1.5f : 0.018f;
            for (int iteration = 0; iteration < GradientIterations; iteration++)
            {
                bool accepted = false;
                float step = nominalStep;
                for (int search = 0; search < 8; search++)
                {
                    var candidate = new float[Horizon];
                    float change = 0f;
                    for (int k = 0; k < Horizon; k++)
                    {
                        candidate[k] = Mathf.Clamp(sequence[k] - step * gradient[k],
                            -WrenchLimit[axis], WrenchLimit[axis]);
                        change += Mathf.Abs(candidate[k] - sequence[k]);
                    }
                    if (change < 1e-5f) break;
                    float nextCost = Rollout(axis, t, stateP[axis], stateV[axis],
                        candidate, useResidual, null);
                    if (nextCost < currentCost - 1e-6f)
                    {
                        sequence = candidate;
                        currentCost = Rollout(axis, t, stateP[axis], stateV[axis],
                            sequence, useResidual, gradient);
                        accepted = true;
                        break;
                    }
                    step *= 0.5f;
                }
                if (!accepted) break;
            }
            predictedFinalCost[axis] = currentCost;
            wrench[axis] = sequence[0];
            for (int k = 0; k < Horizon; k++) warm[axis, k] = sequence[k];
        }
        timer.Stop();
        lastSolveMs = (float)timer.Elapsed.TotalMilliseconds;
    }

    float Rollout(int axis, float t, float initialP, float initialV,
        float[] sequence, bool useResidual, float[] gradient)
    {
        var positions = new float[Horizon + 1];
        var velocities = new float[Horizon + 1];
        var derivativeP = new float[Horizon];
        var derivativeV = new float[Horizon];
        var derivativeU = new float[Horizon];
        positions[0] = initialP;
        velocities[0] = initialV;
        float cost = 0f;
        for (int k = 0; k < Horizon; k++)
        {
            float p = positions[k], v = velocities[k], u = sequence[k];
            float residual = 0f, residualV = 0f, residualU = 0f;
            if (useResidual && residualAllowed[axis])
                residual = Residual(axis, v, u, out residualV, out residualU);
            float acceleration = NominalAcceleration(axis, p, v, u) + residual;
            velocities[k + 1] = v + ControlDt * acceleration;
            positions[k + 1] = p + ControlDt * velocities[k + 1];
            derivativeP[k] = ControlDt * (-spring[axis] / mass[axis]);
            derivativeV[k] = 1f + ControlDt *
                ((-linearDrag[axis] - 2f * quadraticDrag[axis] * Mathf.Abs(v)) / mass[axis] + residualV);
            derivativeU[k] = ControlDt * (1f / mass[axis] + residualU);
            float futureT = t + ControlDt * (k + 1);
            float positionError = positions[k + 1] - Reference(axis, futureT);
            float velocityError = velocities[k + 1] - ReferenceVelocity(axis, futureT);
            float terminal = k == Horizon - 1 ? 3f : 1f;
            cost += terminal * (PositionWeight[axis] * positionError * positionError +
                VelocityWeight[axis] * velocityError * velocityError) +
                EffortWeight[axis] * u * u;
        }
        if (gradient == null) return cost;
        float adjointP = 0f, adjointV = 0f;
        for (int k = Horizon - 1; k >= 0; k--)
        {
            float futureT = t + ControlDt * (k + 1);
            float terminal = k == Horizon - 1 ? 3f : 1f;
            float nextP = 2f * terminal * PositionWeight[axis] *
                (positions[k + 1] - Reference(axis, futureT)) + adjointP;
            float nextV = 2f * terminal * VelocityWeight[axis] *
                (velocities[k + 1] - ReferenceVelocity(axis, futureT)) + adjointV;
            gradient[k] = 2f * EffortWeight[axis] * sequence[k] +
                (ControlDt * nextP + nextV) * derivativeU[k];
            adjointP = nextP * (1f + ControlDt * derivativeP[k]) + nextV * derivativeP[k];
            adjointV = (ControlDt * nextP + nextV) * derivativeV[k];
        }
        return cost;
    }

    void Allocate(float[] desired, float[] result)
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
        var normal = new double[AxisCount, AxisCount];
        var rhs = new double[AxisCount];
        for (int a = 0; a < AxisCount; a++)
        {
            rhs[a] = desired[a];
            for (int b = 0; b < AxisCount; b++)
            {
                normal[a, b] = a == b ? 0.04 : 0d;
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
            residualSquared += (delivered - desired[a]) * (delivered - desired[a]);
        }
        lastAllocationResidual = Mathf.Sqrt(residualSquared);
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
        foreach (var motor in thrusters) motor.ApplyForceRequest(0f);
        body.position = new Vector3(0f, -1.1f, 0f);
        body.rotation = Quaternion.Euler(5f, 7f, -4f);
        body.linearVelocity = Vector3.zero;
        body.angularVelocity = Vector3.zero;
        body.Sleep();
        Physics.SyncTransforms();
        hydro.ResetBackendState();
        Array.Clear(wrench, 0, AxisCount);
        Array.Clear(warm, 0, warm.Length);
        nextControl = Time.fixedTime;
        nextSample = Time.fixedTime;
        previousSampleTime = -1f;
    }

    void AdvancePhase()
    {
        if (phase == Phase.Identify)
        {
            FitResidual();
            phase = Phase.SettleBaseline;
        }
        else if (phase == Phase.SettleBaseline)
        {
            phase = Phase.BaselineMpc;
            noise = new System.Random(evaluationSeed);
        }
        else if (phase == Phase.BaselineMpc)
            phase = Phase.SettleKoopman;
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
            Debug.Log($"FINSSIM_KOOPMAN_MPC_RESULT baselineRmseM={baseline:F4} koopmanRmseM={koopman:F4} " +
                $"improvementPct={100d*(baseline-koopman)/Math.Max(1e-9,baseline):F2} " +
                $"samplesBaseline={baselineSamples} samplesKoopman={koopmanSamples}");
            trialCsv?.Dispose(); identifyCsv?.Dispose();
            body.isKinematic = true;
            if (Environment.GetEnvironmentVariable("FINSSIM_KOOPMAN_MPC_AUTOQUIT") == "1")
                Application.Quit(0);
            return;
        }
        ResetVehicle();
        phaseStart = Time.fixedTime;
        Debug.Log("FINSSIM_KOOPMAN_MPC_PHASE " + phase);
    }

    void OnDisable()
    {
        trialCsv?.Dispose(); identifyCsv?.Dispose();
    }
}
