using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using FinsSim.Actuators;
using FinsSim.Hydrodynamics;
using UnityEngine;

// Self-contained, paired Unity physics experiment. The fitted operator maps
// lifted (velocity, requested wrench) features to the one-step acceleration
// residual of the manually calibrated Fossen model. Identification and
// evaluation use disjoint intervals; the fitted coefficients are frozen.
[DefaultExecutionOrder(-50)]
public sealed class FinsRovKoopmanSmoke : MonoBehaviour
{
    enum Phase { Identify, SettleBaseline, Baseline, SettleKoopman, Koopman, Done }
    const int FeatureCount = 7;
    const float IdentifySeconds = 24f;
    const float EvaluationSeconds = 32f;
    const float SettleSeconds = 2f;
    const float SamplePeriod = 0.1f;
    const float Period = 32f;
    const float NoiseStdN = 0.20f;
    const float ThrustScale = 0.78f;
    const float CorrectionGain = 0.85f;
    static readonly string[] Names =
    {
        "Vertical1", "Vertical2", "Vertical3", "Vertical4",
        "Horizontal1", "Horizontal2", "Horizontal3", "Horizontal4"
    };

    readonly Thruster[] thrusters = new Thruster[8];
    readonly List<double[]> features = new List<double[]>();
    readonly List<Vector2> targets = new List<Vector2>();
    readonly double[,] fitted = new double[2, FeatureCount];
    readonly float[] requested = new float[8];
    Rigidbody body;
    HydrodynamicsController hydro;
    StreamWriter trialCsv;
    StreamWriter identifyCsv;
    Phase phase;
    float phaseStart;
    float nextSample;
    float previousSampleTime = -1f;
    Vector2 previousVelocity;
    Vector2 previousWrench;
    Vector2 nominalMass;
    Vector2 nominalLinearDrag;
    Vector2 nominalQuadraticDrag;
    System.Random noise;
    string artifactDirectory;
    double sumSquaredBaseline;
    double sumSquaredKoopman;
    int countBaseline;
    int countKoopman;
    float maxCommand;

    public string ViewerPhase => phase.ToString();
    public float ViewerPhaseTime => phase == Phase.Done ? EvaluationSeconds :
        Mathf.Max(0f, Time.fixedTime - phaseStart);
    public float ViewerPhaseDuration => phase == Phase.Identify ? IdentifySeconds :
        phase == Phase.Baseline || phase == Phase.Koopman || phase == Phase.Done ?
        EvaluationSeconds : SettleSeconds;
    public bool ViewerEvaluating => phase == Phase.Baseline || phase == Phase.Koopman;
    public Vector2 ViewerReference { get; private set; }
    public float ViewerError { get; private set; }
    public float ViewerTilt { get; private set; }
    public float ViewerRoll { get; private set; }
    public float ViewerPitch { get; private set; }
    public float ViewerYaw { get; private set; }
    public float ViewerBaselineRmse => countBaseline > 0 ?
        (float)Math.Sqrt(sumSquaredBaseline / countBaseline) : 0f;
    public float ViewerKoopmanRmse => countKoopman > 0 ?
        (float)Math.Sqrt(sumSquaredKoopman / countKoopman) : 0f;

    void Start()
    {
        body = GetComponent<Rigidbody>();
        hydro = GetComponent<HydrodynamicsController>();
        if (body == null || hydro == null || hydro.profile == null)
            throw new InvalidOperationException("Koopman smoke requires the calibrated FinsROV Rigidbody and Fossen profile");
        foreach (var thruster in GetComponentsInChildren<Thruster>(true))
            for (int i = 0; i < Names.Length; i++)
                if (thruster.name == Names[i]) thrusters[i] = thruster;
        foreach (var thruster in thrusters)
            if (thruster == null) throw new InvalidOperationException("Eight calibrated thrusters were not resolved");

        var nominal = hydro.profile;
        nominalMass = new Vector2(body.mass + nominal.addedMassDiagonal.u,
            body.mass + nominal.addedMassDiagonal.v);
        nominalLinearDrag = new Vector2(nominal.linearDamping.u, nominal.linearDamping.v);
        nominalQuadraticDrag = new Vector2(nominal.quadraticDamping.u, nominal.quadraticDamping.v);

        // Only the runtime copy is perturbed. The source calibrated asset and
        // the controller's nominal parameters stay untouched.
        var truth = hydro.RuntimeProfile;
        truth.addedMassDiagonal.u *= 1.35f;
        truth.addedMassDiagonal.v *= 1.25f;
        truth.linearDamping.u *= 1.55f;
        truth.linearDamping.v *= 1.45f;
        truth.quadraticDamping.u += 12f;
        truth.quadraticDamping.v += 16f;
        hydro.ResetBackendState();

        artifactDirectory = Environment.GetEnvironmentVariable("FINSSIM_KOOPMAN_DIR");
        if (string.IsNullOrEmpty(artifactDirectory))
            artifactDirectory = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts/koopman-smoke"));
        Directory.CreateDirectory(artifactDirectory);
        trialCsv = new StreamWriter(Path.Combine(artifactDirectory, "trials.csv"), false);
        trialCsv.WriteLine("phase,t_s,x_m,y_m,z_m,ref_x_m,ref_z_m,error_m,vx_mps,vz_mps,requested_fx_n,requested_fz_n,applied_h1_n,hydro_force_n,roll_x_deg,pitch_z_deg,yaw_y_deg,tilt_deg,angular_speed_degps");
        identifyCsv = new StreamWriter(Path.Combine(artifactDirectory, "identification.csv"), false);
        identifyCsv.WriteLine("t_s,vx_mps,vz_mps,fx_n,fz_n,residual_ax_mps2,residual_az_mps2");
        phase = Phase.Identify;
        phaseStart = Time.fixedTime;
        noise = new System.Random(24001);
        Debug.Log("FINSSIM_KOOPMAN_SMOKE_START " + artifactDirectory);
    }

    void FixedUpdate()
    {
        if (phase == Phase.Done) return;
        float t = Time.fixedTime - phaseStart;
        float limit = phase == Phase.Identify ? IdentifySeconds :
            phase == Phase.Baseline || phase == Phase.Koopman ? EvaluationSeconds : SettleSeconds;
        if (t >= limit)
        {
            AdvancePhase();
            if (phase == Phase.Done) return;
            t = 0f;
        }

        Vector3 p = body.position;
        Vector3 v = body.linearVelocity;
        Vector2 horizontalVelocity = new Vector2(v.x, v.z);
        Vector2 requestedWrench = Vector2.zero;
        Vector2 reference = Vector2.zero;
        if (phase == Phase.Identify)
        {
            requestedWrench = new Vector2(
                4f * Mathf.Sin(0.77f * t) + 2f * Mathf.Sin(1.61f * t),
                4f * Mathf.Cos(0.57f * t) + 2f * Mathf.Sin(1.29f * t));
        }
        else if (phase == Phase.Baseline || phase == Phase.Koopman)
        {
            float omega = 2f * Mathf.PI / Period;
            reference = new Vector2(0.60f * Mathf.Sin(omega * t),
                0.35f * Mathf.Sin(2f * omega * t));
            Vector2 refVelocity = new Vector2(0.60f * omega * Mathf.Cos(omega * t),
                0.70f * omega * Mathf.Cos(2f * omega * t));
            Vector2 refAcceleration = new Vector2(-0.60f * omega * omega * Mathf.Sin(omega * t),
                -1.40f * omega * omega * Mathf.Sin(2f * omega * t));
            Vector2 positionError = reference - new Vector2(p.x, p.z);
            Vector2 targetAcceleration = refAcceleration + 0.95f * positionError +
                1.6f * (refVelocity - horizontalVelocity);
            requestedWrench = NominalForce(horizontalVelocity, targetAcceleration);
            if (phase == Phase.Koopman)
            {
                Vector2 residual = PredictResidual(horizontalVelocity, requestedWrench);
                requestedWrench -= CorrectionGain * Vector2.Scale(nominalMass, residual);
            }
        }

        ViewerReference = reference;
        ViewerError = Vector2.Distance(new Vector2(p.x, p.z), reference);
        ViewerTilt = Vector3.Angle(body.rotation * Vector3.up, Vector3.up);
        Vector3 viewerEuler = body.rotation.eulerAngles;
        ViewerRoll = Mathf.DeltaAngle(0f, viewerEuler.x);
        ViewerPitch = Mathf.DeltaAngle(0f, viewerEuler.z);
        ViewerYaw = Mathf.DeltaAngle(0f, viewerEuler.y);

        float depthForce = 18f * (-1.2f - p.y) - 9f * v.y;
        float yaw = Mathf.DeltaAngle(0f, transform.eulerAngles.y) * Mathf.Deg2Rad;
        float yawMoment = -4f * yaw - 1.8f * body.angularVelocity.y;
        Allocate(new[] { requestedWrench.x, depthForce, requestedWrench.y, yawMoment }, requested);
        for (int i = 0; i < thrusters.Length; i++)
        {
            float command = phase == Phase.SettleBaseline || phase == Phase.SettleKoopman
                ? 0f : ThrustScale * requested[i] + NoiseStdN * Gaussian(noise);
            thrusters[i].ApplyForceRequest(command);
            maxCommand = Mathf.Max(maxCommand, Mathf.Abs(command));
        }

        if (Time.fixedTime + 0.0001f < nextSample) return;
        nextSample = Time.fixedTime + SamplePeriod;
        if (phase == Phase.Identify)
        {
            if (previousSampleTime >= 0f)
            {
                float dt = Time.fixedTime - previousSampleTime;
                Vector2 observedAcceleration = (horizontalVelocity - previousVelocity) / dt;
                Vector2 residual = observedAcceleration - NominalAcceleration(previousVelocity, previousWrench);
                features.Add(Lift(previousVelocity, previousWrench));
                targets.Add(residual);
                identifyCsv.WriteLine(string.Format(CultureInfo.InvariantCulture,
                    "{0:F3},{1:F5},{2:F5},{3:F5},{4:F5},{5:F5},{6:F5}",
                    t, previousVelocity.x, previousVelocity.y, previousWrench.x,
                    previousWrench.y, residual.x, residual.y));
                identifyCsv.Flush();
            }
            previousSampleTime = Time.fixedTime;
            previousVelocity = horizontalVelocity;
            previousWrench = requestedWrench;
        }
        if (phase == Phase.Baseline || phase == Phase.Koopman)
        {
            float error = Vector2.Distance(new Vector2(p.x, p.z), reference);
            if (phase == Phase.Baseline) { sumSquaredBaseline += error * error; countBaseline++; }
            else { sumSquaredKoopman += error * error; countKoopman++; }
            Vector3 euler = body.rotation.eulerAngles;
            float roll = Mathf.DeltaAngle(0f, euler.x);
            float pitch = Mathf.DeltaAngle(0f, euler.z);
            float heading = Mathf.DeltaAngle(0f, euler.y);
            float tilt = Vector3.Angle(body.rotation * Vector3.up, Vector3.up);
            float angularSpeed = body.angularVelocity.magnitude * Mathf.Rad2Deg;
            trialCsv.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "{0},{1:F3},{2:F5},{3:F5},{4:F5},{5:F5},{6:F5},{7:F5},{8:F5},{9:F5},{10:F5},{11:F5},{12:F5},{13:F5},{14:F5},{15:F5},{16:F5},{17:F5},{18:F5}",
                phase, t, p.x, p.y, p.z, reference.x, reference.y, error,
                v.x, v.z, requestedWrench.x, requestedWrench.y,
                thrusters[4].LastForceRequest, hydro.LastWrench.WorldForce.magnitude,
                roll, pitch, heading, tilt, angularSpeed));
            trialCsv.Flush();
        }
    }

    Vector2 NominalForce(Vector2 velocity, Vector2 acceleration)
    {
        return Vector2.Scale(nominalMass, acceleration) +
            Vector2.Scale(nominalLinearDrag, velocity) +
            Vector2.Scale(nominalQuadraticDrag,
                new Vector2(velocity.x * Mathf.Abs(velocity.x), velocity.y * Mathf.Abs(velocity.y)));
    }

    Vector2 NominalAcceleration(Vector2 velocity, Vector2 force)
    {
        Vector2 drag = Vector2.Scale(nominalLinearDrag, velocity) +
            Vector2.Scale(nominalQuadraticDrag,
                new Vector2(velocity.x * Mathf.Abs(velocity.x), velocity.y * Mathf.Abs(velocity.y)));
        return new Vector2((force.x - drag.x) / nominalMass.x,
            (force.y - drag.y) / nominalMass.y);
    }

    static double[] Lift(Vector2 velocity, Vector2 force)
    {
        return new[] { 1.0, 5.0 * velocity.x, 5.0 * velocity.y,
            0.2 * force.x, 0.2 * force.y,
            25.0 * velocity.x * Math.Abs(velocity.x),
            25.0 * velocity.y * Math.Abs(velocity.y) };
    }

    Vector2 PredictResidual(Vector2 velocity, Vector2 force)
    {
        double[] phi = Lift(velocity, force);
        double x = 0, z = 0;
        for (int j = 0; j < FeatureCount; j++)
        {
            x += fitted[0, j] * phi[j];
            z += fitted[1, j] * phi[j];
        }
        return new Vector2(Mathf.Clamp((float)x, -0.35f, 0.35f),
            Mathf.Clamp((float)z, -0.35f, 0.35f));
    }

    void FitKoopmanResidual()
    {
        if (features.Count < 80) throw new InvalidOperationException("Insufficient Koopman identification samples");
        int trainCount = features.Count * 4 / 5;
        var gram = new double[FeatureCount, FeatureCount];
        var rhs = new double[2, FeatureCount];
        for (int i = 0; i < trainCount; i++)
            for (int a = 0; a < FeatureCount; a++)
            {
                rhs[0, a] += features[i][a] * targets[i].x;
                rhs[1, a] += features[i][a] * targets[i].y;
                for (int b = 0; b < FeatureCount; b++)
                    gram[a, b] += features[i][a] * features[i][b];
            }
        for (int a = 0; a < FeatureCount; a++) gram[a, a] += a == 0 ? 0.001 : 0.03;
        for (int output = 0; output < 2; output++)
        {
            var matrix = (double[,])gram.Clone();
            var vector = new double[FeatureCount];
            for (int a = 0; a < FeatureCount; a++) vector[a] = rhs[output, a];
            var solution = Solve(matrix, vector);
            for (int a = 0; a < FeatureCount; a++) fitted[output, a] = solution[a];
        }
        double baselineError = 0, fittedError = 0;
        for (int i = trainCount; i < features.Count; i++)
        {
            double[] phi = features[i];
            for (int output = 0; output < 2; output++)
            {
                double truth = output == 0 ? targets[i].x : targets[i].y;
                double prediction = 0;
                for (int a = 0; a < FeatureCount; a++) prediction += fitted[output, a] * phi[a];
                baselineError += truth * truth;
                fittedError += (prediction - truth) * (prediction - truth);
            }
        }
        Debug.Log($"FINSSIM_KOOPMAN_FIT samples={trainCount} holdout={features.Count - trainCount} " +
            $"holdoutMseRatio={fittedError / Math.Max(1e-12, baselineError):F4}");
    }

    static double[] Solve(double[,] a, double[] b)
    {
        int n = b.Length;
        for (int column = 0; column < n; column++)
        {
            int pivot = column;
            for (int row = column + 1; row < n; row++)
                if (Math.Abs(a[row, column]) > Math.Abs(a[pivot, column])) pivot = row;
            if (Math.Abs(a[pivot, column]) < 1e-10) throw new InvalidOperationException("Singular Koopman fit");
            for (int k = column; k < n; k++)
            {
                double tmp = a[column, k]; a[column, k] = a[pivot, k]; a[pivot, k] = tmp;
            }
            double value = b[column]; b[column] = b[pivot]; b[pivot] = value;
            double scale = a[column, column];
            for (int k = column; k < n; k++) a[column, k] /= scale;
            b[column] /= scale;
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

    void Allocate(float[] desiredWrench, float[] result)
    {
        var basis = new float[4, 8];
        for (int i = 0; i < 8; i++)
        {
            Vector3 direction = thrusters[i].transform.TransformDirection(
                thrusters[i].LocalForceDirection.normalized);
            Vector3 offset = thrusters[i].transform.position - body.worldCenterOfMass;
            basis[0, i] = direction.x;
            basis[1, i] = direction.y;
            basis[2, i] = direction.z;
            basis[3, i] = Vector3.Cross(offset, direction).y;
        }
        var normal = new double[4, 4];
        var rhs = new double[4];
        for (int a = 0; a < 4; a++)
        {
            rhs[a] = desiredWrench[a];
            for (int b = 0; b < 4; b++)
            {
                normal[a, b] = a == b ? 0.02 : 0.0;
                for (int i = 0; i < 8; i++) normal[a, b] += basis[a, i] * basis[b, i];
            }
        }
        double[] multiplier = Solve(normal, rhs);
        for (int i = 0; i < 8; i++)
        {
            double command = 0;
            for (int a = 0; a < 4; a++) command += basis[a, i] * multiplier[a];
            result[i] = Mathf.Clamp((float)command,
                -Mathf.Abs(thrusters[i].MaxReverseForceN), Mathf.Abs(thrusters[i].MaxForwardForceN));
        }
    }

    static float Gaussian(System.Random random)
    {
        double a = Math.Max(1e-9, random.NextDouble());
        return (float)(Math.Sqrt(-2.0 * Math.Log(a)) * Math.Cos(2.0 * Math.PI * random.NextDouble()));
    }

    void ResetVehicle()
    {
        foreach (var thruster in thrusters) thruster.ApplyForceRequest(0f);
        body.position = new Vector3(0f, -1.2f, 0f);
        body.rotation = Quaternion.identity;
        body.linearVelocity = Vector3.zero;
        body.angularVelocity = Vector3.zero;
        body.Sleep();
        Physics.SyncTransforms();
        hydro.ResetBackendState();
        nextSample = Time.fixedTime;
        previousSampleTime = -1f;
    }

    void AdvancePhase()
    {
        if (phase == Phase.Identify)
        {
            FitKoopmanResidual();
            phase = Phase.SettleBaseline;
        }
        else if (phase == Phase.SettleBaseline)
        {
            phase = Phase.Baseline;
            noise = new System.Random(9091);
        }
        else if (phase == Phase.Baseline)
            phase = Phase.SettleKoopman;
        else if (phase == Phase.SettleKoopman)
        {
            phase = Phase.Koopman;
            noise = new System.Random(9091);
        }
        else
        {
            phase = Phase.Done;
            foreach (var thruster in thrusters) thruster.ApplyForceRequest(0f);
            double baseRmse = Math.Sqrt(sumSquaredBaseline / Math.Max(1, countBaseline));
            double koopmanRmse = Math.Sqrt(sumSquaredKoopman / Math.Max(1, countKoopman));
            Debug.Log($"FINSSIM_KOOPMAN_SMOKE_RESULT baselineRmseM={baseRmse:F4} koopmanRmseM={koopmanRmse:F4} " +
                $"improvementPct={100.0 * (baseRmse - koopmanRmse) / Math.Max(baseRmse, 1e-9):F2} " +
                $"samplesBaseline={countBaseline} samplesKoopman={countKoopman} maxCommandN={maxCommand:F3}");
            trialCsv?.Dispose(); identifyCsv?.Dispose();
            body.isKinematic = true;
            if (Environment.GetEnvironmentVariable("FINSSIM_KOOPMAN_AUTOQUIT") == "1")
                Application.Quit(0);
            return;
        }
        ResetVehicle();
        phaseStart = Time.fixedTime;
        Debug.Log("FINSSIM_KOOPMAN_PHASE " + phase);
    }

    void OnDisable()
    {
        trialCsv?.Dispose(); identifyCsv?.Dispose();
    }
}
