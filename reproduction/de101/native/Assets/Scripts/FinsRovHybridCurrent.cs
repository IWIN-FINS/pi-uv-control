using System;
using System.Globalization;
using System.IO;
using FinsSim.Hydrodynamics;
using UnityEngine;

// UUV-Simulator-inspired water-current input for the existing Fossen backend.
// The controller consumes this as water velocity and computes its own relative-
// flow damping, added-mass, and Coriolis terms. No extra wrench is double-added.
public sealed partial class FinsRovHybridBasisSmoke
{
    enum CurrentMode { Off, Constant, GaussMarkov }

    CurrentMode currentMode;
    string currentModeName = "off";
    Vector3 currentMeanWorld;
    Vector3 currentSigmaWorld = new Vector3(0.03f, 0f, 0.03f);
    float currentTimeConstantSeconds = 8f;
    float currentMaxSpeedMps = 0.35f;
    Vector3 currentStateWorld;
    int currentSeed = int.MinValue;
    int currentStep;
    double currentElapsedSeconds;
    bool currentEnabled;
    System.Random currentRandom;
    FlatWaterKinematicsProvider currentWaterProvider;
    StreamWriter currentTrace;

    void InitializeCurrentModel()
    {
        currentModeName = (Environment.GetEnvironmentVariable("FINSSIM_HYBRID_CURRENT_MODE") ?? "off")
            .Trim().ToLowerInvariant();
        switch (currentModeName)
        {
            case "off": currentMode = CurrentMode.Off; break;
            case "constant": currentMode = CurrentMode.Constant; break;
            case "gauss_markov": currentMode = CurrentMode.GaussMarkov; break;
            default: throw new ArgumentException("Unknown current mode: " + currentModeName);
        }

        currentMeanWorld = ReadCurrentVector("FINSSIM_HYBRID_CURRENT_MEAN_XYZ_MPS", Vector3.zero);
        currentSigmaWorld = ReadCurrentVector("FINSSIM_HYBRID_CURRENT_SIGMA_XYZ_MPS",
            new Vector3(0.03f, 0f, 0.03f));
        currentTimeConstantSeconds = ReadCurrentScalar("FINSSIM_HYBRID_CURRENT_TAU_S", 8f);
        currentMaxSpeedMps = ReadCurrentScalar("FINSSIM_HYBRID_CURRENT_MAX_SPEED_MPS", 0.35f);
        if (!IsFinite(currentMeanWorld) || !IsFinite(currentSigmaWorld) ||
            !IsFinite(currentTimeConstantSeconds) || !IsFinite(currentMaxSpeedMps))
            throw new ArgumentOutOfRangeException("Current parameters must be finite numbers.");
        if (currentSigmaWorld.x < 0f || currentSigmaWorld.y < 0f || currentSigmaWorld.z < 0f)
            throw new ArgumentOutOfRangeException("Current standard deviations must be non-negative.");
        if (currentTimeConstantSeconds < 0.1f || currentTimeConstantSeconds > 120f)
            throw new ArgumentOutOfRangeException("Current time constant must be in [0.1, 120] seconds.");
        if (currentMaxSpeedMps <= 0f || currentMaxSpeedMps > 2f || currentMeanWorld.magnitude > currentMaxSpeedMps)
            throw new ArgumentOutOfRangeException("Current mean/max speed is outside the supported range.");

        currentWaterProvider = hydro.waterProviderBehaviour as FlatWaterKinematicsProvider;
        if (currentMode != CurrentMode.Off && currentWaterProvider == null)
            throw new InvalidOperationException("Current disturbance requires the scene's FlatWaterKinematicsProvider.");
    }

    static float ReadCurrentScalar(string name, float fallback)
    {
        string value = Environment.GetEnvironmentVariable(name);
        return string.IsNullOrWhiteSpace(value)
            ? fallback
            : float.Parse(value, NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    static Vector3 ReadCurrentVector(string name, Vector3 fallback)
    {
        string value = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(value)) return fallback;
        string[] parts = value.Split(',');
        if (parts.Length != 3)
            throw new FormatException(name + " must contain three comma-separated Unity XYZ values.");
        return new Vector3(
            float.Parse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture),
            float.Parse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture),
            float.Parse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture));
    }

    static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
    static bool IsFinite(Vector3 value) => IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);

    void ResetCurrentProcess(int seed)
    {
        currentSeed = seed;
        currentRandom = new System.Random(seed);
        currentStateWorld = currentMeanWorld;
        currentStep = 0;
        currentElapsedSeconds = 0d;
        currentEnabled = false;
        if (currentMode != CurrentMode.Off && currentWaterProvider != null)
            currentWaterProvider.SetCurrent(Vector3.zero);
    }

    void StepCurrentModel()
    {
        if (currentMode == CurrentMode.Off || currentWaterProvider == null) return;
        bool active = responseActive || phase == Phase.Identify ||
            phase == Phase.BaselineMpc || phase == Phase.KoopmanMpc;
        if (!active)
        {
            if (currentEnabled) currentWaterProvider.SetCurrent(Vector3.zero);
            currentEnabled = false;
            return;
        }

        int seed = responseActive ? responseProbe.seed :
            (phase == Phase.Identify ? trainingSeed : evaluationSeed);
        if (!currentEnabled || currentSeed != seed)
        {
            currentSeed = seed;
            currentRandom = new System.Random(seed);
            currentStateWorld = currentMeanWorld;
            if(responseActive && responseProbe.flowInitial!=null && responseProbe.flowInitial.Length==3)
                currentStateWorld=new Vector3(responseProbe.flowInitial[0],responseProbe.flowInitial[1],responseProbe.flowInitial[2]);
            currentStep = 0;
            currentElapsedSeconds = 0d;
            currentEnabled = true;
        }

        if (currentMode == CurrentMode.Constant)
        {
            currentStateWorld = De82FlowAt(currentElapsedSeconds,currentSeed);
        }
        else
        {
            // Exact discrete first-order Gauss-Markov / Ornstein-Uhlenbeck step.
            // Sigma is the stationary per-axis standard deviation.
            float rho = Mathf.Exp(-Time.fixedDeltaTime / currentTimeConstantSeconds);
            float innovation = Mathf.Sqrt(Mathf.Max(0f, 1f - rho * rho));
            currentStateWorld = currentMeanWorld + rho * (currentStateWorld - currentMeanWorld) +
                innovation * new Vector3(
                    currentSigmaWorld.x * (float)GaussianCurrent(),
                    currentSigmaWorld.y * (float)GaussianCurrent(),
                    currentSigmaWorld.z * (float)GaussianCurrent());
        }
        float speed = currentStateWorld.magnitude;
        if (speed > currentMaxSpeedMps)
            currentStateWorld *= currentMaxSpeedMps / speed;

        currentWaterProvider.SetCurrent(currentStateWorld);
        currentElapsedSeconds += Time.fixedDeltaTime;
        currentStep++;
        if (currentTrace != null)
        {
            currentTrace.WriteLine(currentStep.ToString(CultureInfo.InvariantCulture) + "," +
                currentElapsedSeconds.ToString("R", CultureInfo.InvariantCulture) + "," + phase + "," +
                currentStateWorld.x.ToString("R", CultureInfo.InvariantCulture) + "," +
                currentStateWorld.y.ToString("R", CultureInfo.InvariantCulture) + "," +
                currentStateWorld.z.ToString("R", CultureInfo.InvariantCulture) + "," +
                currentStateWorld.magnitude.ToString("R", CultureInfo.InvariantCulture));
            if (currentStep % 50 == 0) currentTrace.Flush();
        }
    }

    double GaussianCurrent()
    {
        double u1 = Math.Max(1e-12, currentRandom.NextDouble());
        double u2 = currentRandom.NextDouble();
        return Math.Sqrt(-2d * Math.Log(u1)) * Math.Cos(2d * Math.PI * u2);
    }

    void StopCurrentModel()
    {
        currentEnabled = false;
        if (currentMode != CurrentMode.Off && currentWaterProvider != null)
            currentWaterProvider.SetCurrent(Vector3.zero);
    }

    void OpenCurrentArtifacts(string directory, bool append)
    {
        if (currentMode == CurrentMode.Off) return;
        Directory.CreateDirectory(directory);
        string configuration = "{\"schema\":1,\"mode\":\"" + currentModeName +
            "\",\"mean_world_xyz_mps\":[" + Vec(currentMeanWorld) +
            "],\"stationary_sigma_world_xyz_mps\":[" + Vec(currentSigmaWorld) +
            "],\"time_constant_s\":" + currentTimeConstantSeconds.ToString("R", CultureInfo.InvariantCulture) +
            ",\"max_speed_mps\":" + currentMaxSpeedMps.ToString("R", CultureInfo.InvariantCulture) +
            ",\"process\":\"" + (currentMode == CurrentMode.Constant
                ? "constant world-velocity flow"
                : "discrete first-order Gauss-Markov; exact OU coefficient rho=exp(-dt/tau)") + "\"," +
            "\"seed_policy\":\"training seed for identification; evaluation seed reset identically for baseline and learned phases; probe seed per response arm\"," +
            "\"flow_frame\":\"Unity world XYZ; m/s\",\"legacy_synthetic_hydro_scale\":" +
            hydroScale.ToString("R", CultureInfo.InvariantCulture) + "}";
        string configPath = Path.Combine(directory, "current_model_config.json");
        if (File.Exists(configPath))
        {
            if (!append || File.ReadAllText(configPath) != configuration)
                throw new IOException("Refusing to overwrite or change current configuration: " + configPath);
        }
        else File.WriteAllText(configPath, configuration);

        string tracePath = Path.Combine(directory, "current_velocity_trace.csv");
        if (File.Exists(tracePath) && !append)
            throw new IOException("Refusing to overwrite current trace: " + tracePath);
        currentTrace = new StreamWriter(tracePath, append);
        if (!append) currentTrace.WriteLine("step,episode_time_s,phase,flow_x_world_mps,flow_y_world_mps,flow_z_world_mps,speed_mps");
    }

    static string Vec(Vector3 value)
    {
        return value.x.ToString("R", CultureInfo.InvariantCulture) + "," +
            value.y.ToString("R", CultureInfo.InvariantCulture) + "," +
            value.z.ToString("R", CultureInfo.InvariantCulture);
    }

    void CloseCurrentArtifacts()
    {
        currentTrace?.Flush();
        currentTrace?.Dispose();
        currentTrace = null;
    }
}
