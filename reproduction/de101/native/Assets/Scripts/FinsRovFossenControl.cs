using System;
using System.Globalization;
using System.IO;
using FinsSim.Actuators;
using FinsSim.Hydrodynamics;
using UnityEngine;
using UnityEngine.InputSystem;

public sealed class FinsRovFossenControl : MonoBehaviour
{
    public bool rosMode;
    private static readonly string[] ThrusterNames =
    {
        "Vertical1", "Vertical2", "Vertical3", "Vertical4",
        "Horizontal1", "Horizontal2", "Horizontal3", "Horizontal4"
    };

    private readonly Thruster[] thrusters = new Thruster[8];
    private Rigidbody body;
    private HydrodynamicsController hydro;
    private StreamWriter telemetry;
    private float nextSampleTime;
    private float startTime;
    private float initialX;
    private float maxHydroForce;
    private float maxThrusterForce;
    private bool autoTest;
    private string telemetryPath;

    private void Start()
    {
        body = GetComponent<Rigidbody>();
        hydro = GetComponent<HydrodynamicsController>();
        foreach (var thruster in GetComponentsInChildren<Thruster>(true))
            for (int i = 0; i < thrusters.Length; i++)
                if (thruster.name == ThrusterNames[i]) thrusters[i] = thruster;
        foreach (var thruster in thrusters)
            if (thruster == null) throw new InvalidOperationException("FinsROV eight-thruster order is incomplete");
        autoTest = Environment.GetEnvironmentVariable("FINSSIM_FINSROV_AUTOTEST") == "1";
        startTime = Time.time;
        initialX = body.position.x;
        string outputDir = Environment.GetEnvironmentVariable("FINSSIM_TELEMETRY_DIR");
        if (string.IsNullOrEmpty(outputDir))
            outputDir = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts"));
        Directory.CreateDirectory(outputDir);
        telemetryPath = Path.Combine(outputDir, "finsrov-fossen-telemetry.csv");
        telemetry = new StreamWriter(telemetryPath, false);
        telemetry.WriteLine("time_s,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps,hydro_force_n,hydro_torque_nm,thruster_force_n");
        Debug.Log("FINSSIM_FINSROV_TELEMETRY " + telemetryPath);
    }

    private void FixedUpdate()
    {
        float elapsed = Time.time - startTime;
        float requested = autoTest
            ? (elapsed >= 1f && elapsed < 3f ? 4f : 0f)
            : (Keyboard.current != null && Keyboard.current.wKey.isPressed ? 4f :
               Keyboard.current != null && Keyboard.current.sKey.isPressed ? -4f : 0f);
        // A single calibrated horizontal thruster is enough to demonstrate a
        // real off-center force and resulting six-DOF response.
        if (!rosMode) thrusters[4].ApplyForceRequest(requested);
        float hydroForce = hydro.LastWrench.WorldForce.magnitude;
        maxHydroForce = Mathf.Max(maxHydroForce, hydroForce);
        maxThrusterForce = Mathf.Max(maxThrusterForce, Mathf.Abs(thrusters[4].LastForceRequest));

        if (Time.time >= nextSampleTime)
        {
            nextSampleTime = Time.time + 0.1f;
            Vector3 p = body.position;
            Vector3 v = body.linearVelocity;
            telemetry.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "{0:F3},{1:F5},{2:F5},{3:F5},{4:F5},{5:F5},{6:F5},{7:F5},{8:F5},{9:F5}",
                elapsed, p.x, p.y, p.z, v.x, v.y, v.z,
                hydroForce, hydro.LastWrench.WorldTorque.magnitude, thrusters[4].LastForceRequest));
            telemetry.Flush();
        }

        if (autoTest && elapsed > 4f)
        {
            bool passed = maxHydroForce > 1f && maxThrusterForce > 1f &&
                Vector3.Distance(body.position, new Vector3(initialX, -1.2f, 0f)) > 0.02f;
            Debug.Log($"FINSSIM_FINSROV_FOSSEN_RESULT passed={passed} displacement={Vector3.Distance(body.position, new Vector3(initialX, -1.2f, 0f)):F3} maxHydroForce={maxHydroForce:F3} maxThrusterForce={maxThrusterForce:F3}");
            Application.Quit(passed ? 0 : 2);
            autoTest = false;
        }
    }

    private void OnGUI()
    {
        if (body == null) return;
        GUI.Box(new Rect(12, 12, 510, 92), "FinsROV calibrated Fossen / Rigidbody");
        GUI.Label(new Rect(24, 39, 490, 20), rosMode
            ? "ROS2: /sim/finsrov/thrusters_out (8 forces in N)"
            : "W/S: Horizontal1 thrust +/-4 N");
        GUI.Label(new Rect(24, 62, 490, 20), $"Position {body.position:F2} m   Velocity {body.linearVelocity:F2} m/s");
        GUI.Label(new Rect(24, 83, 490, 20), $"Hydrodynamic force {hydro.LastWrench.WorldForce.magnitude:F2} N");
    }

    private void OnDisable()
    {
        telemetry?.Dispose();
        telemetry = null;
    }
}
