using System;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.InputSystem;

public sealed class BlueRovManualControl : MonoBehaviour
{
    [SerializeField] private float forceNewtons = 50f;
    [SerializeField] private float torqueNewtonMeters = 8f;

    private ArticulationBody baseBody;
    private StreamWriter telemetry;
    private float nextSampleTime;
    private string telemetryPath;
    private float autoTestSeconds;

    private void Start()
    {
        foreach (var body in GetComponentsInChildren<ArticulationBody>())
        {
            body.useGravity = false;
            if (body.isRoot) baseBody = body;
        }
        if (baseBody == null)
        {
            Debug.LogError("BlueROV2 has no root ArticulationBody");
            enabled = false;
            return;
        }
        baseBody.immovable = false;

        string outputDir = Environment.GetEnvironmentVariable("FINSSIM_TELEMETRY_DIR");
        if (string.IsNullOrEmpty(outputDir))
            outputDir = Path.GetFullPath(Path.Combine(Application.dataPath, "../artifacts"));
        Directory.CreateDirectory(outputDir);
        telemetryPath = Path.Combine(outputDir, "bluerov2-telemetry.csv");
        telemetry = new StreamWriter(telemetryPath, false);
        telemetry.WriteLine("time_s,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps");
        Debug.Log("BlueROV2 telemetry: " + telemetryPath);
        float.TryParse(Environment.GetEnvironmentVariable("FINSSIM_AUTOTEST_SECONDS"),
            NumberStyles.Float, CultureInfo.InvariantCulture, out autoTestSeconds);
    }

    private void FixedUpdate()
    {
        if (autoTestSeconds > 0f)
        {
            if (Time.time >= autoTestSeconds)
            {
                Application.Quit();
                return;
            }
            baseBody.AddForce(Vector3.right * forceNewtons);
        }
        var keyboard = Keyboard.current;
        if (keyboard != null)
        {
            Vector3 direction = Vector3.zero;
            if (keyboard.wKey.isPressed) direction += Vector3.forward;
            if (keyboard.sKey.isPressed) direction += Vector3.back;
            if (keyboard.aKey.isPressed) direction += Vector3.left;
            if (keyboard.dKey.isPressed) direction += Vector3.right;
            if (keyboard.rKey.isPressed) direction += Vector3.up;
            if (keyboard.fKey.isPressed) direction += Vector3.down;
            if (direction.sqrMagnitude > 0f)
                baseBody.AddForce(transform.TransformDirection(direction.normalized) * forceNewtons);
            if (keyboard.qKey.isPressed) baseBody.AddTorque(Vector3.down * torqueNewtonMeters);
            if (keyboard.eKey.isPressed) baseBody.AddTorque(Vector3.up * torqueNewtonMeters);
        }

        if (Time.time < nextSampleTime) return;
        nextSampleTime = Time.time + 0.1f;
        Vector3 p = baseBody.transform.position;
        Vector3 v = baseBody.linearVelocity;
        telemetry.WriteLine(string.Format(CultureInfo.InvariantCulture,
            "{0:F3},{1:F5},{2:F5},{3:F5},{4:F5},{5:F5},{6:F5}",
            Time.time, p.x, p.y, p.z, v.x, v.y, v.z));
        telemetry.Flush();
    }

    private void OnGUI()
    {
        if (baseBody == null) return;
        GUI.Box(new Rect(12, 12, 490, 94), "BlueROV2 URDF control / telemetry");
        GUI.Label(new Rect(24, 38, 470, 22), "W/S forward, A/D sideways, R/F vertical, Q/E yaw");
        GUI.Label(new Rect(24, 61, 470, 22),
            $"Position {baseBody.transform.position:F2} m   Velocity {baseBody.linearVelocity:F2} m/s");
        GUI.Label(new Rect(24, 82, 470, 20), telemetryPath);
    }

    private void OnDisable()
    {
        telemetry?.Dispose();
        telemetry = null;
    }
}
