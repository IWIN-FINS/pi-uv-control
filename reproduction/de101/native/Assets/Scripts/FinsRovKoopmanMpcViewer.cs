using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

// Read-only experiment display. The trajectory plot reads the same Rigidbody
// pose used by telemetry; it does not participate in control or physics.
[DefaultExecutionOrder(100)]
public sealed class FinsRovKoopmanMpcViewer : MonoBehaviour
{
    readonly List<Vector2> baselinePoints = new List<Vector2>();
    readonly List<Vector2> koopmanPoints = new List<Vector2>();
    FinsRovKoopmanMpcSmoke experiment;
    Rigidbody body;
    Camera followCamera;
    Texture2D panel;
    Texture2D pixel;
    GUIStyle titleStyle;
    GUIStyle bodyStyle;
    GUIStyle smallStyle;
    string captureDirectory;
    bool baselineCaptured;
    bool koopmanCaptured;
    bool finalCaptured;

    void Start()
    {
        experiment = GetComponent<FinsRovKoopmanMpcSmoke>();
        body = GetComponent<Rigidbody>();
        followCamera = GameObject.Find("MPC Follow Camera")?.GetComponent<Camera>();
        if (experiment == null || body == null || followCamera == null)
            throw new InvalidOperationException("MPC viewer scene is incomplete");
        captureDirectory = Environment.GetEnvironmentVariable("FINSSIM_KOOPMAN_MPC_CAPTURE_DIR");
        if (!string.IsNullOrEmpty(captureDirectory)) Directory.CreateDirectory(captureDirectory);
    }

    void FixedUpdate()
    {
        if (experiment == null) return;
        string phase = experiment.ViewerPhase;
        if (experiment.ViewerEvaluating)
        {
            var points = phase == "BaselineMpc" ? baselinePoints : koopmanPoints;
            Vector2 position = experiment.ViewerPosition;
            if (points.Count == 0 || Vector2.Distance(points[points.Count - 1], position) > 0.004f)
                points.Add(position);
        }
        if (string.IsNullOrEmpty(captureDirectory)) return;
        if (phase == "BaselineMpc" && !baselineCaptured && experiment.ViewerPhaseTime >= 12f)
        {
            baselineCaptured = true;
            StartCoroutine(Capture("baseline_mpc.png"));
        }
        if (phase == "KoopmanMpc" && !koopmanCaptured && experiment.ViewerPhaseTime >= 12f)
        {
            koopmanCaptured = true;
            StartCoroutine(Capture("koopman_mpc.png"));
        }
        if (phase == "Done" && !finalCaptured)
        {
            finalCaptured = true;
            StartCoroutine(Capture("final_mpc.png"));
        }
    }

    IEnumerator Capture(string filename)
    {
        yield return new WaitForEndOfFrame();
        ScreenCapture.CaptureScreenshot(Path.Combine(captureDirectory, filename));
    }

    void LateUpdate()
    {
        if (followCamera == null || body == null) return;
        Vector3 focus = body.position;
        followCamera.transform.position = focus + new Vector3(0.83f, 0.48f, -0.88f);
        followCamera.transform.LookAt(focus, Vector3.up);
    }

    void OnGUI()
    {
        if (experiment == null) return;
        if (panel == null)
        {
            panel = Solid(new Color(0.025f, 0.06f, 0.11f, 0.90f));
            pixel = Solid(Color.white);
            titleStyle = new GUIStyle(GUI.skin.label) { fontSize = 21, fontStyle = FontStyle.Bold };
            titleStyle.normal.textColor = Color.white;
            bodyStyle = new GUIStyle(GUI.skin.label) { fontSize = 17 };
            bodyStyle.normal.textColor = Color.white;
            smallStyle = new GUIStyle(GUI.skin.label) { fontSize = 15 };
            smallStyle.normal.textColor = new Color(0.78f, 0.85f, 0.92f);
        }
        float scale = Mathf.Max(0.8f, Screen.height / 800f);
        GUI.matrix = Matrix4x4.Scale(new Vector3(scale, scale, 1f));
        float width = Screen.width / scale, height = Screen.height / scale;
        DrawTrajectory(width, height);
        string phase = experiment.ViewerPhase;
        string title = phase == "Identify" ? "01 / 03  IDENTIFY 6-AXIS RESIDUAL" :
            phase == "BaselineMpc" ? "02 / 03  FOSSEN MPC" :
            phase == "KoopmanMpc" ? "03 / 03  KOOPMAN MPC" :
            phase == "SettleBaseline" ? "RESET  ->  FOSSEN MPC" :
            phase == "SettleKoopman" ? "RESET  ->  KOOPMAN MPC" : "EXPERIMENT COMPLETE";
        GUI.DrawTexture(new Rect(16, 16, 420, 280), panel);
        GUI.Label(new Rect(30, 26, 405, 32), title, titleStyle);
        GUI.Label(new Rect(30, 64, 400, 25),
            $"Time  {experiment.ViewerPhaseTime:F1} / {experiment.ViewerPhaseDuration:F0} s", bodyStyle);
        GUI.Label(new Rect(30, 98, 400, 25),
            $"Horizontal error  {Vector2.Distance(experiment.ViewerPosition, experiment.ViewerReference):F3} m", bodyStyle);
        GUI.Label(new Rect(30, 128, 400, 25),
            $"Depth error  {experiment.ViewerDepthError:+0.000;-0.000;0.000} m", bodyStyle);
        GUI.Label(new Rect(30, 159, 400, 25),
            $"Tilt  {experiment.ViewerTilt:F2} deg    Yaw error  {experiment.ViewerYawError:F1} deg", bodyStyle);
        GUI.Label(new Rect(30, 198, 400, 25),
            phase == "Identify" || phase == "SettleBaseline" ?
                "Fossen MPC RMSE  pending" :
                $"Fossen MPC RMSE  {experiment.ViewerBaselineRmse:F3} m", bodyStyle);
        GUI.Label(new Rect(30, 228, 400, 25),
            phase == "KoopmanMpc" || phase == "Done" ?
                $"Koopman MPC RMSE  {experiment.ViewerKoopmanRmse:F3} m" :
                "Koopman MPC RMSE  pending", bodyStyle);
        GUI.Label(new Rect(30, 258, 400, 21), "Horizon 1.2 s  |  6-axis wrench constraints", smallStyle);
        GUI.DrawTexture(new Rect(width - 310, 16, 294, 38), panel);
        GUI.Label(new Rect(width - 298, 19, 280, 30), "FOLLOW / ATTITUDE", bodyStyle);
        GUI.DrawTexture(new Rect(16, height - 70, 570, 54), panel);
        GUI.Label(new Rect(30, height - 65, 550, 24),
            "YELLOW target   ORANGE Fossen MPC   CYAN Koopman MPC", bodyStyle);
        GUI.Label(new Rect(30, height - 40, 550, 20),
            "Large view: live Rigidbody path  |  Upper right: 3D attitude", smallStyle);
    }

    void DrawTrajectory(float width, float height)
    {
        if (Event.current.type != EventType.Repaint) return;
        float pixelsPerMetre = Mathf.Min(width * 0.205f / 0.60f, height * 0.23f / 0.35f);
        Vector2 center = new Vector2(width * 0.485f, height * 0.64f);
        Vector2 Map(Vector2 value) => new Vector2(center.x + value.x * pixelsPerMetre,
            center.y - value.y * pixelsPerMetre);
        float omega = 2f * Mathf.PI / 32f;
        Vector2 previous = Vector2.zero;
        for (int i = 1; i <= 192; i++)
        {
            float t = 32f * i / 192f;
            Vector2 current = new Vector2(0.60f * Mathf.Sin(omega * t),
                0.35f * Mathf.Sin(2f * omega * t));
            DrawLine(Map(previous), Map(current), new Color(1f, 0.91f, 0.24f), 2f);
            previous = current;
        }
        DrawPath(baselinePoints, Map, new Color(1f, 0.38f, 0.20f));
        DrawPath(koopmanPoints, Map, new Color(0.20f, 0.88f, 1f));
        if (!experiment.ViewerEvaluating) return;
        Vector2 target = Map(experiment.ViewerReference);
        Vector2 actual = Map(experiment.ViewerPosition);
        Color old = GUI.color;
        GUI.color = new Color(1f, 0.91f, 0.24f);
        GUI.DrawTexture(new Rect(target.x - 6f, target.y - 6f, 12f, 12f), pixel);
        GUI.color = experiment.ViewerPhase == "BaselineMpc" ?
            new Color(1f, 0.38f, 0.20f) : new Color(0.20f, 0.88f, 1f);
        GUI.DrawTexture(new Rect(actual.x - 8f, actual.y - 8f, 16f, 16f), pixel);
        GUI.color = old;
    }

    void DrawPath(List<Vector2> points, Func<Vector2, Vector2> map, Color color)
    {
        for (int i = 1; i < points.Count; i++) DrawLine(map(points[i - 1]), map(points[i]), color, 3f);
    }

    void DrawLine(Vector2 start, Vector2 end, Color color, float thickness)
    {
        Vector2 delta = end - start;
        if (delta.sqrMagnitude < 0.01f) return;
        Matrix4x4 oldMatrix = GUI.matrix;
        Color oldColor = GUI.color;
        GUI.color = color;
        GUIUtility.RotateAroundPivot(Mathf.Atan2(delta.y, delta.x) * Mathf.Rad2Deg, start);
        GUI.DrawTexture(new Rect(start.x, start.y - thickness / 2f, delta.magnitude, thickness), pixel);
        GUI.matrix = oldMatrix;
        GUI.color = oldColor;
    }

    static Texture2D Solid(Color color)
    {
        var result = new Texture2D(1, 1);
        result.SetPixel(0, 0, color);
        result.Apply();
        return result;
    }

    void OnDestroy()
    {
        if (panel != null) Destroy(panel);
        if (pixel != null) Destroy(pixel);
    }
}
