using System;
using System.Collections.Generic;
using UnityEngine;

// Display-only view for the frozen dual-basis controller experiment.
// It reads the controller's Rigidbody telemetry and never changes its dynamics.
[DefaultExecutionOrder(100)]
public sealed class FinsRovHybridBasisViewer : MonoBehaviour
{
    readonly List<Vector2> baselinePoints = new List<Vector2>();
    readonly List<Vector2> koopmanPoints = new List<Vector2>();
    FinsRovHybridBasisSmoke experiment;
    Rigidbody body;
    Camera followCamera;
    Texture2D panel;
    Texture2D pixel;
    GUIStyle titleStyle;
    GUIStyle bodyStyle;
    GUIStyle smallStyle;

    void Start()
    {
        experiment = GetComponent<FinsRovHybridBasisSmoke>();
        body = GetComponent<Rigidbody>();
        followCamera = GameObject.Find("MPC Follow Camera")?.GetComponent<Camera>();
        if (experiment == null || body == null || followCamera == null)
            throw new InvalidOperationException("Hybrid-basis viewer scene is incomplete");
    }

    void FixedUpdate()
    {
        if (experiment == null || !experiment.ViewerEvaluating) return;
        var points = experiment.ViewerPhase == "BaselineMpc" ? baselinePoints : koopmanPoints;
        Vector2 position = experiment.ViewerPosition;
        if (points.Count == 0 || Vector2.Distance(points[points.Count - 1], position) > 0.004f)
            points.Add(position);
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
            panel = Solid(new Color(0.025f, 0.06f, 0.11f, 0.91f));
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
        string title = phase == "BaselineMpc" ? "02 / 03  CALIBRATED FOSSEN MPC" :
            phase == "KoopmanMpc" ? "03 / 03  DUAL-BASIS KOOPMAN MPC" :
            phase == "Done" ? "EXPERIMENT COMPLETE" : "PREPARING FROZEN DUAL-BASIS MODEL";
        GUI.DrawTexture(new Rect(16, 16, 470, 292), panel);
        GUI.Label(new Rect(30, 26, 450, 32), title, titleStyle);
        GUI.Label(new Rect(30, 64, 430, 25),
            $"Time  {experiment.ViewerPhaseTime:F1} / {experiment.ViewerPhaseDuration:F1} s", bodyStyle);
        GUI.Label(new Rect(30, 98, 430, 25),
            $"Horizontal error  {Vector2.Distance(experiment.ViewerPosition, experiment.ViewerReference):F3} m", bodyStyle);
        GUI.Label(new Rect(30, 128, 430, 25),
            $"Depth error  {experiment.ViewerDepthError:+0.000;-0.000;0.000} m", bodyStyle);
        GUI.Label(new Rect(30, 159, 430, 25),
            $"Tilt  {experiment.ViewerTilt:F2} deg    Yaw error  {experiment.ViewerYawError:F1} deg", bodyStyle);
        GUI.Label(new Rect(30, 198, 430, 25),
            $"Calibrated MPC RMSE  {experiment.ViewerBaselineRmse:F3} m", bodyStyle);
        GUI.Label(new Rect(30, 228, 430, 25),
            phase == "KoopmanMpc" || phase == "Done" ?
                $"Dual-basis MPC RMSE  {experiment.ViewerKoopmanRmse:F3} m" :
                "Dual-basis MPC RMSE  pending", bodyStyle);
        GUI.Label(new Rect(30, 266, 430, 21), "SQP 32  |  six-axis MPC  |  eight-thruster allocation", smallStyle);
        GUI.DrawTexture(new Rect(width - 310, 16, 294, 38), panel);
        GUI.Label(new Rect(width - 298, 19, 280, 30), "FOLLOW / ATTITUDE", bodyStyle);
        GUI.DrawTexture(new Rect(16, height - 70, 650, 54), panel);
        GUI.Label(new Rect(30, height - 65, 630, 24),
            "YELLOW target   ORANGE calibrated MPC   CYAN dual-basis MPC", bodyStyle);
        GUI.Label(new Rect(30, height - 40, 630, 20),
            "Large view: live Rigidbody path  |  Upper right: 3D attitude", smallStyle);
    }

    void DrawTrajectory(float width, float height)
    {
        if (Event.current.type != EventType.Repaint) return;
        float duration = Mathf.Max(0.01f, experiment.ViewerPhaseDuration);
        float xExtent = 0.60f * 1.15f;
        float zExtent = 0.35f * 1.15f;
        float pixelsPerMetre = Mathf.Min(width * 0.205f / xExtent, height * 0.23f / zExtent);
        Vector2 center = new Vector2(width * 0.485f, height * 0.64f);
        Vector2 Map(Vector2 value) => new Vector2(center.x + value.x * pixelsPerMetre,
            center.y - value.y * pixelsPerMetre);
        Vector2 previous = experiment.ViewerReferenceAt(0f);
        for (int i = 1; i <= 192; i++)
        {
            Vector2 current = experiment.ViewerReferenceAt(duration * i / 192f);
            DrawLine(Map(previous), Map(current), new Color(1f, 0.91f, 0.24f), 2f);
            previous = current;
        }
        DrawPath(baselinePoints, Map, new Color(1f, 0.38f, 0.20f));
        DrawPath(koopmanPoints, Map, new Color(0.20f, 0.88f, 1f));
        if (!experiment.ViewerEvaluating) return;
        Color old = GUI.color;
        Vector2 target = Map(experiment.ViewerReference);
        Vector2 actual = Map(experiment.ViewerPosition);
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
