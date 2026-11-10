using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

// Presentation only: cameras, reference/actual paths and a readable experiment HUD.
// Nothing in this component applies a force or changes the experiment clock.
[DefaultExecutionOrder(100)]
public sealed class FinsRovKoopmanViewer : MonoBehaviour
{
    const float PathHeight = -1.48f;
    FinsRovKoopmanSmoke experiment;
    Rigidbody body;
    Camera followCamera;
    LineRenderer baselinePath;
    LineRenderer koopmanPath;
    Transform targetMarker;
    GUIStyle titleStyle;
    GUIStyle bodyStyle;
    GUIStyle smallStyle;
    Texture2D panelTexture;
    Texture2D lineTexture;
    readonly List<Vector2> baselinePoints = new List<Vector2>();
    readonly List<Vector2> koopmanPoints = new List<Vector2>();
    string lastPhase;
    string captureDirectory;
    bool capturedBaseline;
    bool capturedKoopman;

    void Start()
    {
        experiment = GetComponent<FinsRovKoopmanSmoke>();
        body = GetComponent<Rigidbody>();
        if (experiment == null || body == null)
            throw new InvalidOperationException("Koopman viewer needs the experiment and FinsROV Rigidbody");
        followCamera = GameObject.Find("Attitude Follow Camera")?.GetComponent<Camera>();
        CreateLine("Reference figure-eight", new Color(1f, 0.91f, 0.24f), 0.018f, out var referencePath);
        var positions = new Vector3[257];
        for (int i = 0; i < positions.Length; i++)
        {
            float t = 32f * i / (positions.Length - 1);
            float omega = 2f * Mathf.PI / 32f;
            positions[i] = new Vector3(0.60f * Mathf.Sin(omega * t), PathHeight,
                0.35f * Mathf.Sin(2f * omega * t));
        }
        referencePath.positionCount = positions.Length;
        referencePath.SetPositions(positions);
        CreateLine("Fossen baseline trail", new Color(1f, 0.38f, 0.20f), 0.024f, out baselinePath);
        CreateLine("Koopman residual trail", new Color(0.20f, 0.88f, 1f), 0.024f, out koopmanPath);
        var marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        marker.name = "Moving reference target";
        marker.transform.localScale = Vector3.one * 0.085f;
        Destroy(marker.GetComponent<Collider>());
        marker.GetComponent<Renderer>().material = MakeMaterial(new Color(1f, 0.91f, 0.24f));
        targetMarker = marker.transform;
        captureDirectory = Environment.GetEnvironmentVariable("FINSSIM_KOOPMAN_CAPTURE_DIR");
        if (!string.IsNullOrEmpty(captureDirectory)) Directory.CreateDirectory(captureDirectory);
    }

    void CreateLine(string name, Color color, float width, out LineRenderer line)
    {
        var go = new GameObject(name);
        line = go.AddComponent<LineRenderer>();
        line.useWorldSpace = true;
        line.loop = false;
        line.widthMultiplier = width;
        line.numCapVertices = 4;
        line.numCornerVertices = 4;
        line.material = MakeMaterial(color);
        line.startColor = color;
        line.endColor = color;
        line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        line.receiveShadows = false;
    }

    static Material MakeMaterial(Color color)
    {
        var shader = Shader.Find("HDRP/Unlit") ?? Shader.Find("Unlit/Color");
        if (shader == null) throw new InvalidOperationException("No unlit path shader available");
        var material = new Material(shader);
        material.color = color;
        if (material.HasProperty("_UnlitColor")) material.SetColor("_UnlitColor", color);
        return material;
    }

    void FixedUpdate()
    {
        if (experiment == null) return;
        string phase = experiment.ViewerPhase;
        if (phase != lastPhase)
        {
            lastPhase = phase;
            if (phase == "Baseline") baselinePath.positionCount = 0;
            if (phase == "Koopman") koopmanPath.positionCount = 0;
        }
        bool evaluating = experiment.ViewerEvaluating;
        targetMarker.gameObject.SetActive(evaluating);
        if (!evaluating) return;
        Vector2 reference = experiment.ViewerReference;
        targetMarker.position = new Vector3(reference.x, PathHeight + 0.045f, reference.y);
        LineRenderer trail = phase == "Baseline" ? baselinePath : koopmanPath;
        Vector3 trailPoint = new Vector3(body.position.x, PathHeight + 0.018f, body.position.z);
        if (trail.positionCount == 0 ||
            Vector3.Distance(trail.GetPosition(trail.positionCount - 1), trailPoint) > 0.006f)
        {
            trail.positionCount++;
            trail.SetPosition(trail.positionCount - 1, trailPoint);
            (phase == "Baseline" ? baselinePoints : koopmanPoints)
                .Add(new Vector2(body.position.x, body.position.z));
        }
        if (!string.IsNullOrEmpty(captureDirectory) && experiment.ViewerPhaseTime >= 8f)
        {
            if (phase == "Baseline" && !capturedBaseline)
            {
                capturedBaseline = true;
                StartCoroutine(Capture("baseline.png"));
            }
            if (phase == "Koopman" && !capturedKoopman)
            {
                capturedKoopman = true;
                StartCoroutine(Capture("koopman.png"));
            }
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
        if (titleStyle == null)
        {
            panelTexture = new Texture2D(1, 1);
            panelTexture.SetPixel(0, 0, new Color(0.025f, 0.06f, 0.11f, 0.88f));
            panelTexture.Apply();
            lineTexture = new Texture2D(1, 1);
            lineTexture.SetPixel(0, 0, Color.white);
            lineTexture.Apply();
            titleStyle = new GUIStyle(GUI.skin.label) { fontSize = 22, fontStyle = FontStyle.Bold };
            titleStyle.normal.textColor = Color.white;
            bodyStyle = new GUIStyle(GUI.skin.label) { fontSize = 17 };
            bodyStyle.normal.textColor = Color.white;
            smallStyle = new GUIStyle(GUI.skin.label) { fontSize = 15 };
            smallStyle.normal.textColor = new Color(0.78f, 0.85f, 0.92f);
        }
        float scale = Mathf.Max(0.8f, Screen.height / 800f);
        GUI.matrix = Matrix4x4.Scale(new Vector3(scale, scale, 1f));
        DrawTrajectory(Screen.width / scale, Screen.height / scale);
        GUI.DrawTexture(new Rect(16, 16, 405, 250), panelTexture);
        string phase = experiment.ViewerPhase;
        string heading = phase == "Identify" ? "01 / 03  IDENTIFY MODEL" :
            phase == "Baseline" ? "02 / 03  FOSSEN BASELINE" :
            phase == "Koopman" ? "03 / 03  KOOPMAN RESIDUAL" :
            phase == "SettleBaseline" ? "RESET  ->  FOSSEN BASELINE" :
            phase == "SettleKoopman" ? "RESET  ->  KOOPMAN RESIDUAL" : "EXPERIMENT COMPLETE";
        GUI.Label(new Rect(30, 26, 380, 35), heading, titleStyle);
        GUI.Label(new Rect(30, 66, 380, 28),
            $"Time  {experiment.ViewerPhaseTime:F1} / {experiment.ViewerPhaseDuration:F0} s", bodyStyle);
        GUI.Label(new Rect(30, 100, 380, 26),
            $"Position error   {experiment.ViewerError:F3} m", bodyStyle);
        GUI.Label(new Rect(30, 132, 380, 26),
            phase == "Identify" || phase == "SettleBaseline" ?
                "Fossen RMSE   pending" : $"Fossen RMSE   {experiment.ViewerBaselineRmse:F3} m", bodyStyle);
        GUI.Label(new Rect(30, 161, 380, 26),
            phase == "Koopman" || phase == "Done" ?
                $"Koopman RMSE  {experiment.ViewerKoopmanRmse:F3} m" :
                "Koopman RMSE  pending", bodyStyle);
        GUI.Label(new Rect(30, 194, 380, 26),
            $"Roll {experiment.ViewerRoll:+0.0;-0.0;0.0}°   Pitch {experiment.ViewerPitch:+0.0;-0.0;0.0}°", smallStyle);
        GUI.Label(new Rect(30, 219, 380, 26),
            $"Yaw {experiment.ViewerYaw:+0.0;-0.0;0.0}°   Tilt {experiment.ViewerTilt:F1}°", smallStyle);
        float bottom = Screen.height / scale - 70;
        GUI.DrawTexture(new Rect(16, bottom, 570, 54), panelTexture);
        GUI.Label(new Rect(30, bottom + 5, 550, 24),
            "YELLOW target   ORANGE Fossen   CYAN Koopman", bodyStyle);
        GUI.Label(new Rect(30, bottom + 29, 550, 20),
            "Large view: complete path     Upper right: following attitude camera", smallStyle);
        float right = Screen.width / scale - 310;
        GUI.DrawTexture(new Rect(right, 16, 294, 38), panelTexture);
        GUI.Label(new Rect(right + 12, 19, 280, 30), "FOLLOW / ATTITUDE", bodyStyle);
    }

    void OnDestroy()
    {
        if (panelTexture != null) Destroy(panelTexture);
        if (lineTexture != null) Destroy(lineTexture);
    }

    void DrawTrajectory(float width, float height)
    {
        if (Event.current.type != EventType.Repaint) return;
        float pixelsPerMetre = Mathf.Min(width * 0.205f / 0.60f, height * 0.23f / 0.35f);
        Vector2 center = new Vector2(width * 0.485f, height * 0.64f);
        Vector2 Map(Vector2 point) => new Vector2(center.x + point.x * pixelsPerMetre,
            center.y - point.y * pixelsPerMetre);
        float omega = 2f * Mathf.PI / 32f;
        Vector2 previous = new Vector2(0f, 0f);
        for (int i = 1; i <= 192; i++)
        {
            float t = 32f * i / 192f;
            Vector2 next = new Vector2(0.60f * Mathf.Sin(omega * t),
                0.35f * Mathf.Sin(2f * omega * t));
            DrawLine(Map(previous), Map(next), new Color(1f, 0.91f, 0.24f), 2f);
            previous = next;
        }
        DrawPath(baselinePoints, Map, new Color(1f, 0.38f, 0.20f));
        DrawPath(koopmanPoints, Map, new Color(0.20f, 0.88f, 1f));
        if (experiment.ViewerEvaluating)
        {
            Vector2 target = Map(experiment.ViewerReference);
            GUI.color = new Color(1f, 0.91f, 0.24f);
            GUI.DrawTexture(new Rect(target.x - 6f, target.y - 6f, 12f, 12f), lineTexture);
            Vector2 actual = Map(new Vector2(body.position.x, body.position.z));
            GUI.color = experiment.ViewerPhase == "Baseline" ?
                new Color(1f, 0.38f, 0.20f) : new Color(0.20f, 0.88f, 1f);
            GUI.DrawTexture(new Rect(actual.x - 8f, actual.y - 8f, 16f, 16f), lineTexture);
            float yaw = experiment.ViewerYaw * Mathf.Deg2Rad;
            DrawLine(actual, actual + new Vector2(Mathf.Sin(yaw), -Mathf.Cos(yaw)) * 30f,
                Color.white, 3f);
            GUI.color = Color.white;
        }
    }

    void DrawPath(List<Vector2> points, Func<Vector2, Vector2> map, Color color)
    {
        for (int i = 1; i < points.Count; i++)
            DrawLine(map(points[i - 1]), map(points[i]), color, 3f);
    }

    void DrawLine(Vector2 start, Vector2 end, Color color, float thickness)
    {
        Vector2 delta = end - start;
        if (delta.sqrMagnitude < 0.01f) return;
        Matrix4x4 oldMatrix = GUI.matrix;
        Color oldColor = GUI.color;
        GUI.color = color;
        GUIUtility.RotateAroundPivot(Mathf.Atan2(delta.y, delta.x) * Mathf.Rad2Deg, start);
        GUI.DrawTexture(new Rect(start.x, start.y - thickness / 2f,
            delta.magnitude, thickness), lineTexture);
        GUI.matrix = oldMatrix;
        GUI.color = oldColor;
    }
}
