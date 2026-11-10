using System;
using System.IO;
using UnityEngine;

public sealed class WaterSceneCapture : MonoBehaviour
{
    private int frames;

    private void Update()
    {
        string output = Environment.GetEnvironmentVariable("FINSSIM_WATER_SCREENSHOT");
        if (string.IsNullOrEmpty(output) || ++frames != 180) return;
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output)));
        ScreenCapture.CaptureScreenshot(output);
        Debug.Log("FINSSIM_HDRP_WATER_SCREENSHOT " + output);
    }
}
