using UnityEngine;

// Read applied forces after actuators update; do not change physics or commands.
[DefaultExecutionOrder(10000)]
public sealed class FinsRovEdmdActuationObserver : MonoBehaviour
{
    public FinsRovEdmdMpcSmoke Owner;
    void FixedUpdate() => Owner?.ObserveAppliedStep();
}
