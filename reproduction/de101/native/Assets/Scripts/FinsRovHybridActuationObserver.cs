using UnityEngine;

// Read applied forces after actuators update; do not change physics or commands.
[DefaultExecutionOrder(10000)]
public sealed class FinsRovHybridActuationObserver : MonoBehaviour
{
    public FinsRovHybridBasisSmoke Owner;
    void FixedUpdate() => Owner?.ObserveAppliedStep();
}
