using UnityEngine;

public class DemoHUD : MonoBehaviour
{
    public AlphaStateReceiver bciInput;
    public Rigidbody carRigidbody;

    void OnGUI()
    {
        GUI.Label(new Rect(10, 10, 320, 25),
            bciInput != null && bciInput.isConnected ? "LSL: connected" : "LSL: waiting for stream ...");
        GUI.Label(new Rect(10, 35, 320, 25),
            $"State: {(bciInput != null && bciInput.eyesClosed ? "EYES CLOSED (braking)" : "EYES OPEN (gas)")}");
        GUI.Label(new Rect(10, 60, 320, 25),
            $"Confidence: {(bciInput != null ? bciInput.confidence : 0):F2}");
        if (carRigidbody != null)
            GUI.Label(new Rect(10, 85, 320, 25), $"Speed: {carRigidbody.velocity.magnitude:F1} m/s");
    }
}