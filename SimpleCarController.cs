using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class SimpleCarController : MonoBehaviour
{
    [Header("References")]
    public AlphaStateReceiver bciInput;

    [Header("Driving parameters")]
    public float maxSpeed = 20f;
    public float accelerationForce = 8f;
    public float brakeForce = 15f;

    [Header("Keyboard fallback (test without live_classify.py running)")]
    public bool allowKeyboardOverride = true;
    public KeyCode simulateEyesClosedKey = KeyCode.Space;

    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
    }

    void FixedUpdate()
    {
        bool eyesClosed = bciInput != null && bciInput.eyesClosed;
        float confidence = bciInput != null ? bciInput.confidence : 1f;

        // Lets you test the demo instantly without Simulink/live_classify.py
        // running - hold Space to simulate "eyes closed".
        if (allowKeyboardOverride && Input.GetKey(simulateEyesClosedKey))
            eyesClosed = true;

        float forwardSpeed = Vector3.Dot(rb.velocity, transform.forward);

        if (eyesClosed)
        {
            // Brake force scales with confidence - mirrors demo_visual.py's
            // alpha-blending of the status indicator: low confidence ->
            // gentler, more tentative braking, not an abrupt full stop.
            rb.AddForce(-transform.forward * brakeForce * confidence, ForceMode.Acceleration);
        }
        else if (forwardSpeed < maxSpeed)
        {
            rb.AddForce(transform.forward * accelerationForce * confidence, ForceMode.Acceleration);
        }
    }
}
