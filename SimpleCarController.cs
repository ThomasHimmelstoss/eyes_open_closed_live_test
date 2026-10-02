using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class SimpleCarController : MonoBehaviour
{
    [Header("References")]
    public AlphaStateReceiver bciInput;

    [Header("Driving parameters")]
    public float maxSpeed = 20f;
    public float accelerationForce = 8f;

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

        if (allowKeyboardOverride && Input.GetKey(simulateEyesClosedKey))
            eyesClosed = true;

        float forwardSpeed = Vector3.Dot(rb.velocity, transform.forward);

        // Nur Gas, solange "Eyes Open" erkannt wird - bei "Eyes Closed" wird
        // einfach nichts mehr angetrieben, das Auto rollt durch Linear
        // Damping von selbst aus, statt aktiv gebremst zu werden. Klarer,
        // eindeutiger Zusammenhang zwischen BCI-Zustand und Fahrzeugverhalten
        // als eine gleichzeitige Gas+Brems-Logik.
        if (!eyesClosed && forwardSpeed < maxSpeed)
        {
            rb.AddForce(transform.forward * accelerationForce * confidence, ForceMode.Acceleration);
        }
    }
}