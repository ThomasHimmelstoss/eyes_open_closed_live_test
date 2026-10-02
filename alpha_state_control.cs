using UnityEngine;
using Assets.LSL4Unity.Scripts.AbstractInlets; // Namespace je nach Paketversion ggf. anders

public class AlphaStateInlet : InletStringSamples  // oder passende String-Basisklasse
{
    public float gasBrakeInput = 0f; // an eure Fahrzeug-/Fahrsimulator-Logik anbinden

    protected override void Process(string[] newSample, double timeStamp)
    {
        string raw = newSample[0];           // z.B. "EYES OPEN|0.87|0.12,-0.34,0.56,0.11"
        string state = raw.Split('|')[0];

        if (state == "EYES OPEN")
            gasBrakeInput = 1f;   // Gas
        else if (state == "EYES CLOSED")
            gasBrakeInput = -1f;  // Bremse
    }

    protected override void OnStreamAvailable()
    {
        Debug.Log("AlphaState Stream gefunden, Verbindung aktiv.");
    }
}