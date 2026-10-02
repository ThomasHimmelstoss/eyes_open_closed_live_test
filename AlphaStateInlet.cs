// Script to receive eye state (open/closed) from LSL stream and control gas/brake input in Unity.
// Prerequisite: Install official LSL4Unity (labstreaminglayer/LSL4Unity from GitHub).
// Package Manager → "Add package from git URL" → https://github.com/labstreaminglayer/LSL4Unity.git

using System.Collections;
using System.Globalization;
using UnityEngine;
using LSL;

public class AlphaStateReceiver : MonoBehaviour
{
    [Header("LSL Stream")]
    public string streamName = "AlphaState";

    [Header("Current state (read by other scripts, e.g. SimpleCarController)")]
    public bool eyesClosed = false;
    public float confidence = 0f;
    public bool isConnected = false;

    private StreamInlet inlet;
    private readonly string[] sample = new string[1]; // single string channel, see live_classify.py

    void Start()
    {
        StartCoroutine(ResolveStream());
    }

    private IEnumerator ResolveStream()
    {
        Debug.Log($"[AlphaStateReceiver] Resolving LSL stream '{streamName}' ...");
        StreamInfo[] results = new StreamInfo[0];

        while (results.Length == 0)
        {
            results = LSL.LSL.resolve_stream("name", streamName, 1, 1.0);
            yield return null; // retry next frame, non-blocking - keeps Unity responsive while waiting
        }

        inlet = new StreamInlet(results[0]);
        isConnected = true;
        Debug.Log("[AlphaStateReceiver] Connected to stream.");
    }

    void Update()
    {
        if (inlet == null) return;

        // Pull ALL samples buffered since last frame, keep only the latest.
        // Unity typically runs at 60+ fps while live_classify.py only pushes
        // one update per STEP_SECONDS_LIVE (~1s) - most frames will find
        // nothing new, that's expected, not an error.
        double timestamp;
        while ((timestamp = inlet.pull_sample(sample, 0.0)) != 0.0)
        {
            ParseSample(sample[0]);
        }
    }

    private void ParseSample(string raw)
    {
        // Format from live_classify.py: "STATE|confidence|f1,f2,f3,f4"
        string[] parts = raw.Split('|');
        eyesClosed = parts[0] == "EYES CLOSED";
        // InvariantCulture is important: a German/European OS locale uses
        // ',' as decimal separator, which would otherwise break parsing.
        confidence = float.Parse(parts[1], CultureInfo.InvariantCulture);
    }
}