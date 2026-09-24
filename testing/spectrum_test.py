import numpy as np
from scipy.signal import welch, ellip, freqz

data = np.load("recordings/subject_01/2026-09-23_132052/calibration_raw.npz", allow_pickle=True)
closed = data["closed_raw_windows"]

window = closed[0]
print(window.shape)  # sollte (1000, 4) sein - 1000 Samples, 4 Kanaele (P3,P4,O1,O2)

for i, name in enumerate(["P3", "P4", "O1", "O2"]):
    freqs, psd = welch(window[:, i], fs=250, nperseg=500)
    print(name, freqs[np.argmax(psd)], "Hz")



sos = ellip(N=4, rp=0.1, rs=60, Wn=1.0, btype="highpass", fs=250, output="sos")
from scipy.signal import sosfreqz
w, h = sosfreqz(sos, worN=2000, fs=250)
magnitude_db = 20 * np.log10(np.abs(h))
peak_freq = w[np.argmax(magnitude_db)]
print(f"Filter-eigener Verstaerkungs-Peak bei: {peak_freq:.2f} Hz, Wert: {magnitude_db.max():.2f} dB")