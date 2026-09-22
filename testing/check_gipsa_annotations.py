"""
check_gipsa_annotations.py - einmaliger Sanity-Check vor dem vollen
pretrain_gipsa.py-Lauf.
"""

from moabb.datasets import Rodrigues2017

dataset = Rodrigues2017()
data = dataset.get_data(subjects=[1])  # nur EIN Proband, zum schnellen Pruefen

# Verschachtelte Struktur einmal aufschluesseln und das erste Raw-Objekt holen
for subject_id, sessions in data.items():
    for session_id, runs in sessions.items():
        for run_id, raw in runs.items():
            print(f"Subject {subject_id}, Session {session_id}, Run {run_id}")
            print("Kanalnamen:", raw.ch_names)
            print("Samplingrate:", raw.info["sfreq"])
            print("Annotation-Labels (einzigartig):", set(raw.annotations.description))
            print("Anzahl Annotationen gesamt:", len(raw.annotations))