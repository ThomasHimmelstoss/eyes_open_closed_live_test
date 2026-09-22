"""
check_subject_count.py - einmaliger Check: welche GIPSA-Subject-ID fehlt
gegenueber den erwarteten 20?
"""

from pretrain_gipsa import load_raws

raws = load_raws()
found_ids = sorted(set(subject_id for _, subject_id in raws))

print(f"Gefundene Subject-IDs ({len(found_ids)}):", found_ids)
print("Erwartet: 1 bis 20")

missing = set(range(1, 21)) - set(found_ids)
print("Fehlend:", missing if missing else "keine")