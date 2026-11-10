import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "provenance/files.json").read_text())
for name, expected in manifest.items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Hash mismatch: {name}")
print(f"Verified {len(manifest)} distributed source, model and data files.")
