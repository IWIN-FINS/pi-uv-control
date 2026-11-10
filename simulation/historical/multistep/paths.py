import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "de100-data-scaling-20261027"
EMPTY = ROOT.parent / "de101-10datasets-400-20261107"
LOADED = ROOT.parent / "de101-ballast300-10datasets-400-20261108"
PILOT = ROOT.parent / "de101-ballast300-pilot-20261108"
WORK = Path(r"C:/Users/admin/.codex/worktrees/multistep-bias-review/Underwater-Adaptive-Dynamics")
PAIRS = ROOT.parent.parent / "diagnostics/de101-multistep-bias-paired-20261108/paired_conditions.csv"

def read(path):
    return Path(path).read_json() if hasattr(path, "read_json") else json.loads(Path(path).read_text(encoding="utf-8-sig"))

def save(path, value):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    os.replace(temp, path)

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def check(cond, message):
    if not cond:
        raise RuntimeError(str(message))

def verify(frozen):
    for path, digest in frozen.items():
        check(sha(path) == digest, f"Hash mismatch: {path}")
