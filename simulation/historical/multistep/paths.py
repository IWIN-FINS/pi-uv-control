from finshybrid.experiment_paths import config_path, load_record
import hashlib
import json
import os
from pathlib import Path

ROOT = config_path("multistep_run")
BASE = config_path("de100")
EMPTY = config_path("training_run")
LOADED = config_path("loaded_run")
PILOT = config_path("pilot_run")
WORK = config_path("repository")
PAIRS = config_path("paired_conditions")

def read(path):
    return load_record(path)

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
