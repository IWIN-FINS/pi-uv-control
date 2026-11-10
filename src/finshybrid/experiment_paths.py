"""Configured filesystem locations for the archived simulation coordinators."""
import json
import os
import sys
from pathlib import Path


def settings():
    filename = os.environ.get("PI_UV_CONFIG")
    if not filename:
        raise RuntimeError("Set PI_UV_CONFIG to a simulation configuration JSON; see simulation/REPRODUCE.md")
    config = Path(filename).expanduser().resolve()
    values = json.loads(config.read_text(encoding="utf-8-sig"))
    paths = {}
    for key, value in values["paths"].items():
        location = Path(os.path.expandvars(value)).expanduser()
        paths[key] = (config.parent / location).resolve() if not location.is_absolute() else location.resolve()
    return values, paths


def config_path(key):
    _, paths = settings()
    if key == "python":
        return Path(sys.executable)
    if key not in paths:
        raise ValueError(f"Missing configured path: {key}")
    return paths[key]


def relocate_text(text):
    """Relocate quoted compiler-response-file paths using the artifact aliases."""
    values, paths = settings()
    for original, key in sorted(values.get("artifact_path_aliases", {}).items(), key=lambda item: -len(item[0])):
        replacement = paths[key].as_posix()
        text = text.replace(original + "/", replacement + "/")
        text = text.replace(original.replace("/", "\\") + "\\", replacement + "/")
    return text


def load_record(path):
    """Read a relocated artifact without modifying its original bytes or hashes."""
    values, paths = settings()
    aliases = sorted(values.get("artifact_path_aliases", {}).items(), key=lambda item: -len(item[0]))

    def relocate(value):
        if isinstance(value, str):
            normalized = value.replace("\\", "/")
            for original, key in aliases:
                original = original.rstrip("/")
                if normalized == original or normalized.startswith(original + "/"):
                    return str(paths[key] / normalized[len(original):].lstrip("/"))
            return value
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, dict):
            return {relocate(key): relocate(item) for key, item in value.items()}
        return value

    return relocate(json.loads(Path(path).read_text(encoding="utf-8-sig")))
