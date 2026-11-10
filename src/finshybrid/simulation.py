"""Prepare the portable DE101 plan or check a relocated historical environment."""
import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from .experiment_paths import config_path, settings


def check_historical():
    """Report all missing files/modules before any collection or execution."""
    required = [
        ("collection_pool", "collection/queue.json"),
        ("collection_pool", "build_audit.json"),
        ("collection_pool", "data/audit.json"),
        ("collection_pool", "data/n3060/validation.npz"),
        ("collection_pool", "player/FinsROVHybridBasis.exe"),
        ("de101", "build_audit.json"),
        ("de101", "player/FinsROVHybridBasis.exe"),
        ("de101", "formal/still/train281/F/seed1101221/invariant-seed1101221/physical_profile_fixed.json"),
        ("de100", "n3060_u1200/execution/preflight/audit_template.json"),
        ("de101", "execution/truncated_log_repair/repair_protocol.json"),
        ("repository", "data/validation.npz"),
    ]
    missing = [str(config_path(key) / name) for key, name in required if not (config_path(key) / name).is_file()]
    roots = [config_path(key) for key in ("training_source", "legacy_source", "comparison_source", "finssim_tools")]
    # Direct imports in all archived coordinators, including the multi-step branch.
    modules = set()
    for key in ("training_source", "legacy_source", "comparison_source"):
        for script in config_path(key).rglob("*.py"):
            for node in ast.walk(ast.parse(script.read_text(encoding="utf-8-sig"))):
                if isinstance(node, ast.Import):
                    modules.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules.add(node.module.split(".")[0])
    local = {file.stem for directory in roots for file in directory.rglob("*.py")} 
    unavailable = []
    for name in sorted(modules - local):
        if name not in sys.stdlib_module_names and importlib.util.find_spec(name) is None:
            unavailable.append(name)
    if os.name != "nt":
        unavailable.append("Windows runtime: msvcrt, Unity Player and CPU-affinity policy")
    return {"ready": not missing and not unavailable, "missing_files": missing, "missing_modules": unavailable}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("action", choices=("check", "prepare", "run"))
    parser.add_argument("--stage", choices=("de101", "historical"), default="de101")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.environ["PI_UV_CONFIG"] = str(args.config.expanduser().resolve())
    settings()
    if args.stage == "historical":
        report = check_historical()
        print(json.dumps(report, indent=2))
        if not report["ready"]:
            raise SystemExit(2)
        if args.action != "check":
            parser.error("Historical execution requires the staged commands in simulation/REPRODUCE.md")
        return
    repository = config_path("repository")
    bundle = repository / "reproduction/de101"
    if args.action == "check":
        subprocess.run([sys.executable, str(bundle / "verify.py"), "--full-objectives"], check=True)
        return
    if args.output is None:
        parser.error("--output is required for prepare and run")
    command = [sys.executable, str(bundle / "run.py"), "--output-root", str(args.output.resolve())]
    if args.resume:
        command.append("--resume")
    if args.action == "run":
        command.extend(["--run", "--player", str(config_path("player"))])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
