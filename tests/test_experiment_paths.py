import json
import sys

import pytest

from finshybrid.experiment_paths import config_path, load_record, relocate_text


def test_paths_resolve_relative_to_config_not_working_directory(tmp_path, monkeypatch):
    folder = tmp_path / "configuration with spaces"
    folder.mkdir()
    config = folder / "settings.json"
    config.write_text(json.dumps({"paths": {"repository": "../checkout with spaces"}}))
    monkeypatch.setenv("PI_UV_CONFIG", str(config))
    monkeypatch.chdir(tmp_path.parent)
    assert config_path("repository") == tmp_path / "checkout with spaces"
    assert config_path("python") == __import__("pathlib").Path(sys.executable)


def test_nested_artifact_paths_use_longest_prefix_and_preserve_bytes(tmp_path, monkeypatch):
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({
        "paths": {"root": "external", "study": "new study"},
        "artifact_path_aliases": {"X:/old": "root", "X:/old/study": "study"},
    }))
    monkeypatch.setenv("PI_UV_CONFIG", str(config))
    artifact = tmp_path / "record.json"
    artifact.write_text(json.dumps({
        "X:/old/study/model.json": ["X:\\old\\study\\input.csv", "X:/older/keep.txt"],
        "value": 0.05,
    }))
    before = artifact.read_bytes()
    result = load_record(artifact)
    assert result[str(tmp_path / "new study/model.json")] == [
        str(tmp_path / "new study/input.csv"), "X:/older/keep.txt",
    ]
    assert result["value"] == 0.05
    assert artifact.read_bytes() == before
    response = relocate_text('-r:"X:/old/study/native.dll" "Assets/Scripts/Plant.cs"')
    assert response == f'-r:"{(tmp_path / "new study/native.dll").as_posix()}" "Assets/Scripts/Plant.cs"'


def test_missing_config_and_missing_key_are_actionable(tmp_path, monkeypatch):
    monkeypatch.delenv("PI_UV_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="PI_UV_CONFIG"):
        config_path("player")
    file = tmp_path / "settings.json"
    file.write_text('{"paths": {}}')
    monkeypatch.setenv("PI_UV_CONFIG", str(file))
    with pytest.raises(ValueError, match="player"):
        config_path("player")
