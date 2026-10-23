"""Check documentation paths and runnable examples without external services."""

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from urllib.parse import unquote, urlsplit

import pytest

from prompt_learning.cli import _validate_runtime_config
from prompt_learning.config import load_config


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ("README.md", "README.ko.md", "docs/architecture.md", "artifacts/experiment-summary.md")


@pytest.mark.parametrize("filename", DOCUMENTS)
def test_documentation_local_links_resolve(filename):
    document = ROOT / filename
    for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", document.read_text(encoding="utf-8")):
        url = urlsplit(target.strip("<>"))
        if not url.scheme and url.path:
            assert (document.parent / unquote(url.path)).exists(), f"{filename}: {target}"


@pytest.mark.parametrize("filename", ("README.md", "README.ko.md"))
def test_documented_offline_commands_and_experiment_preflight(filename, tmp_path):
    """Broken flags, config paths, or smoke preparation must fail here, offline."""
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    environment = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT / "src"),
        "OPENAI_API_KEY": "",
        "PHOENIX_ENABLED": "false",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "EMOTION_DATASET_REVISION": "a" * 40,
        "FINANCIAL_DATASET_REVISION": "b" * 40,
    }
    document = (ROOT / filename).read_text(encoding="utf-8")
    blocks = re.findall(r"```bash offline\n(.*?)```", document, re.DOTALL)
    assert blocks, f"{filename}: no offline examples to execute"
    for block in blocks:
        result = subprocess.run(
            ["/bin/bash", "-eu", "-c", block], cwd=tmp_path, env=environment,
            text=True, capture_output=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    for task in ("emotion", "financial"):
        config = load_config(tmp_path / "runs" / "configs" / f"{task}.yaml")
        default = load_config(ROOT / "configs" / f"{task}.yaml")
        _validate_runtime_config(config)
        assert config.dataset.revision == ("a" if task == "emotion" else "b") * 40
        assert config.max_iterations == 1
        assert not config.phoenix.enabled
        assert config.split == default.split
        assert config.evaluation == default.evaluation
        assert config.models.inference == config.models.evaluator == config.models.meta == "gpt-4o-mini"
        if task == "emotion":
            assert config.dataset.samples_per_label == 5
        else:
            assert config.dataset.sample_count == 5

    commands = re.findall(r"^python -m prompt_learning\.cli run .+$", document, re.MULTILINE)
    assert commands, f"{filename}: no experiment commands to preflight"
    for command in commands:
        arguments = shlex.split(command)[1:]
        if "--help" in arguments or "-h" in arguments:
            continue  # Help already executes in the marked offline block.
        result = subprocess.run(
            [sys.executable, *arguments],
            cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=30,
        )
        assert result.returncode == 2, result.stdout + result.stderr
        assert "Missing API key" in result.stderr, result.stderr


def test_requirements_install_the_canonical_project_dependencies():
    """A compatibility install must consume pyproject, preventing dependency drift."""
    requirements = [
        line.strip() for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert requirements == ["-e ."]
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["dependencies"]
