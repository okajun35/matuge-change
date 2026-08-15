"""Dependency declarations stay reproducible across Docker builds."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _dependencies(filename: str) -> list[str]:
    lines = (ROOT / filename).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith(("#", "-"))]


@pytest.mark.parametrize("filename", ["requirements.txt", "requirements-dev.txt"])
def test_direct_dependencies_are_exactly_pinned(filename: str):
    for dependency in _dependencies(filename):
        name, separator, version = dependency.partition("==")
        assert separator and name and version, f"{filename}: dependency is not exactly pinned: {dependency}"


def _pinned_version(filename: str, package: str) -> str:
    for dependency in _dependencies(filename):
        name, _, version = dependency.partition("==")
        if name == package:
            return version
    raise AssertionError(f"{package} is not pinned in {filename}")


def _yaml(filename: str) -> dict:
    return yaml.safe_load((ROOT / filename).read_text(encoding="utf-8"))


def test_ruff_version_is_the_same_everywhere():
    """A version split between local, pre-commit and CI makes CI fail on rules that
    pass locally (0.9.6 vs 0.16.3 disagree on UP038, for instance)."""
    expected = _pinned_version("requirements-dev.txt", "ruff")

    hooks = _yaml(".pre-commit-config.yaml")["repos"]
    ruff_repo = next(repo for repo in hooks if "ruff-pre-commit" in repo["repo"])
    assert ruff_repo["rev"] == f"v{expected}", ".pre-commit-config.yaml is on a different ruff"

    steps = _yaml(".github/workflows/ci.yml")["jobs"]["lint"]["steps"]
    versions = [step["with"]["version"] for step in steps if "ruff-action" in str(step.get("uses", ""))]
    assert versions, "the lint job no longer runs ruff-action"
    for version in versions:
        assert str(version) == expected, "ci.yml is on a different ruff"
