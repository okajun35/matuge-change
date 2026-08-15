"""Dependency declarations stay reproducible across Docker builds."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _dependencies(filename: str) -> list[str]:
    lines = (ROOT / filename).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith(("#", "-"))]


@pytest.mark.parametrize("filename", ["requirements.txt", "requirements-dev.txt"])
def test_direct_dependencies_are_exactly_pinned(filename: str):
    for dependency in _dependencies(filename):
        name, separator, version = dependency.partition("==")
        assert separator and name and version, f"{filename}: dependency is not exactly pinned: {dependency}"
