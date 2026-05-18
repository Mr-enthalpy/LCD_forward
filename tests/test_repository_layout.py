"""Smoke test: verify that the repository layout is valid.

This test checks that key documentation files and directories exist.
It does not run forward models or reconstruction solvers.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_readme_exists():
    assert (ROOT / "README.md").is_file()


def test_agents_exists():
    assert (ROOT / "AGENTS.md").is_file()


def test_docs_data_contracts_exists():
    assert (ROOT / "docs" / "data_contracts.md").is_file()


def test_docs_bishe_plan_exists():
    assert (ROOT / "docs" / "bishe_plan.md").is_file()


def test_docs_thesis_background_exists():
    assert (ROOT / "docs" / "thesis_background.md").is_file()


def test_docs_project_boundary_exists():
    assert (ROOT / "docs" / "project_boundary.md").is_file()


def test_config_forward_validation_exists():
    assert (ROOT / "configs" / "forward_validation.yaml").is_file()


def test_config_linear_recon_exists():
    assert (ROOT / "configs" / "linear_recon.yaml").is_file()


def test_data_readme_exists():
    assert (ROOT / "data" / "README.md").is_file()


def test_outputs_readme_exists():
    assert (ROOT / "outputs" / "README.md").is_file()


def test_pyproject_exists():
    assert (ROOT / "pyproject.toml").is_file()
