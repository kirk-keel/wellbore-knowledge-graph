"""Tests for portfolio compliance: structure, configs, docker, and documentation files."""

from pathlib import Path
import os
import yaml


def test_standard_directory_structure():
    root = Path(__file__).parent.parent
    expected_dirs = ["src", "tests", "configs", "docker", "docs"]
    for dir_name in expected_dirs:
        dir_path = root / dir_name
        assert dir_path.is_dir(), f"Expected directory '{dir_name}' does not exist at {dir_path}"


def test_required_root_and_docker_files():
    root = Path(__file__).parent.parent
    required_files = [
        "LICENSE",
        ".env.example",
        ".gitignore",
        "Dockerfile",
        "docker-compose.yml",
        "README.md",
        "docker/Dockerfile",
        "docker/docker-compose.yml",
        "docker/docker-compose.dev.yml",
        "docker/entrypoint.sh",
        "docker/README.md",
        "configs/app_config.yaml",
        "configs/logging.yaml",
        "configs/neo4j.conf",
        "docs/architecture.md",
        "docs/api_reference.md",
        "docs/deployment.md",
        "docs/domain_glossary.md",
        "docs/quickstart.md",
    ]
    for rel_path in required_files:
        file_path = root / rel_path
        assert file_path.is_file(), f"Required file '{rel_path}' is missing at {file_path}"
        assert file_path.stat().st_size > 0, f"File '{rel_path}' is empty"


def test_app_config_yaml_validity():
    root = Path(__file__).parent.parent
    config_file = root / "configs" / "app_config.yaml"
    assert config_file.exists()
    with open(config_file, "r") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    assert "application" in data
    assert "physics" in data
    assert "retrieval" in data
    assert data["physics"]["hydraulics"]["rheology_model"] == "bingham_plastic"


def test_entrypoint_executable():
    root = Path(__file__).parent.parent
    entrypoint = root / "docker" / "entrypoint.sh"
    assert os.access(entrypoint, os.X_OK), f"entrypoint.sh must be executable"
