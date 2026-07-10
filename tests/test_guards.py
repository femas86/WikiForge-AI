import os
import pytest
import yaml

from pkms.guards import (
    guard_write,
    list_projects,
    load_config,
    validate_project,
    WRITE_BOUNDARIES,
)


@pytest.fixture()
def vault(tmp_path):
    """Return a tmp vault root with minimal per-project structure."""
    (tmp_path / "vault" / "default" / "raw").mkdir(parents=True)
    (tmp_path / "vault" / "default" / "wiki" / "articles").mkdir(parents=True)
    (tmp_path / "vault" / "default" / "outputs").mkdir(parents=True)
    return tmp_path / "vault"


# ── guard_write happy paths ──────────────────────────────────────────────────

def test_ingestor_allowed_raw(vault):
    target = str(vault / "default" / "raw" / "doc.pdf")
    guard_write("ingestor", target, str(vault))  # must not raise


def test_ingestor_allowed_other_project(vault):
    target = str(vault / "robotics" / "raw" / "doc.pdf")
    guard_write("ingestor", target, str(vault))


def test_compiler_allowed_wiki(vault):
    target = str(vault / "default" / "wiki" / "articles" / "foo.md")
    guard_write("compiler", target, str(vault))


def test_compiler_allowed_search_index(vault):
    target = str(vault / ".search-index")
    guard_write("compiler", target, str(vault))


def test_querier_allowed_outputs(vault):
    target = str(vault / "default" / "outputs" / "answer.md")
    guard_write("querier", target, str(vault))


def test_querier_allowed_search_index(vault):
    target = str(vault / ".search-index")
    guard_write("querier", target, str(vault))


def test_linter_allowed_outputs(vault):
    target = str(vault / "default" / "outputs" / "lint_report.md")
    guard_write("linter", target, str(vault))


def test_coordinator_allowed_search_index(vault):
    target = str(vault / ".search-index")
    guard_write("coordinator", target, str(vault))


# ── guard_write error paths ──────────────────────────────────────────────────

def test_ingestor_cannot_write_wiki(vault):
    target = str(vault / "default" / "wiki" / "articles" / "secret.md")
    with pytest.raises(PermissionError, match=r"\[ingestor\]"):
        guard_write("ingestor", target, str(vault))


def test_linter_cannot_write_wiki(vault):
    target = str(vault / "default" / "wiki" / "articles" / "foo.md")
    with pytest.raises(PermissionError, match=r"\[linter\]"):
        guard_write("linter", target, str(vault))


def test_querier_cannot_write_raw(vault):
    target = str(vault / "default" / "raw" / "injected.txt")
    with pytest.raises(PermissionError, match=r"\[querier\]"):
        guard_write("querier", target, str(vault))


def test_unknown_agent_is_denied(vault):
    target = str(vault / "default" / "raw" / "x.txt")
    with pytest.raises(PermissionError):
        guard_write("unknown_agent", target, str(vault))


def test_path_traversal_blocked(vault):
    # Attempt to escape vault root
    target = str(vault / ".." / "evil.txt")
    with pytest.raises(PermissionError):
        guard_write("ingestor", target, str(vault))


def test_invalid_project_segment_denied(vault):
    # First segment must be a valid project name — dotfiles are not
    target = str(vault / ".hidden" / "raw" / "x.txt")
    with pytest.raises(PermissionError):
        guard_write("ingestor", target, str(vault))


def test_vault_root_file_denied_for_ingestor(vault):
    # Project-relative boundary must not match files at the vault root
    target = str(vault / "raw")
    with pytest.raises(PermissionError):
        guard_write("ingestor", target, str(vault))


# ── validate_project ─────────────────────────────────────────────────────────

def test_validate_project_accepts_safe_names():
    for name in ("default", "robotics", "llm-research", "q3.2026", "a"):
        assert validate_project(name) == name


def test_validate_project_rejects_unsafe_names():
    for name in ("", "UPPER", "has space", "../etc", ".hidden", "x" * 65, "a/b"):
        with pytest.raises(ValueError):
            validate_project(name)


# ── list_projects ────────────────────────────────────────────────────────────

def test_list_projects_returns_valid_dirs(vault):
    (vault / "robotics" / "raw").mkdir(parents=True)
    (vault / ".git").mkdir()           # invalid name — excluded
    (vault / ".search-index").touch()  # file — excluded
    assert list_projects(vault) == ["default", "robotics"]


def test_list_projects_missing_vault(tmp_path):
    assert list_projects(tmp_path / "nope") == []


# ── load_config ──────────────────────────────────────────────────────────────

def test_load_config_returns_dict(tmp_path):
    cfg = {"vault": {"root": "./vault"}, "lock": {"ttl_seconds": 300}}
    cfg_file = tmp_path / "pkms.config.yaml"
    cfg_file.write_text(yaml.dump(cfg))
    result = load_config(str(cfg_file))
    assert result["vault"]["root"] == "./vault"
    assert result["lock"]["ttl_seconds"] == 300


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/pkms.config.yaml")


def test_load_config_env_overrides(tmp_path, monkeypatch):
    cfg = {"qdrant": {"host": "localhost", "port": 6333},
           "ollama": {"host": "localhost", "port": 11434}}
    cfg_file = tmp_path / "pkms.config.yaml"
    cfg_file.write_text(yaml.dump(cfg))

    monkeypatch.setenv("QDRANT_HOST", "qdrant.internal")
    monkeypatch.setenv("QDRANT_PORT", "7333")
    monkeypatch.setenv("OLLAMA_PORT", "21434")

    result = load_config(str(cfg_file))
    assert result["qdrant"]["host"] == "qdrant.internal"
    assert result["qdrant"]["port"] == 7333
    assert result["ollama"]["host"] == "localhost"  # not overridden
    assert result["ollama"]["port"] == 21434


def test_load_config_no_env_keeps_yaml_values(tmp_path, monkeypatch):
    for var in ("QDRANT_HOST", "QDRANT_PORT", "OLLAMA_HOST", "OLLAMA_PORT"):
        monkeypatch.delenv(var, raising=False)
    cfg = {"qdrant": {"host": "localhost", "port": 6333}}
    cfg_file = tmp_path / "pkms.config.yaml"
    cfg_file.write_text(yaml.dump(cfg))
    result = load_config(str(cfg_file))
    assert result["qdrant"]["port"] == 6333


def test_validate_project_rejects_reserved_names():
    for name in ("raw", "wiki", "outputs"):
        with pytest.raises(ValueError, match="Reserved"):
            validate_project(name)


def test_list_projects_excludes_legacy_dirs(vault):
    # Pre-project layout dirs must not appear as projects
    for legacy in ("raw", "wiki", "outputs"):
        (vault / legacy).mkdir(exist_ok=True)
    assert list_projects(vault) == ["default"]


def test_normalize_project_folds_case_and_spaces():
    from pkms.guards import normalize_project
    assert normalize_project("OG-MDAI") == "og-mdai"
    assert normalize_project("  My Project  ") == "my-project"
    assert normalize_project("default") == "default"
    assert normalize_project("") == ""
