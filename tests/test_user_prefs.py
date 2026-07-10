from pathlib import Path

import pkms.user_prefs as up


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "_users_dir", lambda: tmp_path / "users")
    up.save_user_style("alice", "Answer in Italian. TL;DR first.")
    assert up.load_user_style("alice") == "Answer in Italian. TL;DR first."


def test_load_absent_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "_users_dir", lambda: tmp_path / "users")
    assert up.load_user_style("nobody") == ""


def test_user_id_sanitized_to_safe_filename(tmp_path, monkeypatch):
    users = tmp_path / "users"
    monkeypatch.setattr(up, "_users_dir", lambda: users)
    # path-traversal-y id must not escape the users dir
    up.save_user_style("../../etc/passwd", "x")
    written = list(users.glob("*.md"))
    assert len(written) == 1
    assert written[0].parent == users
    assert "/" not in written[0].name


def test_empty_user_id_falls_back_to_default(tmp_path, monkeypatch):
    users = tmp_path / "users"
    monkeypatch.setattr(up, "_users_dir", lambda: users)
    up.save_user_style("", "x")
    assert (users / "default.md").exists()


def test_list_users_includes_default_and_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "_users_dir", lambda: tmp_path / "users")
    assert up.list_users() == ["default"]          # default always present
    up.save_user_style("bob", "y")
    up.save_user_style("alice", "x")
    assert up.list_users() == ["alice", "bob", "default"]  # sorted, deduped
