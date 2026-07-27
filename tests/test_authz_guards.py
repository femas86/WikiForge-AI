"""B4 guards: per-project role enforcement in the coordinator handlers."""
import pytest

from pkms.auth import AccessDenied
from pkms.coordinator import check_access, handle_ingest, handle_query
from pkms.db import add_member, init_db

CFG = {"auth": {"user_header": "X-Auth-User", "default_user": "local"}}


def _vault(tmp_path):
    """A vault_root with an initialised db; returns (vault_root, db_path)."""
    (tmp_path / "vault").mkdir()
    db = str(tmp_path / "vault" / ".search-index")
    init_db(db)
    return str(tmp_path), db


# ── check_access primitive ─────────────────────────────────────────────────────

def test_internal_call_skips_authz(tmp_path):
    vr, _ = _vault(tmp_path)
    check_access(vr, "demo", None, "owner")   # auth_user None → internal, never raises


def test_unclaimed_project_is_open(tmp_path):
    vr, _ = _vault(tmp_path)
    check_access(vr, "demo", "bob", "editor")  # no members → open even to a random user


def test_claimed_project_enforces_role(tmp_path):
    vr, db = _vault(tmp_path)
    add_member(db, "demo", "alice", "owner", "t0")
    add_member(db, "demo", "carol", "viewer", "t1")
    # owner passes any required role
    check_access(vr, "demo", "alice", "owner")
    check_access(vr, "demo", "alice", "editor")
    # viewer passes viewer, fails editor
    check_access(vr, "demo", "carol", "viewer")
    with pytest.raises(AccessDenied):
        check_access(vr, "demo", "carol", "editor")
    # non-member denied outright
    with pytest.raises(AccessDenied):
        check_access(vr, "demo", "bob", "viewer")


# ── handler enforcement (denied path raises before doing work) ──────────────────

def test_handle_ingest_denies_non_member_on_claimed_project(tmp_path):
    vr, db = _vault(tmp_path)
    add_member(db, "demo", "alice", "owner", "t0")
    with pytest.raises(AccessDenied):
        handle_ingest("x.pdf", vr, CFG, project="demo", auth_user="bob")


def test_handle_query_denies_non_member_on_claimed_project(tmp_path):
    vr, db = _vault(tmp_path)
    add_member(db, "demo", "alice", "owner", "t0")
    with pytest.raises(AccessDenied):
        handle_query("q?", "bob", vr, CFG, project="demo", auth_user="bob")


def test_handle_ingest_internal_call_bypasses(tmp_path):
    # auth_user omitted (internal/CLI) → guard skipped even on a claimed project.
    vr, db = _vault(tmp_path)
    add_member(db, "demo", "alice", "owner", "t0")
    # No AccessDenied; it proceeds past the guard (then fails later on the missing
    # file, which is fine — we only assert the guard didn't block).
    with pytest.raises(Exception) as ei:
        handle_ingest("/nonexistent.pdf", vr, CFG, project="demo")
    assert not isinstance(ei.value, AccessDenied)


# ── member management (B4 (4)) ──────────────────────────────────────────────────

from pkms.coordinator import handle_member


def test_member_list_empty_on_unclaimed(tmp_path):
    vr, _ = _vault(tmp_path)
    res = handle_member("list", "demo", vr, CFG, auth_user="anyone")
    assert res["members"] == []


def test_cli_admin_add_and_list(tmp_path):
    # CLI path: auth_user=None → trusted-local admin, may assign any role.
    vr, _ = _vault(tmp_path)
    handle_member("add", "demo", vr, CFG, user_id="alice", role="owner", auth_user=None)
    handle_member("add", "demo", vr, CFG, user_id="bob", role="editor", auth_user=None)
    members = {m["user_id"]: m["role"] for m in handle_member("list", "demo", vr, CFG, auth_user=None)["members"]}
    assert members == {"alice": "owner", "bob": "editor"}


def test_add_invalid_role_rejected(tmp_path):
    vr, _ = _vault(tmp_path)
    with pytest.raises(ValueError):
        handle_member("add", "demo", vr, CFG, user_id="alice", role="admin", auth_user=None)


def test_web_self_claim_as_owner_allowed(tmp_path):
    # Unclaimed project: a web user may claim it by adding THEMSELVES as owner.
    vr, _ = _vault(tmp_path)
    res = handle_member("add", "demo", vr, CFG, user_id="alice", role="owner", auth_user="alice")
    assert res["status"] == "ADDED"


def test_web_claim_must_be_self_as_owner(tmp_path):
    vr, _ = _vault(tmp_path)
    # adding someone else on an unclaimed project (web) is refused
    with pytest.raises(AccessDenied):
        handle_member("add", "demo", vr, CFG, user_id="bob", role="owner", auth_user="alice")
    # claiming as a non-owner role is refused (would strand it ownerless)
    with pytest.raises(AccessDenied):
        handle_member("add", "demo", vr, CFG, user_id="alice", role="editor", auth_user="alice")


def test_web_non_owner_cannot_manage_claimed(tmp_path):
    vr, db = _vault(tmp_path)
    add_member(db, "demo", "alice", "owner", "t0")
    add_member(db, "demo", "carol", "editor", "t1")
    with pytest.raises(AccessDenied):
        handle_member("add", "demo", vr, CFG, user_id="dave", role="viewer", auth_user="carol")
    with pytest.raises(AccessDenied):
        handle_member("remove", "demo", vr, CFG, user_id="alice", auth_user="carol")


def test_web_owner_can_add_and_remove(tmp_path):
    vr, db = _vault(tmp_path)
    add_member(db, "demo", "alice", "owner", "t0")
    handle_member("add", "demo", vr, CFG, user_id="dave", role="viewer", auth_user="alice")
    assert any(m["user_id"] == "dave" for m in handle_member("list", "demo", vr, CFG, auth_user="alice")["members"])
    handle_member("remove", "demo", vr, CFG, user_id="dave", auth_user="alice")
    assert not any(m["user_id"] == "dave" for m in handle_member("list", "demo", vr, CFG, auth_user="alice")["members"])


def test_cannot_remove_last_owner(tmp_path):
    vr, db = _vault(tmp_path)
    add_member(db, "demo", "alice", "owner", "t0")
    with pytest.raises(ValueError):
        handle_member("remove", "demo", vr, CFG, user_id="alice", auth_user=None)


def test_can_remove_owner_when_another_remains(tmp_path):
    vr, db = _vault(tmp_path)
    add_member(db, "demo", "alice", "owner", "t0")
    add_member(db, "demo", "bob", "owner", "t1")
    handle_member("remove", "demo", vr, CFG, user_id="alice", auth_user=None)
    roles = [m["role"] for m in handle_member("list", "demo", vr, CFG, auth_user=None)["members"]]
    assert roles == ["owner"]


def test_non_member_cannot_list_claimed(tmp_path):
    vr, db = _vault(tmp_path)
    add_member(db, "demo", "alice", "owner", "t0")
    with pytest.raises(AccessDenied):
        handle_member("list", "demo", vr, CFG, auth_user="stranger")
