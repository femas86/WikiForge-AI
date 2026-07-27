"""B4 member-management web UI — exercised through the new DI seam:
identity/config/vault are injected via app.dependency_overrides rather than
monkeypatching module globals."""
import pytest
from fastapi.testclient import TestClient

from pkms.db import add_member, init_db
from pkms.web import app, get_config, get_current_user, get_vault_root

CLIENT = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def _setup(tmp_path, user="alice", cfg=None):
    (tmp_path / "vault").mkdir()
    init_db(str(tmp_path / "vault" / ".search-index"))
    app.dependency_overrides[get_vault_root] = lambda: str(tmp_path)
    app.dependency_overrides[get_config] = lambda: (cfg or {})
    app.dependency_overrides[get_current_user] = lambda: user
    return str(tmp_path / "vault" / ".search-index")


def _act_as(user):
    app.dependency_overrides[get_current_user] = lambda: user


# ── page ────────────────────────────────────────────────────────────────────

def test_members_page_unclaimed_offers_claim(tmp_path):
    _setup(tmp_path, user="alice")
    resp = CLIENT.get("/members?project=demo")
    assert resp.status_code == 200
    assert "Claim project" in resp.text
    assert "unclaimed" in resp.text


def test_members_page_renders_roster_for_member(tmp_path):
    db = _setup(tmp_path, user="alice")
    add_member(db, "demo", "alice", "owner", "t0")
    add_member(db, "demo", "bob", "viewer", "t1")
    resp = CLIENT.get("/members?project=demo")
    assert resp.status_code == 200
    assert "bob" in resp.text and "viewer" in resp.text
    assert "Add user" in resp.text          # owner sees management controls


def test_members_page_hides_roster_from_non_member(tmp_path):
    db = _setup(tmp_path, user="stranger")
    add_member(db, "demo", "alice", "owner", "t0")
    resp = CLIENT.get("/members?project=demo")
    assert resp.status_code == 200
    assert "not a member" in resp.text
    assert "Add user" not in resp.text


# ── claim + add + remove ──────────────────────────────────────────────────────

def test_self_claim_makes_caller_owner(tmp_path):
    db = _setup(tmp_path, user="alice")
    resp = CLIENT.post("/ui/members/add", data={"project": "demo", "user_id": "alice", "role": "owner"})
    assert resp.status_code == 200
    assert "alice" in resp.text and "owner" in resp.text
    from pkms.db import get_member_role
    assert get_member_role(db, "demo", "alice") == "owner"


def test_owner_adds_member(tmp_path):
    db = _setup(tmp_path, user="alice")
    add_member(db, "demo", "alice", "owner", "t0")
    resp = CLIENT.post("/ui/members/add", data={"project": "demo", "user_id": "bob", "role": "editor"})
    assert resp.status_code == 200
    from pkms.db import get_member_role
    assert get_member_role(db, "demo", "bob") == "editor"


def test_non_owner_add_is_forbidden(tmp_path):
    db = _setup(tmp_path, user="carol")
    add_member(db, "demo", "alice", "owner", "t0")
    add_member(db, "demo", "carol", "editor", "t1")
    resp = CLIENT.post("/ui/members/add", data={"project": "demo", "user_id": "dave", "role": "viewer"})
    assert resp.status_code == 403


def test_web_claim_must_be_self(tmp_path):
    _setup(tmp_path, user="alice")
    # claiming by adding someone else on an unclaimed project → forbidden
    resp = CLIENT.post("/ui/members/add", data={"project": "demo", "user_id": "bob", "role": "owner"})
    assert resp.status_code == 403


def test_owner_removes_member(tmp_path):
    db = _setup(tmp_path, user="alice")
    add_member(db, "demo", "alice", "owner", "t0")
    add_member(db, "demo", "bob", "viewer", "t1")
    resp = CLIENT.post("/ui/members/remove", data={"project": "demo", "user_id": "bob"})
    assert resp.status_code == 200
    from pkms.db import get_member_role
    assert get_member_role(db, "demo", "bob") is None


def test_remove_last_owner_shows_error(tmp_path):
    db = _setup(tmp_path, user="alice")
    add_member(db, "demo", "alice", "owner", "t0")
    resp = CLIENT.post("/ui/members/remove", data={"project": "demo", "user_id": "alice"})
    assert resp.status_code == 200          # panel re-rendered with the error, not a 500
    assert "last owner" in resp.text
    from pkms.db import get_member_role
    assert get_member_role(db, "demo", "alice") == "owner"   # not removed


def test_add_invalid_role_shows_error(tmp_path):
    db = _setup(tmp_path, user="alice")
    add_member(db, "demo", "alice", "owner", "t0")
    resp = CLIENT.post("/ui/members/add", data={"project": "demo", "user_id": "bob", "role": "superuser"})
    assert resp.status_code == 200
    assert "invalid role" in resp.text
