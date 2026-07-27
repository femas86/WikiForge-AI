from pkms.auth import ROLES, current_user, role_allows


# ── role model ────────────────────────────────────────────────────────────────

def test_roles_ascending_privilege():
    assert ROLES == ("viewer", "editor", "owner")


def test_role_allows_hierarchy():
    # owner ≥ editor ≥ viewer
    assert role_allows("owner", "viewer")
    assert role_allows("owner", "editor")
    assert role_allows("owner", "owner")
    assert role_allows("editor", "viewer")
    assert role_allows("editor", "editor")
    assert not role_allows("editor", "owner")
    assert not role_allows("viewer", "editor")
    assert role_allows("viewer", "viewer")


def test_role_allows_rejects_non_member_and_unknown():
    assert not role_allows(None, "viewer")        # not a member
    assert not role_allows("", "viewer")
    assert not role_allows("admin", "viewer")     # unknown role


# ── current_user resolution ─────────────────────────────────────────────────────

CFG = {"auth": {"user_header": "X-Auth-User", "default_user": "local"}}


def test_current_user_reads_trusted_header():
    assert current_user({"X-Auth-User": "alice"}, CFG) == "alice"


def test_current_user_header_case_insensitive():
    assert current_user({"x-auth-user": "bob"}, CFG) == "bob"


def test_current_user_falls_back_to_default():
    assert current_user({}, CFG) == "local"           # header absent
    assert current_user(None, CFG) == "local"         # CLI (no headers)
    assert current_user({"X-Auth-User": "  "}, CFG) == "local"  # blank → default


def test_current_user_defaults_when_config_missing():
    assert current_user(None, {}) == "local"          # no auth section → built-in default


def test_current_user_honours_custom_header_and_default():
    cfg = {"auth": {"user_header": "X-Forwarded-User", "default_user": "anon"}}
    assert current_user({"X-Forwarded-User": "carol"}, cfg) == "carol"
    assert current_user({}, cfg) == "anon"
