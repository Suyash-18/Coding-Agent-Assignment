from sample_project.utils import normalize_email

VALID = {"name": "Asha Rao", "email": "Asha@Example.com", "age": 30}


def test_create_user(client):
    r = client.post("/users", json=VALID)
    assert r.status_code == 201
    body = r.json()
    assert body["id"] == 1
    assert body["email"] == "asha@example.com"


def test_get_user(client):
    created = client.post("/users", json=VALID).json()
    r = client.get(f"/users/{created['id']}")
    assert r.status_code == 200
    assert r.json()["name"] == "Asha Rao"


def test_get_missing_user_returns_404(client):
    assert client.get("/users/999").status_code == 404


def test_list_users(client):
    client.post("/users", json=VALID)
    assert len(client.get("/users").json()) == 1


def test_filter_users_by_email(client):
    client.post("/users", json=VALID)
    r = client.get("/users", params={"email": "ASHA@example.com"})
    assert len(r.json()) == 1


def test_delete_user(client):
    created = client.post("/users", json=VALID).json()
    assert client.delete(f"/users/{created['id']}").status_code == 204
    assert client.get(f"/users/{created['id']}").status_code == 404


def test_normalize_email():
    assert normalize_email("  Foo@Bar.COM ") == "foo@bar.com"