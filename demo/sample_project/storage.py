from sample_project.utils import normalize_email

_users: dict[int, dict] = {}
_next_id = 1


def add_user(data: dict) -> dict:
    global _next_id
    user = {"id": _next_id, **data}
    _users[_next_id] = user
    _next_id += 1
    return user


def get_user(user_id: int) -> dict | None:
    return _users.get(user_id)


def list_users() -> list[dict]:
    return list(_users.values())


def find_by_email(email: str) -> list[dict]:
    target = normalize_email(email)
    return [u for u in _users.values() if normalize_email(u["email"]) == target]


def delete_user(user_id: int) -> bool:
    return _users.pop(user_id, None) is not None


def clear() -> None:
    global _next_id
    _users.clear()
    _next_id = 1