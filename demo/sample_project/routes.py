from fastapi import APIRouter, HTTPException

import sample_project.storage as storage
from sample_project.models import User, UserCreate
from sample_project.utils import normalize_email

router = APIRouter(prefix="/users")


@router.post("", response_model=User, status_code=201)
def create_user(payload: UserCreate):
    data = payload.model_dump()
    data["email"] = normalize_email(data["email"])
    return storage.add_user(data)


@router.get("", response_model=list[User])
def list_users(email: str | None = None):
    if email:
        return storage.find_by_email(email)
    return storage.list_users()


@router.get("/{user_id}", response_model=User)
def get_user(user_id: int):
    user = storage.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.delete("/{user_id}", status_code=204)
def delete_user(user_id: int):
    if not storage.delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")