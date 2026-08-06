from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auth import hash_password, require_admin
from db import get_db
from models import User
from routers.auth import UserResponse, _require_password_length

router = APIRouter(prefix="/users", tags=["users"])


class CreateUserRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


@router.post("", response_model=UserResponse, status_code=201)
def create_user(
    body: CreateUserRequest,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create a regular user account. The at-most-one-admin rule stays intact:
    provisioned accounts always get the 'user' role."""
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name must not be empty.")
    _require_password_length(body.password)

    if db.scalar(select(User).where(User.email == body.email)) is not None:
        raise HTTPException(status_code=409, detail="Email is already in use.")

    user = User(
        name=name,
        email=body.email,
        password_hash=hash_password(body.password),
        role="user",
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Email is already in use.")
    db.refresh(user)
    return user


@router.get("", response_model=list[UserResponse])
def list_users(
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return db.scalars(select(User).order_by(User.created_at.asc())).all()
