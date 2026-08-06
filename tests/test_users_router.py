"""Tests for the admin-only user provisioning router."""

import uuid

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from auth import verify_auth, verify_password
from db import Base, get_db
from models import User
from routers import users as users_router

ADMIN_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
MEMBER_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        yield session
    finally:
        session.close()


def _make_client(db_session, acting_user):
    app = FastAPI()
    app.include_router(users_router.router)
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[verify_auth] = lambda: acting_user
    return TestClient(app)


@pytest.fixture
def admin(db_session):
    user = User(
        id=ADMIN_ID,
        name="Admin",
        email="admin@example.com",
        password_hash="unused",
        role="admin",
    )
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def member(db_session):
    user = User(
        id=MEMBER_ID,
        name="Member",
        email="member@example.com",
        password_hash="unused",
        role="user",
    )
    db_session.add(user)
    db_session.commit()
    return user


def test_admin_creates_user_with_user_role(db_session, admin):
    client = _make_client(db_session, admin)

    response = client.post(
        "/users",
        json={"name": "Neue Person", "email": "np@example.com", "password": "initial-pass-1"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "np@example.com"
    assert body["role"] == "user"
    assert "password" not in body
    created = db_session.scalar(select(User).where(User.email == "np@example.com"))
    assert created.role == "user"
    assert created.password_hash != "initial-pass-1"
    assert verify_password("initial-pass-1", created.password_hash)


def test_client_cannot_grant_admin_role(db_session, admin):
    client = _make_client(db_session, admin)

    response = client.post(
        "/users",
        json={
            "name": "Sneaky",
            "email": "sneaky@example.com",
            "password": "initial-pass-1",
            "role": "admin",
        },
    )

    assert response.status_code == 201
    assert response.json()["role"] == "user"


def test_non_admin_cannot_create_user(db_session, admin, member):
    client = _make_client(db_session, member)

    response = client.post(
        "/users",
        json={"name": "X", "email": "x@example.com", "password": "initial-pass-1"},
    )

    assert response.status_code == 403
    assert db_session.scalar(select(User).where(User.email == "x@example.com")) is None


def test_duplicate_email_conflicts(db_session, admin, member):
    client = _make_client(db_session, admin)

    response = client.post(
        "/users",
        json={"name": "Copy", "email": "member@example.com", "password": "initial-pass-1"},
    )

    assert response.status_code == 409


def test_short_password_rejected(db_session, admin):
    client = _make_client(db_session, admin)

    response = client.post(
        "/users",
        json={"name": "X", "email": "x@example.com", "password": "short"},
    )

    assert response.status_code == 400
    assert db_session.scalar(select(User).where(User.email == "x@example.com")) is None


def test_blank_name_rejected(db_session, admin):
    client = _make_client(db_session, admin)

    response = client.post(
        "/users",
        json={"name": "   ", "email": "x@example.com", "password": "initial-pass-1"},
    )

    assert response.status_code == 400


def test_admin_lists_users_without_secrets(db_session, admin, member):
    client = _make_client(db_session, admin)

    response = client.get("/users")

    assert response.status_code == 200
    body = response.json()
    assert [entry["email"] for entry in body] == ["admin@example.com", "member@example.com"]
    assert all("password_hash" not in entry for entry in body)


def test_non_admin_cannot_list_users(db_session, admin, member):
    client = _make_client(db_session, member)

    response = client.get("/users")

    assert response.status_code == 403
