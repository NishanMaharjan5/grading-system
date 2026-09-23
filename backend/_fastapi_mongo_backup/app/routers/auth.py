import os
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.errors import DuplicateKeyError

from app.db.connection import users_collection
from app.schemas.auth_schema import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from app.security import create_token, get_current_user, hash_password, verify_password

router = APIRouter()


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest):
    # Without this gate anyone could self-register as a teacher and approve their own grades.
    # If TEACHER_SIGNUP_CODE is set, it must be supplied to create a teacher account.
    if body.role == "teacher":
        required = os.getenv("TEACHER_SIGNUP_CODE")
        if required and body.teacher_code != required:
            raise HTTPException(status_code=403, detail="Invalid teacher signup code")

    user_doc = {
        "name": body.name.strip(),
        "email": body.email.lower(),
        "password": hash_password(body.password),
        "role": body.role,
        "created_at": datetime.now(timezone.utc),
    }
    try:
        result = await users_collection.insert_one(user_doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Email already registered")

    token = create_token(str(result.inserted_id), user_doc["email"], body.role)
    return TokenResponse(access_token=token, name=user_doc["name"], role=body.role)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    user = await users_collection.find_one({"email": body.email.lower()})
    if not user or not verify_password(body.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token(str(user["_id"]), user["email"], user["role"])
    return TokenResponse(access_token=token, name=user["name"], role=user["role"])


@router.get("/me", response_model=UserResponse)
async def me(current_user: dict = Depends(get_current_user)):
    user = await users_collection.find_one({"_id": ObjectId(current_user["sub"])})
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return UserResponse(id=str(user["_id"]), name=user["name"], email=user["email"], role=user["role"])
