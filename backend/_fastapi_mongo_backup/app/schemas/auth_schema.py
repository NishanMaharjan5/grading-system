from typing import Literal, Optional

from pydantic import BaseModel, EmailStr, Field

Role = Literal["student", "teacher"]


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    # bcrypt only uses the first 72 bytes, so cap the length rather than silently truncate
    password: str = Field(min_length=8, max_length=72)
    role: Role = "student"
    teacher_code: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    name: str
    role: Role


class UserResponse(BaseModel):
    id: str
    name: str
    email: EmailStr
    role: Role
