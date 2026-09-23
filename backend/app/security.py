from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import current_app, g, jsonify, request
from jose import JWTError, jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_token(user_id: int, email: str, role: str) -> str:
    cfg = current_app.config
    expire = datetime.now(timezone.utc) + timedelta(minutes=cfg["JWT_EXPIRE_MINUTES"])
    payload = {"sub": str(user_id), "email": email, "role": role, "exp": expire}
    return jwt.encode(payload, cfg["JWT_SECRET"], algorithm=cfg["JWT_ALGORITHM"])


def decode_token(token: str) -> dict:
    cfg = current_app.config
    return jwt.decode(token, cfg["JWT_SECRET"], algorithms=[cfg["JWT_ALGORITHM"]])


def login_required(fn):
    """Sets g.current_user to the decoded JWT payload, or responds 401.
    Flask has no dependency-injection like FastAPI's Depends, so this is a
    plain decorator instead of app.security.get_current_user."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify(detail="Missing bearer token"), 401
        token = auth_header[len("Bearer "):]
        try:
            g.current_user = decode_token(token)
        except JWTError:
            return jsonify(detail="Invalid or expired token"), 401
        return fn(*args, **kwargs)
    return wrapper


def require_role(*roles):
    """`@require_role("teacher")` -- implies login_required, then checks role."""
    def decorator(fn):
        @wraps(fn)
        def role_checked(*args, **kwargs):
            if g.current_user.get("role") not in roles:
                return jsonify(detail="You do not have permission to do this"), 403
            return fn(*args, **kwargs)
        return login_required(role_checked)
    return decorator
