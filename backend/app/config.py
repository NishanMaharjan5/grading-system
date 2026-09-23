import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    JWT_SECRET = os.getenv("JWT_SECRET")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
    TEACHER_SIGNUP_CODE = os.getenv("TEACHER_SIGNUP_CODE") or None

    @staticmethod
    def validate():
        if not Config.SQLALCHEMY_DATABASE_URI:
            raise RuntimeError(
                "DATABASE_URL environment variable is not set. Put a PostgreSQL URL "
                "(e.g. postgresql+psycopg2://user:pass@localhost:5432/grading_system) "
                "in backend/.env before starting the app."
            )
        if not Config.JWT_SECRET:
            raise RuntimeError(
                "JWT_SECRET environment variable is not set. Generate one with "
                "python3 -c \"import secrets; print(secrets.token_hex(32))\" and put it "
                "in backend/.env before starting the app."
            )
