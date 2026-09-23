from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.connection import assignments_collection, db, submissions_collection, users_collection
from app.routers.auth import router as auth_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await db.command("ping")
        # Unique index is what makes register's duplicate-email check race-free
        await users_collection.create_index("email", unique=True)
        await assignments_collection.create_index("created_by")
        # one submission per student per assignment
        await submissions_collection.create_index([("assignment_id", 1), ("student_id", 1)], unique=True)
        print("Connected to MongoDB")
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
    yield


app = FastAPI(
    title="Automated Assignment Grading and Feedback System",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/auth", tags=["auth"])


@app.get("/health")
async def health():
    return {"status": "healthy"}
