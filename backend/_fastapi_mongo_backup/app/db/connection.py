import os

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "grading_system")

client = AsyncIOMotorClient(MONGODB_URI)
db = client[MONGODB_DB]

# Collections
users_collection = db["users"]
assignments_collection = db["assignments"]
submissions_collection = db["submissions"]
