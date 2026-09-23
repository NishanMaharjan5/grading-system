from app import create_app
from app.extensions import db

app = create_app()

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # fine for a capstone; switch to Flask-Migrate/Alembic before this schema needs to evolve safely
    app.run(host="0.0.0.0", port=8000, debug=True)
