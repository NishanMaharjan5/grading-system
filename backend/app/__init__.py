from flask import Flask
from flask_cors import CORS

from app.config import Config
from app.extensions import db
from app.routes.auth import auth_bp
from app.routes.rubrics import rubrics_bp
from app.routes.submissions import submissions_bp


def create_app(config_object=Config):
    config_object.validate()

    app = Flask(__name__)
    app.config.from_object(config_object)

    db.init_app(app)
    CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}})

    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(rubrics_bp, url_prefix="/api/rubrics")
    app.register_blueprint(submissions_bp, url_prefix="/api/submissions")

    from app import models  # noqa: F401  (registers models on db.metadata before create_all)

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    return app
