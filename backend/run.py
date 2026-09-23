from app import create_app

app = create_app()

if __name__ == "__main__":
    # Schema is managed by Alembic, not create_all(): run `flask db upgrade`
    # (see migrations/README) before starting for the first time or after
    # pulling a change that touches the models.
    app.run(host="0.0.0.0", port=8000, debug=True)
