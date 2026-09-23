"""Frozen Sentence-BERT feature extractor -- loaded once per process and
reused by both the training script and the live grading path. all-MiniLM-L6-v2
itself is never fine-tuned; only the small classifier trained on top of its
embeddings (see model_store.py, scripts/train_grader.py) is task-specific."""

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts):
    """list[str] -> (n, 384) float32 embeddings."""
    return get_embedder().encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
