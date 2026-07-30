import logging

import requests

logger = logging.getLogger(__name__)


class Embedder:
    """
    Generates text embeddings via Ollama API.
    Single responsibility: convert text to vector.
    """

    def __init__(self, ollama_url: str, model: str = "nomic-embed-text"):
        self.ollama_url = ollama_url
        self.model = model

    def embed(self, text: str) -> list[float]:
        """Generates an embedding for a single text."""
        response = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generates embeddings for a list of texts."""
        return [self.embed(text) for text in texts]