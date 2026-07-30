import hashlib
import logging
import time

import requests
from opensearchpy import OpenSearch, helpers

from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

INDEX_NAME = "angioedema"

INDEX_MAPPING = {
    "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 100, "number_of_replicas": 0}},
    "mappings": {
        "properties": {
            "text": {"type": "text", "analyzer": "standard"},
            "vector": {
                "type": "knn_vector",
                "dimension": 768,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                    "parameters": {"ef_construction": 128, "m": 16},
                },
            },
            "source": {"type": "keyword"},
            "page": {"type": "integer"},
            "chunk_id": {"type": "keyword"},
            "language": {"type": "keyword"},
            "strategy": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "token_count": {"type": "integer"},
        }
    },
}


def build_chunk_id(source: str, chunk_index: int, strategy: str) -> str:
    """
    Generates a deterministic chunk ID based on source, index and strategy.
    Using a hash ensures the ID is always the same for the same chunk,
    which allows safe re-indexing without duplicates.
    """
    raw = f"{source}_{strategy}_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_embedding(
    text: str,
    ollama_url: str,
    model: str = "nomic-embed-text",
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> list[float]:
    """
    Calls the Ollama API to generate an embedding for a text.
    Retries on failure with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{ollama_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]

        except Exception as e:
            if attempt < max_retries - 1:
                wait = retry_delay * (2 ** attempt)
                logger.warning(
                    f"Embedding attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                raise

def create_index(client: OpenSearch) -> None:
    """
    Creates the OpenSearch index with the correct mapping.
    Skips creation if the index already exists.
    """
    if client.indices.exists(index=INDEX_NAME):
        logger.info(f"Index '{INDEX_NAME}' already exists — skipping creation")
        return

    client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
    logger.info(f"Index '{INDEX_NAME}' created successfully")


def delete_index(client: OpenSearch) -> None:
    """
    Deletes the index if it exists.
    Useful for re-indexing from scratch.
    """
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
        logger.info(f"Index '{INDEX_NAME}' deleted")
    else:
        logger.info(f"Index '{INDEX_NAME}' does not exist — nothing to delete")


def delete_chunks_by_source(client: OpenSearch, source: str) -> int:
    """
    Deletes all chunks from a specific source document before re-indexing.
    Prevents orphaned chunks when a document changes its number of chunks.
    """
    if not client.indices.exists(index=INDEX_NAME):
        return 0

    response = client.delete_by_query(
        index=INDEX_NAME, body={"query": {"term": {"source": source}}}
    )

    deleted = response.get("deleted", 0)
    logger.info(f"Deleted {deleted} existing chunks for source: {source}")
    return deleted

def chunk_exists(client: OpenSearch, chunk_id: str) -> bool:
    """Checks if a chunk already exists in the index."""
    return client.exists(index=INDEX_NAME, id=chunk_id)

def index_chunks(
    chunks: list[Chunk],
    client: OpenSearch,
    ollama_url: str,
    embedding_model: str = "nomic-embed-text",
    batch_size: int = 10,
    skip_existing: bool = False
) -> dict:
    """
    Indexes a list of chunks into OpenSearch.
    Generates embeddings in batches and uses bulk indexing for efficiency.

    Args:
        skip_existing: if True, skips chunks already indexed (retry mode).
                       if False, deletes existing chunks before re-indexing (normal mode).

    Returns a summary with total indexed, failed and skipped counts.
    """
    total = len(chunks)
    indexed = 0
    failed = 0
    skipped = 0

    # delete existing chunks only when not in retry mode
    if chunks and not skip_existing:
        source = chunks[0].metadata.get("source", "unknown")
        delete_chunks_by_source(client, source)

    logger.info(
        f"Starting indexing of {total} chunks in batches of {batch_size} "
        f"(retry mode: {skip_existing})"
    )

    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start:batch_start + batch_size]
        actions = []

        for chunk in batch:
            try:
                chunk_id = build_chunk_id(
                    source=chunk.metadata.get("source", "unknown"),
                    chunk_index=chunk.chunk_index,
                    strategy=chunk.strategy
                )

                # skip if already indexed in retry mode
                if skip_existing and chunk_exists(client, chunk_id):
                    logger.debug(f"Chunk {chunk_id} already exists — skipping")
                    skipped += 1
                    continue

                vector = get_embedding(
                    text=chunk.text,
                    ollama_url=ollama_url,
                    model=embedding_model
                )

                document = {
                    "chunk_id": chunk_id,
                    "text": chunk.text,
                    "vector": vector,
                    "source": chunk.metadata.get("source", "unknown"),
                    "language": chunk.metadata.get("language", "pt"),
                    "strategy": chunk.strategy,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "page": chunk.metadata.get("page", 0)
                }

                actions.append({
                    "_index": INDEX_NAME,
                    "_id": chunk_id,
                    "_source": document
                })

            except Exception as e:  # noqa: BLE001
                logger.error(
                    f"Failed to generate embedding for chunk "
                    f"{chunk.chunk_index} from {chunk.metadata.get('source')}: {e}"
                )
                logger.error(
                    f"Chunk token count: {chunk.token_count} | "
                    f"Text preview: {chunk.text[:100]}"
                )
                failed += 1
                continue

        if actions:
            success, errors = helpers.bulk(
                client,
                actions,
                raise_on_error=False
            )
            indexed += success
            if errors:
                logger.error(f"Bulk indexing errors: {errors}")
                failed += len(errors)

        logger.info(
            f"Progress: {min(batch_start + batch_size, total)}/{total} chunks processed"
        )

    summary = {
        "total": total,
        "indexed": indexed,
        "failed": failed,
        "skipped": skipped
    }

    logger.info(f"Indexing complete: {summary}")
    return summary
