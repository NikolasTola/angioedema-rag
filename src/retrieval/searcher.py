import logging

from opensearchpy import OpenSearch

from src.retrieval.embedder import Embedder

logger = logging.getLogger(__name__)


class SearchResult:
    """Represents a single search result with its content and scores."""

    def __init__(
        self,
        chunk_id: str,
        text: str,
        source: str,
        score: float,
        metadata: dict
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.source = source
        self.score = score
        self.metadata = metadata

    def __repr__(self):
        return (
            f"SearchResult(source={self.source}, "
            f"score={self.score:.4f}, "
            f"text={self.text[:80]}...)"
        )


class HybridSearcher:
    """
    Performs hybrid search combining semantic (knn) and keyword (BM25) search.
    Uses Reciprocal Rank Fusion (RRF) implemented in Python to merge rankings.
    """

    def __init__(
        self,
        client: OpenSearch,
        embedder: Embedder,
        index_name: str = "angioedema",
        top_k: int = 20,
        rrf_rank_constant: int = 60
    ):
        self.client = client
        self.embedder = embedder
        self.index_name = index_name
        self.top_k = top_k
        self.rrf_rank_constant = rrf_rank_constant

    def _search_semantic(self, query_vector: list[float]) -> list[dict]:
        """Executes knn vector search and returns ranked hits."""
        response = self.client.search(
            index=self.index_name,
            body={
                "size": self.top_k,
                "_source": {"excludes": ["vector"]},
                "query": {
                    "knn": {
                        "vector": {
                            "vector": query_vector,
                            "k": self.top_k
                        }
                    }
                }
            }
        )
        return response["hits"]["hits"]

    def _search_keyword(self, query_text: str) -> list[dict]:
        """Executes BM25 keyword search and returns ranked hits."""
        response = self.client.search(
            index=self.index_name,
            body={
                "size": self.top_k,
                "_source": {"excludes": ["vector"]},
                "query": {
                    "match": {
                        "text": {
                            "query": query_text
                        }
                    }
                }
            }
        )
        return response["hits"]["hits"]

    def _apply_rrf(
        self,
        semantic_hits: list[dict],
        keyword_hits: list[dict]
    ) -> list[tuple[str, float]]:
        """
        Applies Reciprocal Rank Fusion to merge two ranked lists.

        RRF formula: score(d) = sum(1 / (rank + k)) for each ranking list
        where k is the rank constant (default 60).

        Returns list of (chunk_id, rrf_score) sorted by score descending.
        """
        rrf_scores = {}

        for rank, hit in enumerate(semantic_hits):
            chunk_id = hit["_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0)
            rrf_scores[chunk_id] += 1.0 / (rank + self.rrf_rank_constant)

        for rank, hit in enumerate(keyword_hits):
            chunk_id = hit["_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0)
            rrf_scores[chunk_id] += 1.0 / (rank + self.rrf_rank_constant)

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def _build_result(
        self,
        chunk_id: str,
        rrf_score: float,
        all_hits: dict[str, dict]
    ) -> SearchResult:
        """Builds a SearchResult from a chunk_id and its source document."""
        hit = all_hits[chunk_id]
        source = hit["_source"]
        return SearchResult(
            chunk_id=chunk_id,
            text=source.get("text", ""),
            source=source.get("source", ""),
            score=rrf_score,
            metadata={
                "language": source.get("language", ""),
                "strategy": source.get("strategy", ""),
                "chunk_index": source.get("chunk_index", 0),
                "token_count": source.get("token_count", 0),
            }
        )

    def search(self, query: str) -> list[SearchResult]:
        """
        Executes hybrid search with RRF fusion.
        Returns top_k results ranked by RRF score.
        """
        logger.info(f"Searching for: '{query}'")

        query_vector = self.embedder.embed(query)

        semantic_hits = self._search_semantic(query_vector)
        keyword_hits = self._search_keyword(query)

        logger.info(
            f"Semantic hits: {len(semantic_hits)} | "
            f"Keyword hits: {len(keyword_hits)}"
        )

        # merge all hits into a lookup dict for fast access
        all_hits = {hit["_id"]: hit for hit in semantic_hits + keyword_hits}

        # apply rrf and get ranked chunk_ids
        ranked = self._apply_rrf(semantic_hits, keyword_hits)

        # build results limited to top_k
        results = [
            self._build_result(chunk_id, score, all_hits)
            for chunk_id, score in ranked[:self.top_k]
        ]

        logger.info(f"Returning {len(results)} results after RRF fusion")
        return results