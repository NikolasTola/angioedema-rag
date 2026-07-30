import logging

from sentence_transformers import CrossEncoder

from src.retrieval.searcher import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """
    Reranks search results using a cross-encoder model.
    Evaluates query and chunk together for precise relevance scoring.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5
    ):
        self.top_n = top_n
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker model loaded")

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """
        Reranks search results by relevance to the query.
        Returns top_n results ordered by cross-encoder score.
        """
        if not results:
            return []

        pairs = [[query, result.text] for result in results]
        scores = self.model.predict(pairs)

        for result, score in zip(results, scores):
            result.score = float(score)

        reranked = sorted(results, key=lambda r: r.score, reverse=True)

        logger.info(
            f"Reranked {len(results)} results — "
            f"returning top {self.top_n}"
        )

        return reranked[:self.top_n]