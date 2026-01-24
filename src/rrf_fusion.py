"""
Reciprocal Rank Fusion (RRF) Module
Combines results from multiple retrieval methods
"""

from typing import List, Dict
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReciprocalRankFusion:
    """Implements Reciprocal Rank Fusion for combining retrieval results"""

    def __init__(self, k: int = 60):
        """
        Initialize RRF

        Args:
            k: RRF constant (default 60 as per TREC standard)
        """
        self.k = k
        logger.info(f"RRF initialized with k={k}")

    def compute_rrf_score(self, rank: int) -> float:
        """
        Compute RRF score for a given rank

        Args:
            rank: Rank position (1-indexed)

        Returns:
            RRF score
        """
        return 1.0 / (self.k + rank)

    def fuse(self, result_lists: List[List[Dict]], top_n: int = 5) -> List[Dict]:
        """
        Fuse multiple result lists using RRF

        Args:
            result_lists: List of result lists from different retrieval methods
                         Each result should have 'chunk' with 'chunk_id' and 'rank'
            top_n: Number of top results to return

        Returns:
            Fused and ranked results
        """
        # Dictionary to accumulate RRF scores by chunk_id
        rrf_scores = defaultdict(float)

        # Dictionary to store chunk data
        chunk_data = {}

        # Dictionary to store individual method scores and ranks
        method_scores = defaultdict(dict)

        # Process each result list
        for method_idx, results in enumerate(result_lists):
            for result in results:
                chunk = result['chunk']
                chunk_id = chunk['chunk_id']
                rank = result['rank']

                # Accumulate RRF score
                rrf_score = self.compute_rrf_score(rank)
                rrf_scores[chunk_id] += rrf_score

                # Store chunk data (first occurrence)
                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = chunk

                # Store method-specific scores
                method_name = result.get('retrieval_method', f'method_{method_idx}')
                method_scores[chunk_id][method_name] = {
                    'score': result.get('score', 0.0),
                    'rank': rank,
                    'rrf_contribution': rrf_score
                }

        # Sort by RRF score
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        # Prepare final results
        fused_results = []
        for final_rank, (chunk_id, rrf_score) in enumerate(sorted_chunks, 1):
            result = {
                'chunk': chunk_data[chunk_id],
                'rrf_score': rrf_score,
                'final_rank': final_rank,
                'method_details': method_scores[chunk_id],
                'retrieval_method': 'hybrid_rrf'
            }
            fused_results.append(result)

        return fused_results

    def fuse_with_weights(
        self,
        result_lists: List[List[Dict]],
        weights: List[float] = None,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Fuse results with weighted RRF

        Args:
            result_lists: List of result lists
            weights: Weight for each method (default: equal weights)
            top_n: Number of top results to return

        Returns:
            Weighted fused results
        """
        if weights is None:
            weights = [1.0] * len(result_lists)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Dictionary to accumulate weighted RRF scores
        rrf_scores = defaultdict(float)
        chunk_data = {}
        method_scores = defaultdict(dict)

        # Process each result list with weights
        for method_idx, (results, weight) in enumerate(zip(result_lists, weights)):
            for result in results:
                chunk = result['chunk']
                chunk_id = chunk['chunk_id']
                rank = result['rank']

                # Accumulate weighted RRF score
                rrf_score = self.compute_rrf_score(rank) * weight
                rrf_scores[chunk_id] += rrf_score

                # Store chunk data
                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = chunk

                # Store method-specific scores
                method_name = result.get('retrieval_method', f'method_{method_idx}')
                method_scores[chunk_id][method_name] = {
                    'score': result.get('score', 0.0),
                    'rank': rank,
                    'weight': weight,
                    'weighted_rrf_contribution': rrf_score
                }

        # Sort by weighted RRF score
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        # Prepare final results
        fused_results = []
        for final_rank, (chunk_id, rrf_score) in enumerate(sorted_chunks, 1):
            result = {
                'chunk': chunk_data[chunk_id],
                'rrf_score': rrf_score,
                'final_rank': final_rank,
                'method_details': method_scores[chunk_id],
                'retrieval_method': 'hybrid_weighted_rrf'
            }
            fused_results.append(result)

        return fused_results


class HybridRetriever:
    """Combines dense and sparse retrieval with RRF"""

    def __init__(self, dense_retriever, sparse_retriever, rrf_k: int = 60):
        """
        Initialize hybrid retriever

        Args:
            dense_retriever: DenseRetriever instance
            sparse_retriever: BM25Retriever instance
            rrf_k: RRF constant
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf = ReciprocalRankFusion(k=rrf_k)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        top_n: int = 5,
        weights: List[float] = None
    ) -> Dict:
        """
        Hybrid retrieval with RRF fusion

        Args:
            query: Search query
            top_k: Number of results to retrieve from each method
            top_n: Number of final fused results
            weights: Optional weights for each method [dense_weight, sparse_weight]

        Returns:
            Dictionary with fused results and individual method results
        """
        # Retrieve from dense retriever
        dense_results = self.dense_retriever.search(query, top_k=top_k)

        # Retrieve from sparse retriever
        sparse_results = self.sparse_retriever.search(query, top_k=top_k)

        # Fuse results
        if weights:
            fused_results = self.rrf.fuse_with_weights(
                [dense_results, sparse_results],
                weights=weights,
                top_n=top_n
            )
        else:
            fused_results = self.rrf.fuse(
                [dense_results, sparse_results],
                top_n=top_n
            )

        return {
            'query': query,
            'fused_results': fused_results,
            'dense_results': dense_results,
            'sparse_results': sparse_results,
            'num_unique_chunks': len(set(
                [r['chunk']['chunk_id'] for r in dense_results] +
                [r['chunk']['chunk_id'] for r in sparse_results]
            ))
        }

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        top_n: int = 5,
        weights: List[float] = None
    ) -> List[Dict]:
        """
        Batch hybrid retrieval

        Args:
            queries: List of queries
            top_k: Number of results to retrieve from each method
            top_n: Number of final fused results
            weights: Optional weights for each method

        Returns:
            List of retrieval results for each query
        """
        logger.info(f"Performing hybrid retrieval for {len(queries)} queries...")

        all_results = []
        for query in queries:
            results = self.retrieve(query, top_k, top_n, weights)
            all_results.append(results)

        return all_results


if __name__ == "__main__":
    # Example usage
    from dense_retrieval import DenseRetriever
    from sparse_retrieval import BM25Retriever

    # Load retrievers
    dense_retriever = DenseRetriever()
    dense_retriever.load_index("data/vector_index")

    sparse_retriever = BM25Retriever()
    sparse_retriever.load_index("data/bm25_index.pkl")

    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)

    # Test retrieval
    results = hybrid_retriever.retrieve(
        "What is machine learning?",
        top_k=10,
        top_n=5
    )

    print("\nHybrid Retrieval Results:")
    for i, result in enumerate(results['fused_results'], 1):
        print(f"\n{i}. RRF Score: {result['rrf_score']:.4f}")
        print(f"   Title: {result['chunk']['title']}")
        print(f"   Text: {result['chunk']['text'][:200]}...")
        print(f"   Method Details: {result['method_details']}")
