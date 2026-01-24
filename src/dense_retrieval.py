"""
Dense Vector Retrieval Module
Implements dense retrieval using sentence transformers and FAISS
"""

import json
import pickle
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DenseRetriever:
    """Dense vector retrieval using sentence embeddings and FAISS"""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize dense retriever

        Args:
            model_name: Sentence transformer model name
        """
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = None

        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for texts

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings
        """
        logger.info(f"Generating embeddings for {len(texts)} texts...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        return embeddings

    def build_index(self, chunks: List[Dict], save_dir: str = None):
        """
        Build FAISS index from chunks

        Args:
            chunks: List of chunk dictionaries with 'text' field
            save_dir: Directory to save index and chunks
        """
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]

        logger.info(f"Building FAISS index for {len(texts)} chunks...")

        # Generate embeddings
        embeddings = self.embed_texts(texts)

        # Create FAISS index (using Inner Product for normalized vectors = cosine similarity)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))

        logger.info(f"FAISS index built with {self.index.ntotal} vectors")

        # Save index if requested
        if save_dir:
            self.save_index(save_dir)

    def save_index(self, save_dir: str):
        """
        Save FAISS index and chunks

        Args:
            save_dir: Directory to save files
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = Path(save_dir) / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")

        # Save chunks
        chunks_path = Path(save_dir) / "chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved chunks to {chunks_path}")

    def load_index(self, load_dir: str):
        """
        Load FAISS index and chunks

        Args:
            load_dir: Directory containing saved files
        """
        # Load FAISS index
        index_path = Path(load_dir) / "faiss_index.bin"
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index from {index_path} ({self.index.ntotal} vectors)")

        # Load chunks
        chunks_path = Path(load_dir) / "chunks.json"
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        logger.info(f"Loaded {len(self.chunks)} chunks from {chunks_path}")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search for relevant chunks

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of dictionaries with chunk data and scores
        """
        if self.index is None or self.chunks is None:
            raise ValueError("Index not built or loaded. Call build_index() or load_index() first.")

        # Embed query
        query_embedding = self.embed_texts([query], show_progress=False)

        # Search FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):  # Valid index
                result = {
                    'chunk': self.chunks[idx],
                    'score': float(score),
                    'rank': len(results) + 1,
                    'retrieval_method': 'dense'
                }
                results.append(result)

        return results

    def batch_search(self, queries: List[str], top_k: int = 10) -> List[List[Dict]]:
        """
        Search for multiple queries

        Args:
            queries: List of queries
            top_k: Number of results per query

        Returns:
            List of result lists
        """
        if self.index is None or self.chunks is None:
            raise ValueError("Index not built or loaded. Call build_index() or load_index() first.")

        logger.info(f"Searching for {len(queries)} queries...")

        # Embed all queries
        query_embeddings = self.embed_texts(queries)

        # Search FAISS index
        scores, indices = self.index.search(query_embeddings.astype('float32'), top_k)

        # Prepare results for each query
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                if idx < len(self.chunks):
                    result = {
                        'chunk': self.chunks[idx],
                        'score': float(score),
                        'rank': len(results) + 1,
                        'retrieval_method': 'dense'
                    }
                    results.append(result)
            all_results.append(results)

        return all_results


if __name__ == "__main__":
    # Example usage
    retriever = DenseRetriever()

    # Load chunks
    with open("data/processed_chunks.json", 'r') as f:
        chunks = json.load(f)

    # Build index
    retriever.build_index(chunks, save_dir="data/vector_index")

    # Test search
    results = retriever.search("What is machine learning?", top_k=5)

    print("\nSearch Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Title: {result['chunk']['title']}")
        print(f"   Text: {result['chunk']['text'][:200]}...")
