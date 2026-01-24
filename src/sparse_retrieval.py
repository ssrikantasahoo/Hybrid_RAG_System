"""
Sparse Keyword Retrieval Module
Implements BM25 algorithm for keyword-based retrieval
"""

import json
import pickle
from typing import List, Dict
from pathlib import Path
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class BM25Retriever:
    """BM25 keyword-based retrieval"""

    def __init__(self, k1: float = 1.5, b: float = 0.75, use_stopwords: bool = True):
        """
        Initialize BM25 retriever

        Args:
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
            use_stopwords: Whether to remove stopwords
        """
        self.k1 = k1
        self.b = b
        self.use_stopwords = use_stopwords
        self.bm25 = None
        self.chunks = None
        self.tokenized_corpus = None

        if use_stopwords:
            self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = set()

        logger.info(f"BM25 Retriever initialized (k1={k1}, b={b}, stopwords={use_stopwords})")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize and preprocess text

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Tokenize
        tokens = word_tokenize(text.lower())

        # Remove stopwords and non-alphabetic tokens
        if self.use_stopwords:
            tokens = [
                token for token in tokens
                if token.isalpha() and token not in self.stopwords
            ]
        else:
            tokens = [token for token in tokens if token.isalpha()]

        return tokens

    def build_index(self, chunks: List[Dict], save_path: str = None):
        """
        Build BM25 index from chunks

        Args:
            chunks: List of chunk dictionaries with 'text' field
            save_path: Path to save index
        """
        self.chunks = chunks
        logger.info(f"Building BM25 index for {len(chunks)} chunks...")

        # Tokenize all chunks
        self.tokenized_corpus = [
            self.tokenize(chunk['text'])
            for chunk in chunks
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b
        )

        logger.info(f"BM25 index built with {len(self.tokenized_corpus)} documents")

        # Save index if requested
        if save_path:
            self.save_index(save_path)

    def save_index(self, save_path: str):
        """
        Save BM25 index

        Args:
            save_path: Path to save index file
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        index_data = {
            'bm25': self.bm25,
            'chunks': self.chunks,
            'tokenized_corpus': self.tokenized_corpus,
            'k1': self.k1,
            'b': self.b,
            'use_stopwords': self.use_stopwords
        }

        with open(save_path, 'wb') as f:
            pickle.dump(index_data, f)

        logger.info(f"Saved BM25 index to {save_path}")

    def load_index(self, load_path: str):
        """
        Load BM25 index

        Args:
            load_path: Path to index file
        """
        with open(load_path, 'rb') as f:
            index_data = pickle.load(f)

        self.bm25 = index_data['bm25']
        self.chunks = index_data['chunks']
        self.tokenized_corpus = index_data['tokenized_corpus']
        self.k1 = index_data['k1']
        self.b = index_data['b']
        self.use_stopwords = index_data['use_stopwords']

        logger.info(f"Loaded BM25 index from {load_path} ({len(self.chunks)} chunks)")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search for relevant chunks using BM25

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of dictionaries with chunk data and scores
        """
        if self.bm25 is None or self.chunks is None:
            raise ValueError("Index not built or loaded. Call build_index() or load_index() first.")

        # Tokenize query
        query_tokens = self.tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        # Prepare results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            result = {
                'chunk': self.chunks[idx],
                'score': float(scores[idx]),
                'rank': rank,
                'retrieval_method': 'sparse'
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
        if self.bm25 is None or self.chunks is None:
            raise ValueError("Index not built or loaded. Call build_index() or load_index() first.")

        logger.info(f"Searching for {len(queries)} queries using BM25...")

        all_results = []
        for query in queries:
            results = self.search(query, top_k)
            all_results.append(results)

        return all_results


if __name__ == "__main__":
    # Example usage
    retriever = BM25Retriever()

    # Load chunks
    with open("data/processed_chunks.json", 'r') as f:
        chunks = json.load(f)

    # Build index
    retriever.build_index(chunks, save_path="data/bm25_index.pkl")

    # Test search
    results = retriever.search("machine learning algorithms", top_k=5)

    print("\nBM25 Search Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Title: {result['chunk']['title']}")
        print(f"   Text: {result['chunk']['text'][:200]}...")
