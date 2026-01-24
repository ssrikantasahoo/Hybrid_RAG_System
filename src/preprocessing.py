"""
Text Preprocessing and Chunking Module
Handles text cleaning, tokenization, and chunking with overlap
"""

import re
import json
import uuid
from typing import List, Dict
from pathlib import Path
import tiktoken
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Handles text cleaning and preprocessing"""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize preprocessor

        Args:
            encoding_name: Tiktoken encoding name for token counting
        """
        self.encoding = tiktoken.get_encoding(encoding_name)

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)

        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\']', ' ', text)

        # Strip whitespace
        text = text.strip()

        return text

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitter (can be enhanced with spaCy for better accuracy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class TextChunker:
    """Handles text chunking with overlap"""

    def __init__(self, chunk_size: int = 300, overlap: int = 50, encoding_name: str = "cl100k_base"):
        """
        Initialize chunker

        Args:
            chunk_size: Target chunk size in tokens
            overlap: Overlap size in tokens
            encoding_name: Tiktoken encoding name
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.preprocessor = TextPreprocessor(encoding_name)

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text with overlap

        Args:
            text: Input text
            metadata: Metadata to attach to each chunk (URL, title, etc.)

        Returns:
            List of chunk dictionaries
        """
        # Clean text
        text = self.preprocessor.clean_text(text)

        # Split into sentences for better chunk boundaries
        sentences = self.preprocessor.split_into_sentences(text)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.preprocessor.count_tokens(sentence)

            # If adding this sentence exceeds chunk size, save current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)

                # Calculate overlap
                overlap_tokens = 0
                overlap_sentences = []

                # Add sentences from the end until we reach overlap size
                for sent in reversed(current_chunk):
                    sent_tokens = self.preprocessor.count_tokens(sent)
                    if overlap_tokens + sent_tokens <= self.overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break

                # Start new chunk with overlap
                current_chunk = overlap_sentences
                current_tokens = overlap_tokens

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)

        # Create chunk dictionaries with metadata
        chunk_dicts = []
        for i, chunk_text in enumerate(chunks):
            chunk_dict = {
                'chunk_id': str(uuid.uuid4()),
                'chunk_index': i,
                'text': chunk_text,
                'token_count': self.preprocessor.count_tokens(chunk_text),
                **(metadata or {})
            }
            chunk_dicts.append(chunk_dict)

        return chunk_dicts

    def chunk_corpus(self, corpus: List[Dict]) -> List[Dict]:
        """
        Chunk entire corpus

        Args:
            corpus: List of documents with 'text', 'url', 'title'

        Returns:
            List of all chunks with metadata
        """
        all_chunks = []

        logger.info(f"Chunking {len(corpus)} documents...")

        for doc in tqdm(corpus, desc="Chunking documents"):
            metadata = {
                'url': doc['url'],
                'title': doc['title'],
                'source_type': doc.get('source_type', 'unknown'),
                'doc_word_count': doc.get('word_count', 0)
            }

            chunks = self.chunk_text(doc['text'], metadata)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(corpus)} documents")

        return all_chunks

    def save_chunks(self, chunks: List[Dict], output_file: str):
        """Save chunks to JSON file"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(chunks)} chunks to {output_file}")

    def load_chunks(self, file_path: str) -> List[Dict]:
        """Load chunks from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def preprocess_corpus(raw_corpus_file: str, output_file: str, chunk_size: int = 300, overlap: int = 50):
    """
    Complete preprocessing pipeline

    Args:
        raw_corpus_file: Path to raw corpus JSON
        output_file: Path to save processed chunks
        chunk_size: Chunk size in tokens
        overlap: Overlap size in tokens
    """
    # Load raw corpus
    with open(raw_corpus_file, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    # Create chunker
    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)

    # Chunk corpus
    chunks = chunker.chunk_corpus(corpus)

    # Save chunks
    chunker.save_chunks(chunks, output_file)

    # Print statistics
    logger.info(f"\nPreprocessing Statistics:")
    logger.info(f"Total documents: {len(corpus)}")
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"Average chunks per document: {len(chunks) / len(corpus):.2f}")
    logger.info(f"Chunk size range: {chunk_size - 50} - {chunk_size + 50} tokens")

    return chunks


if __name__ == "__main__":
    # Example usage
    preprocess_corpus(
        raw_corpus_file="data/raw_corpus.json",
        output_file="data/processed_chunks.json",
        chunk_size=300,
        overlap=50
    )
