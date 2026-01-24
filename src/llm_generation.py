"""
LLM Generation Module
Handles answer generation using transformer models
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from typing import List, Dict
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMGenerator:
    """Generates answers using language models"""

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: str = None
    ):
        """
        Initialize LLM generator

        Args:
            model_name: HuggingFace model name
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading model: {model_name} on {self.device}")
        
        # Define local models directory
        self.local_model_dir = Path(__file__).parent.parent / "models" / model_name.split("/")[-1]
        
        # Load tokenizer
        if self.local_model_dir.exists():
            logger.info(f"Loading tokenizer from local path: {self.local_model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.local_model_dir))
        else:
            logger.info("Downloading tokenizer and saving locally...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.save_pretrained(str(self.local_model_dir))

        # Determine model type
        if "t5" in model_name.lower() or "flan" in model_name.lower():
            if self.local_model_dir.exists() and (self.local_model_dir / "config.json").exists():
               logger.info(f"Loading model from local path: {self.local_model_dir}")
               self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.local_model_dir))
            else:
               logger.info("Downloading model and saving locally...")
               self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
               self.model.save_pretrained(str(self.local_model_dir))
            
            self.model_type = "seq2seq"
        else:
            if self.local_model_dir.exists() and (self.local_model_dir / "config.json").exists():
               logger.info(f"Loading model from local path: {self.local_model_dir}")
               self.model = AutoModelForCausalLM.from_pretrained(str(self.local_model_dir))
            else:
               logger.info("Downloading model and saving locally...")
               self.model = AutoModelForCausalLM.from_pretrained(model_name)
               self.model.save_pretrained(str(self.local_model_dir))
               
            self.model_type = "causal"

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully (type: {self.model_type})")

    def create_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Create prompt from query and context chunks

        Args:
            query: User query
            context_chunks: List of retrieved chunks

        Returns:
            Formatted prompt
        """
        # Combine context from chunks
        context_parts = []
        for i, chunk_result in enumerate(context_chunks, 1):
            chunk = chunk_result['chunk']
            context_parts.append(
                f"[{i}] {chunk['title']}: {chunk['text']}"
            )

        context = "\n\n".join(context_parts)

        # Create prompt based on model type
        if self.model_type == "seq2seq":
            prompt = f"""Answer the following question based on the given context. Be concise and accurate.

Context:
{context}

Question: {query}

Answer:"""
        else:
            prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        return prompt

    def generate(
        self,
        query: str,
        context_chunks: List[Dict],
        max_new_tokens: int = None
    ) -> Dict:
        """
        Generate answer for query with context

        Args:
            query: User query
            context_chunks: Retrieved context chunks
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with answer and metadata
        """
        # Create prompt
        prompt = self.create_prompt(query, context_chunks)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        # Count input tokens
        input_token_count = inputs['input_ids'].shape[1]

        # Generate
        with torch.no_grad():
            if max_new_tokens is None:
                # Calculate available tokens, ensuring minimum of 64 to prevent errors
                # when input is close to or exceeds max_length
                available_tokens = self.max_length - input_token_count
                max_new_tokens = max(64, min(256, available_tokens))

            # Ensure max_new_tokens is always positive
            max_new_tokens = max(1, max_new_tokens)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        if self.model_type == "seq2seq":
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # For causal models, extract only the generated part
            answer = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

        return {
            'query': query,
            'answer': answer.strip(),
            'context_chunks': context_chunks,
            'num_chunks': len(context_chunks),
            'input_tokens': input_token_count,
            'model': self.model_name
        }

    def batch_generate(
        self,
        queries: List[str],
        context_chunks_list: List[List[Dict]],
        max_new_tokens: int = None
    ) -> List[Dict]:
        """
        Generate answers for multiple queries

        Args:
            queries: List of queries
            context_chunks_list: List of context chunks for each query
            max_new_tokens: Maximum tokens to generate

        Returns:
            List of generation results
        """
        logger.info(f"Generating answers for {len(queries)} queries...")

        results = []
        for query, context_chunks in zip(queries, context_chunks_list):
            result = self.generate(query, context_chunks, max_new_tokens)
            results.append(result)

        return results


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation"""

    def __init__(self, hybrid_retriever, llm_generator):
        """
        Initialize RAG pipeline

        Args:
            hybrid_retriever: HybridRetriever instance
            llm_generator: LLMGenerator instance
        """
        self.retriever = hybrid_retriever
        self.generator = llm_generator

    def query(
        self,
        question: str,
        top_k: int = 10,
        top_n: int = 5,
        max_new_tokens: int = None
    ) -> Dict:
        """
        Complete RAG query pipeline

        Args:
            question: User question
            top_k: Number of chunks to retrieve per method
            top_n: Number of chunks to use for generation
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with answer and retrieval details
        """
        # Retrieve relevant chunks
        retrieval_results = self.retriever.retrieve(
            question,
            top_k=top_k,
            top_n=top_n
        )

        # Generate answer
        generation_result = self.generator.generate(
            question,
            retrieval_results['fused_results'],
            max_new_tokens=max_new_tokens
        )

        # Combine results
        return {
            'question': question,
            'answer': generation_result['answer'],
            'retrieval_results': retrieval_results,
            'generation_metadata': {
                'input_tokens': generation_result['input_tokens'],
                'model': generation_result['model']
            }
        }

    def batch_query(
        self,
        questions: List[str],
        top_k: int = 10,
        top_n: int = 5,
        max_new_tokens: int = None
    ) -> List[Dict]:
        """
        Batch RAG queries

        Args:
            questions: List of questions
            top_k: Number of chunks to retrieve per method
            top_n: Number of chunks to use for generation
            max_new_tokens: Maximum tokens to generate

        Returns:
            List of query results
        """
        logger.info(f"Processing {len(questions)} questions through RAG pipeline...")

        results = []
        for question in questions:
            result = self.query(question, top_k, top_n, max_new_tokens)
            results.append(result)

        return results


if __name__ == "__main__":
    # Example usage
    from dense_retrieval import DenseRetriever
    from sparse_retrieval import BM25Retriever
    from rrf_fusion import HybridRetriever

    # Load retrievers
    dense_retriever = DenseRetriever()
    dense_retriever.load_index("data/vector_index")

    sparse_retriever = BM25Retriever()
    sparse_retriever.load_index("data/bm25_index.pkl")

    hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)

    # Initialize generator
    generator = LLMGenerator(model_name="google/flan-t5-base")

    # Create RAG pipeline
    rag_pipeline = RAGPipeline(hybrid_retriever, generator)

    # Test query
    result = rag_pipeline.query("What is machine learning?")

    print("\nRAG Pipeline Result:")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"\nNumber of chunks used: {len(result['retrieval_results']['fused_results'])}")
