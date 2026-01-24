"""
Build System Script
Complete pipeline to build the Hybrid RAG System from scratch
"""

import sys
import argparse
import logging
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_collection import WikipediaDataCollector
from preprocessing import TextChunker
from dense_retrieval import DenseRetriever
from sparse_retrieval import BM25Retriever
from rrf_fusion import HybridRetriever
from llm_generation import LLMGenerator, RAGPipeline
from question_generation import QuestionGenerator, create_multi_hop_questions
from evaluation_pipeline import EvaluationPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_corpus(config):
    """Step 1: Build corpus from Wikipedia"""
    logger.info("=" * 60)
    logger.info("STEP 1: Building Corpus from Wikipedia")
    logger.info("=" * 60)

    collector = WikipediaDataCollector(min_words=config['dataset']['min_article_words'])

    # Collect corpus
    corpus = collector.collect_corpus(
        fixed_urls_file=config['dataset']['fixed_urls_file'],
        random_count=config['dataset']['random_urls_count']
    )

    # Save raw corpus
    corpus_file = Path(config['paths']['data_dir']) / "raw_corpus.json"
    collector.save_corpus(corpus, str(corpus_file))

    logger.info(f"✅ Corpus built with {len(corpus)} articles")

    return corpus


def preprocess_corpus(config, corpus):
    """Step 2: Preprocess and chunk corpus"""
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing and Chunking Corpus")
    logger.info("=" * 60)

    chunker = TextChunker(
        chunk_size=config['chunking']['chunk_size'],
        overlap=config['chunking']['overlap']
    )

    # Chunk corpus
    chunks = chunker.chunk_corpus(corpus)

    # Save chunks
    chunks_file = Path(config['paths']['data_dir']) / "processed_chunks.json"
    chunker.save_chunks(chunks, str(chunks_file))

    logger.info(f"✅ Created {len(chunks)} chunks from {len(corpus)} articles")

    return chunks


def build_indices(config, chunks):
    """Step 3: Build dense and sparse indices"""
    logger.info("=" * 60)
    logger.info("STEP 3: Building Retrieval Indices")
    logger.info("=" * 60)

    # Build dense index
    logger.info("Building dense vector index (FAISS)...")
    dense_retriever = DenseRetriever(model_name=config['dense_retrieval']['model_name'])
    dense_retriever.build_index(chunks, save_dir=config['paths']['vector_index_dir'])

    # Build sparse index
    logger.info("Building sparse index (BM25)...")
    sparse_retriever = BM25Retriever(
        k1=config['sparse_retrieval']['k1'],
        b=config['sparse_retrieval']['b']
    )
    sparse_retriever.build_index(chunks, save_path=config['paths']['bm25_index_file'])

    logger.info("✅ Retrieval indices built successfully")

    return dense_retriever, sparse_retriever


def generate_questions(config, chunks):
    """Step 4: Generate evaluation questions"""
    logger.info("=" * 60)
    logger.info("STEP 4: Generating Evaluation Questions")
    logger.info("=" * 60)

    generator = QuestionGenerator(model_name=config['question_generation']['qg_model'])

    # Generate regular questions
    num_regular = config['question_generation']['num_questions'] - 10
    questions = generator.generate_questions_from_chunks(chunks, num_questions=num_regular)

    # Generate multi-hop questions
    multi_hop = create_multi_hop_questions(chunks, num_questions=10)
    questions.extend(multi_hop)

    # Save questions
    questions_file = config['paths']['questions_file']
    generator.save_questions(questions, questions_file)

    logger.info(f"✅ Generated {len(questions)} evaluation questions")

    return questions


def run_evaluation(config, dense_retriever, sparse_retriever):
    """Step 5: Run evaluation pipeline"""
    logger.info("=" * 60)
    logger.info("STEP 5: Running Evaluation Pipeline")
    logger.info("=" * 60)

    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(
        dense_retriever,
        sparse_retriever,
        rrf_k=config['rrf']['k_constant']
    )

    # Create LLM generator
    llm_generator = LLMGenerator(
        model_name=config['llm']['model_name'],
        max_length=config['llm']['max_length'],
        temperature=config['llm']['temperature'],
        top_p=config['llm']['top_p']
    )

    # Create RAG pipeline
    rag_pipeline = RAGPipeline(hybrid_retriever, llm_generator)

    # Create evaluation pipeline
    eval_pipeline = EvaluationPipeline(rag_pipeline, dense_retriever, sparse_retriever)

    # Run full evaluation
    results = eval_pipeline.run_full_evaluation(
        questions_file=config['paths']['questions_file'],
        output_dir=config['paths']['output_dir'],
        top_k=config['dense_retrieval']['top_k'],
        top_n=config['rrf']['top_n_chunks']
    )

    logger.info("✅ Evaluation complete")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nNumber of Questions: {results['num_questions']}")
    print(f"Average Response Time: {results['avg_response_time']:.3f}s")
    print(f"\n--- Mandatory Metric ---")
    print(f"MRR (URL Level): {results['metrics']['mrr']['mrr']:.4f}")
    print(f"{results['metrics']['mrr']['interpretation']}")
    print(f"\n--- Custom Metric 1: NDCG ---")
    print(f"NDCG@{results['metrics']['ndcg']['k']}: {results['metrics']['ndcg']['ndcg']:.4f}")
    print(f"{results['metrics']['ndcg']['interpretation']}")
    print(f"\n--- Custom Metric 2: BERTScore ---")
    print(f"BERTScore F1: {results['metrics']['bert_score']['bert_score_f1']:.4f}")
    print(f"{results['metrics']['bert_score']['interpretation']}")
    print(f"\n--- Ablation Study ---")
    print(f"Dense Only:   MRR={results['ablation_study']['dense_only']['mrr']:.4f}, "
          f"NDCG={results['ablation_study']['dense_only']['ndcg']:.4f}")
    print(f"Sparse Only:  MRR={results['ablation_study']['sparse_only']['mrr']:.4f}, "
          f"NDCG={results['ablation_study']['sparse_only']['ndcg']:.4f}")
    print(f"Hybrid (RRF): MRR={results['ablation_study']['hybrid_rrf']['mrr']:.4f}, "
          f"NDCG={results['ablation_study']['hybrid_rrf']['ndcg']:.4f}")
    print("=" * 60)

    return results


def main():
    """Main build pipeline"""
    parser = argparse.ArgumentParser(description='Build Hybrid RAG System')
    parser.add_argument(
        '--skip-data',
        action='store_true',
        help='Skip data collection (use existing corpus)'
    )
    parser.add_argument(
        '--skip-questions',
        action='store_true',
        help='Skip question generation (use existing questions)'
    )
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create directories
    for path_key in ['data_dir', 'output_dir', 'logs_dir', 'models_dir']:
        Path(config['paths'][path_key]).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("HYBRID RAG SYSTEM - BUILD PIPELINE")
    print("=" * 60 + "\n")

    # Step 1: Build corpus
    if not args.skip_data:
        corpus = build_corpus(config)
    else:
        logger.info("Skipping data collection, loading existing corpus...")
        from data_collection import WikipediaDataCollector
        collector = WikipediaDataCollector()
        corpus = collector.load_corpus("data/raw_corpus.json")

    # Step 2: Preprocess corpus
    chunks = preprocess_corpus(config, corpus)

    # Step 3: Build indices
    dense_retriever, sparse_retriever = build_indices(config, chunks)

    # Step 4: Generate questions
    if not args.skip_questions:
        questions = generate_questions(config, chunks)
    else:
        logger.info("Skipping question generation, using existing questions...")

    # Step 5: Run evaluation
    if not args.skip_evaluation:
        results = run_evaluation(config, dense_retriever, sparse_retriever)
    else:
        logger.info("Skipping evaluation")

    print("\n" + "=" * 60)
    print("✅ BUILD COMPLETE!")
    print("=" * 60)
    print(f"\nOutputs saved to: {config['paths']['output_dir']}")
    print("\nTo run the UI:")
    print("  streamlit run src/streamlit_app.py")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
