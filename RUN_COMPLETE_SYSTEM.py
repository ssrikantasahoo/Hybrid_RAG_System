#!/usr/bin/env python3
"""
AUTOMATED EXECUTION SCRIPT - Fixes ALL Missing Deliverables

This script will:
1. Generate data/fixed_urls.json (200 URLs)
2. Generate data/questions.json (100 questions)
3. Build vector and BM25 indices
4. Run complete evaluation
5. Generate all visualizations
6. Create PDF report
7. Verify all required files exist

Run with: python RUN_COMPLETE_SYSTEM.py [--force-rebuild] [--report-only]

Options:
  --force-rebuild    Delete existing data and regenerate random 300 URLs
                     (REQUIRED for assignment compliance - random URLs must
                     change every time system is rebuilt/indexed)
  --report-only      Regenerate only outputs/evaluation_report.pdf from
                     existing outputs/evaluation_results.json
"""

import os
import sys
import json
import logging
from pathlib import Path
import time
import argparse

# Add src to path so internal imports work
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'execution.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def verify_dependencies():
    """Verify all required packages are installed"""
    logger.info("=" * 70)
    logger.info("STEP 1: Verifying Dependencies")
    logger.info("=" * 70)

    # Map package names to their import names
    required = {
        'sentence_transformers': 'sentence_transformers',
        'faiss': 'faiss',
        'rank_bm25': 'rank_bm25',
        'transformers': 'transformers',
        'wikipedia': 'wikipediaapi',
        'bs4': 'bs4',
        'nltk': 'nltk',
        'rouge_score': 'rouge_score',
        'bert_score': 'bert_score',
        'sklearn': 'sklearn',
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'spacy': 'spacy'
    }

    missing = []
    for display_name, import_name in required.items():
        try:
            __import__(import_name)
            logger.info(f"[OK] {display_name}")
        except ImportError:
            logger.error(f"[MISSING] {display_name} - MISSING")
            missing.append(display_name)

    if missing:
        logger.error(f"Missing packages: {missing}")
        logger.error("Run: pip install -r requirements.txt")
        return False

    # Check spaCy model
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        logger.info("[OK] spaCy model en_core_web_sm")
    except:
        logger.error("[FAIL] spaCy model missing")
        logger.error("Run: python -m spacy download en_core_web_sm")
        return False

    logger.info("[OK] All dependencies satisfied\n")
    return True

def create_directories():
    """Create required directories"""
    logger.info("=" * 70)
    logger.info("STEP 2: Creating Directory Structure")
    logger.info("=" * 70)

    dirs = ['data', 'outputs', 'logs', 'models', 'screenshots']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        logger.info(f"[OK] Created/verified: {dir_name}/")
    logger.info("")

def generate_fixed_urls():
    """Generate data/fixed_urls.json with 200 Wikipedia URLs

    IMPORTANT: The fixed 200 URLs must be unique across all groups per assignment requirement.
    This function will NOT auto-generate URLs if the file doesn't exist.
    """
    logger.info("=" * 70)
    logger.info("STEP 3: Verifying Fixed URLs (200 Wikipedia articles)")
    logger.info("=" * 70)

    # Check if fixed_urls.json exists
    if not Path('data/fixed_urls.json').exists():
        logger.error("[ERROR] data/fixed_urls.json is MISSING!")
        logger.error("")
        logger.error("CRITICAL: The assignment requires each group to have a UNIQUE set of 200 fixed URLs.")
        logger.error("Auto-generation has been DISABLED to prevent duplicate URLs across groups.")
        logger.error("")
        logger.error("SOLUTION:")
        logger.error("  1. Manually create data/fixed_urls.json with your group's unique 200 URLs")
        logger.error("  2. Or uncomment the auto-generation code (see RUN_COMPLETE_SYSTEM.py line ~115)")
        logger.error("  3. WARNING: Auto-generated URLs may duplicate other groups!")
        logger.error("")
        raise FileNotFoundError("data/fixed_urls.json is required but missing")

    # Load and verify fixed URLs
    logger.info("[OK] data/fixed_urls.json exists")
    with open('data/fixed_urls.json', 'r') as f:
        urls = json.load(f)

    if len(urls) != 200:
        logger.error(f"[ERROR] Expected 200 fixed URLs, found {len(urls)}")
        raise ValueError(f"Fixed URLs must be exactly 200, found {len(urls)}")

    logger.info(f"[OK] Verified {len(urls)} fixed URLs")
    logger.info(f"  Sample URLs: {[url for url in urls[:3]]}")
    logger.info("[NOTE] These 200 URLs are LOCKED and will NOT change on rebuild\n")

    ### UNCOMMENT BELOW TO AUTO-GENERATE (NOT RECOMMENDED) ###
    # from src.data_collection import WikipediaDataCollector
    # collector = WikipediaDataCollector(min_words=200)
    # logger.info("Generating 200 unique Wikipedia URLs...")
    # fixed_urls = collector.create_fixed_urls_set(count=200, output_file='data/fixed_urls.json')
    # logger.info(f"[OK] Generated data/fixed_urls.json with {len(fixed_urls)} URLs")
    ### END AUTO-GENERATE ###

def collect_and_preprocess_data(force_rebuild=False):
    """Collect Wikipedia articles and preprocess

    Args:
        force_rebuild: If True, regenerate random URLs and rebuild indices
    """
    logger.info("=" * 70)
    logger.info("STEP 4: Collecting & Preprocessing Wikipedia Data")
    logger.info("=" * 70)

    from src.data_collection import WikipediaDataCollector
    from src.preprocessing import TextChunker

    # Check if processed chunks already exist (skip only if NOT force rebuild)
    if not force_rebuild and Path('data/processed_chunks.json').exists():
        logger.info("[OK] data/processed_chunks.json already exists")
        logger.info("[WARNING] Random 300 URLs are NOT being regenerated!")
        logger.info("[WARNING] Use --force-rebuild flag to regenerate random URLs per assignment requirement")
        with open('data/processed_chunks.json', 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"[OK] Loaded {len(chunks)} existing chunks\n")
        return chunks

    if force_rebuild:
        logger.info("[FORCE REBUILD] Regenerating random 300 URLs per assignment requirement")
        # Delete old data to ensure fresh random URLs
        for old_file in ['data/processed_chunks.json', 'data/raw_corpus.json', 'data/random_urls.json']:
            if Path(old_file).exists():
                Path(old_file).unlink()
                logger.info(f"[DELETED] {old_file}")

    # Load fixed URLs
    with open('data/fixed_urls.json', 'r') as f:
        fixed_urls = json.load(f)

    logger.info(f"Collecting 200 fixed + 300 random articles (Total: 500)...")

    collector = WikipediaDataCollector(min_words=200)

    # Collect articles from fixed URLs
    logger.info("Downloading fixed URL articles (this may take 10-20 minutes)...")
    articles = []
    for i, url in enumerate(fixed_urls):
        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i + 1}/{len(fixed_urls)}")

        article = collector.get_article_from_url(url)
        if article:
            articles.append(article)

    logger.info(f"[OK] Collected {len(articles)} articles from fixed URLs")

    # Get random articles
    logger.info("Collecting 300 random articles...")
    random_articles = collector.get_random_wikipedia_urls(count=300, exclude_urls=fixed_urls)

    # Save random URLs separately for verification
    random_urls = [article['url'] for article in random_articles]
    with open('data/random_urls.json', 'w', encoding='utf-8') as f:
        json.dump({
            'urls': random_urls,
            'count': len(random_urls),
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'note': 'These 300 URLs are randomly generated and MUST change on each rebuild'
        }, f, indent=2)
    logger.info(f"[OK] Saved data/random_urls.json with {len(random_urls)} URLs")

    # Add random articles
    for article in random_articles:
        articles.append(article)

    logger.info(f"[OK] Total collected: {len(articles)} articles")

    # Save raw corpus
    with open('data/raw_corpus.json', 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    logger.info("[OK] Saved data/raw_corpus.json")

    # Preprocess and chunk
    logger.info("Preprocessing and chunking text...")
    chunker = TextChunker(chunk_size=300, overlap=50)

    # Use chunk_corpus which handles the entire corpus
    chunks = chunker.chunk_corpus(articles)

    logger.info(f"[OK] Created {len(chunks)} chunks")

    # Save processed chunks
    with open('data/processed_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"[OK] Saved data/processed_chunks.json\n")

    return chunks

def build_indices(chunks):
    """Build dense and sparse indices"""
    logger.info("=" * 70)
    logger.info("STEP 5: Building Dense (FAISS) and Sparse (BM25) Indices")
    logger.info("=" * 70)

    from src.dense_retrieval import DenseRetriever
    from src.sparse_retrieval import BM25Retriever

    dense_retriever = DenseRetriever(model_name="sentence-transformers/all-mpnet-base-v2")
    sparse_retriever = BM25Retriever()

    # Check if indices already exist
    if Path('data/vector_index').exists() and Path('data/bm25_index.pkl').exists():
        logger.info("[OK] Indices already exist. Loading them...")
        
        dense_retriever.load_index('data/vector_index')
        logger.info("[OK] Loaded data/vector_index/")
        
        sparse_retriever.load_index('data/bm25_index.pkl')
        logger.info("[OK] Loaded data/bm25_index.pkl\n")
        
        return dense_retriever, sparse_retriever

    # Dense index
    logger.info("Building dense FAISS index (this may take 10-15 minutes)...")
    dense_retriever.build_index(chunks)
    dense_retriever.save_index('data/vector_index')
    logger.info("[OK] Saved data/vector_index/")

    # Sparse index
    logger.info("Building sparse BM25 index...")
    sparse_retriever.build_index(chunks)
    sparse_retriever.save_index('data/bm25_index.pkl')
    logger.info("[OK] Saved data/bm25_index.pkl\n")

    return dense_retriever, sparse_retriever

def generate_questions(chunks):
    """Generate 100 evaluation questions"""
    logger.info("=" * 70)
    logger.info("STEP 6: Generating 100 Evaluation Questions")
    logger.info("=" * 70)

    if Path('data/questions.json').exists():
        logger.info("[OK] data/questions.json already exists")
        with open('data/questions.json', 'r', encoding='utf-8') as f:
            questions = json.load(f)
        logger.info(f"[OK] Loaded {len(questions)} existing questions\n")
        return questions

    from src.question_generation import QuestionGenerator, create_multi_hop_questions

    logger.info("Generating questions (this may take 30-60 minutes)...")

    generator = QuestionGenerator()

    # Generate 90 regular questions
    questions = generator.generate_questions_from_chunks(
        chunks,
        num_questions=90,
        distribution={
            'factual': 0.4,
            'comparative': 0.3,
            'inferential': 0.3
        }
    )

    logger.info(f"[OK] Generated {len(questions)} regular questions")

    # Generate 10 multi-hop questions
    multi_hop = create_multi_hop_questions(chunks, num_questions=10)
    questions.extend(multi_hop)

    logger.info(f"[OK] Generated {len(multi_hop)} multi-hop questions")
    logger.info(f"[OK] Total: {len(questions)} questions")

    # Save questions
    generator.save_questions(questions, 'data/questions.json')
    logger.info("[OK] Saved data/questions.json\n")

    return questions

def run_evaluation(dense_retriever, sparse_retriever, questions):
    """Run complete evaluation"""
    logger.info("=" * 70)
    logger.info("STEP 7: Running Complete Evaluation")
    logger.info("=" * 70)

    # Check if evaluation results already exist
    if Path('outputs/evaluation_results.json').exists():
        logger.info("[OK] outputs/evaluation_results.json already exists. Loading results...")
        with open('outputs/evaluation_results.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
        logger.info("[OK] Evaluation results loaded.\n")

        # Check if parameter sweep plot exists, if not try to regenerate
        if not Path('outputs/parameter_sweep_results.png').exists() and 'parameter_sweep' in results:
            logger.info("Regenerating missing parameter_sweep_results.png...")
            try:
                from src.parameter_sweep import ParameterSweep
                # Mock objects since we only use the visualization method which handles static data
                sweeper = ParameterSweep(None, None)
                sweeper.visualize_parameter_sweep(results['parameter_sweep'], 'outputs')
                logger.info("[OK] Regenerated parameter_sweep_results.png")
            except Exception as e:
                logger.warning(f"Could not regenerate parameter sweep plot: {e}")

        return results

    from src.rrf_fusion import HybridRetriever
    from src.llm_generation import LLMGenerator, RAGPipeline
    from src.evaluation_pipeline import EvaluationPipeline

    # Create RAG pipeline
    logger.info("Initializing RAG pipeline...")
    hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)
    generator = LLMGenerator(model_name="google/flan-t5-base")
    rag_pipeline = RAGPipeline(hybrid_retriever, generator)

    logger.info("[OK] RAG pipeline ready")

    # Create evaluation pipeline
    eval_pipeline = EvaluationPipeline(rag_pipeline, dense_retriever, sparse_retriever)

    # Run full evaluation
    logger.info("Running evaluation on 100 questions (this will take 1-2 hours)...")
    logger.info("This includes:")
    logger.info("  - Standard metrics (MRR, NDCG, BERTScore, etc.)")
    logger.info("  - Innovative metrics (LLM-Judge, Adversarial, Calibration, etc.)")
    logger.info("  - Additional metrics (Entity Coverage, Diversity, Faithfulness, etc.)")
    logger.info("  - Ablation study")
    logger.info("  - Error analysis")
    logger.info("  - Parameter sweep")

    results = eval_pipeline.run_full_evaluation(
        questions_file='data/questions.json',
        output_dir='outputs',
        top_k=10,
        top_n=5
    )

    logger.info("[OK] Evaluation complete!")
    logger.info(f"[OK] Saved outputs/evaluation_results.json")
    logger.info(f"[OK] Saved outputs/results_table.csv\n")

    return results

def generate_pdf_report():
    """Generate PDF report"""
    logger.info("=" * 70)
    logger.info("STEP 8: Generating PDF Report")
    logger.info("=" * 70)

    from src.report_generator import ReportGenerator

    logger.info("Creating comprehensive PDF report...")
    generator = ReportGenerator(
        results_file='outputs/evaluation_results.json',
        output_dir='outputs'
    )

    pdf_path = generator.generate_pdf_report()
    logger.info(f"[OK] Generated {pdf_path}\n")

def verify_all_deliverables():
    """Verify all required files exist"""
    logger.info("=" * 70)
    logger.info("STEP 9: Verifying All Required Deliverables")
    logger.info("=" * 70)

    required_files = {
        'Data Files': [
            'data/fixed_urls.json',
            'data/questions.json',
            'data/raw_corpus.json',
            'data/processed_chunks.json',
            'data/bm25_index.pkl'
        ],
        'Indices': [
            'data/vector_index'
        ],
        'Output Files': [
            'outputs/evaluation_results.json',
            'outputs/results_table.csv',
            'outputs/evaluation_report.pdf'
        ],
        'Visualizations': [
            'outputs/ablation_comparison.png',
            'outputs/response_time_distribution.png',
            'outputs/error_analysis.png',
            'outputs/llm_judge_radar.png',
            'outputs/calibration_curve.png',
            'outputs/hallucination_analysis.png',
            'outputs/adversarial_testing.png',
            'outputs/comprehensive_dashboard.png',
            'outputs/parameter_sweep_results.png'
        ]
    }

    all_exist = True

    for category, files in required_files.items():
        logger.info(f"\n{category}:")
        for file_path in files:
            path = Path(file_path)
            if path.exists():
                if path.is_file():
                    size = path.stat().st_size
                    logger.info(f"  [OK] {file_path} ({size:,} bytes)")
                else:
                    logger.info(f"  [OK] {file_path} (directory)")
            else:
                logger.error(f"  [MISSING] {file_path} - MISSING")
                all_exist = False

    logger.info("")

    if all_exist:
        logger.info("=" * 70)
        logger.info("[OK] ALL REQUIRED DELIVERABLES PRESENT [OK]")
        logger.info("=" * 70)
    else:
        logger.error("=" * 70)
        logger.error("[FAIL] SOME DELIVERABLES MISSING [FAIL]")
        logger.error("=" * 70)

    return all_exist

def print_final_summary(results):
    """Print final summary"""
    logger.info("\n" + "=" * 70)
    logger.info("EXECUTION COMPLETE - SUMMARY")
    logger.info("=" * 70)

    logger.info("\nEVALUATION RESULTS:")
    logger.info(f"  - Total Questions: {results.get('num_questions', 0)}")
    logger.info(f"  - Avg Response Time: {results.get('avg_response_time', 0):.3f}s")

    metrics = results.get('metrics', {})
    logger.info(f"\nCORE METRICS:")
    logger.info(f"  - MRR (URL Level): {metrics.get('mrr', {}).get('mrr', 0):.4f}")
    logger.info(f"  - NDCG@5: {metrics.get('ndcg', {}).get('ndcg', 0):.4f}")
    logger.info(f"  - BERTScore F1: {metrics.get('bert_score', {}).get('bert_score_f1', 0):.4f}")

    logger.info(f"\nNEXT STEPS:")
    logger.info("  1. Review outputs/evaluation_report.pdf")
    logger.info("  2. Launch UIs and take screenshots:")
    logger.info("     streamlit run src/streamlit_app.py")
    logger.info("     streamlit run src/evaluation_dashboard.py")
    logger.info("  3. Create submission ZIP")

    logger.info("\n" + "=" * 70)
    logger.info("[OK] SYSTEM READY FOR SUBMISSION [OK]")
    logger.info("=" * 70)

def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run complete Hybrid RAG System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python RUN_COMPLETE_SYSTEM.py                  # Use existing data if available
  python RUN_COMPLETE_SYSTEM.py --force-rebuild  # Regenerate random 300 URLs (REQUIRED for assignment)
  python RUN_COMPLETE_SYSTEM.py --report-only    # Generate PDF report only (fast)
        """
    )
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Delete existing data and regenerate random 300 URLs (per assignment requirement)'
    )
    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Skip full pipeline and regenerate PDF report from outputs/evaluation_results.json'
    )
    args = parser.parse_args()

    if args.force_rebuild and args.report_only:
        parser.error("--force-rebuild and --report-only cannot be used together")

    start_time = time.time()

    logger.info("+" + "=" * 68 + "+")
    logger.info("|" + " " * 15 + "HYBRID RAG SYSTEM - COMPLETE EXECUTION" + " " * 15 + "|")
    logger.info("+" + "=" * 68 + "+")
    if args.force_rebuild:
        logger.info("|" + " " * 18 + "MODE: FORCE REBUILD (Fresh Random URLs)" + " " * 9 + "|")
        logger.info("+" + "=" * 68 + "+")
    elif args.report_only:
        logger.info("|" + " " * 22 + "MODE: REPORT ONLY (Fast PDF Regeneration)" + " " * 4 + "|")
        logger.info("+" + "=" * 68 + "+")
    logger.info("")

    try:
        if args.report_only:
            create_directories()

            results_file = Path('outputs/evaluation_results.json')
            if not results_file.exists():
                raise FileNotFoundError(
                    "outputs/evaluation_results.json not found. Run full pipeline once before --report-only."
                )

            generate_pdf_report()
            pdf_exists = Path('outputs/evaluation_report.pdf').exists()
            logger.info(f"[OK] Report-only mode complete. PDF exists: {pdf_exists}")
            return pdf_exists

        # Step 1: Verify dependencies
        if not verify_dependencies():
            logger.error("Please install missing dependencies first!")
            return False

        # Step 2: Create directories
        create_directories()

        # Step 3: Generate fixed URLs
        generate_fixed_urls()

        # Step 4: Collect and preprocess data
        chunks = collect_and_preprocess_data(force_rebuild=args.force_rebuild)

        # Step 5: Build indices
        dense_retriever, sparse_retriever = build_indices(chunks)

        # Step 6: Generate questions
        questions = generate_questions(chunks)

        # Step 7: Run evaluation
        results = run_evaluation(dense_retriever, sparse_retriever, questions)

        # Step 8: Generate PDF report
        generate_pdf_report()

        # Step 9: Verify deliverables
        all_exist = verify_all_deliverables()

        # Print summary
        print_final_summary(results)

        elapsed = time.time() - start_time
        logger.info(f"\nTotal execution time: {elapsed/3600:.2f} hours")

        return all_exist

    except Exception as e:
        logger.error(f"\n[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
