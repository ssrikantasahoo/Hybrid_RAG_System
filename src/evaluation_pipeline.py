"""
Automated Evaluation Pipeline
Complete pipeline for evaluating RAG system with multiple metrics
"""

import json
import time
from typing import List, Dict
from pathlib import Path
import logging
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from evaluation_metrics import EvaluationMetrics
from dense_retrieval import DenseRetriever
from sparse_retrieval import BM25Retriever
from rrf_fusion import HybridRetriever
from llm_generation import LLMGenerator, RAGPipeline
from innovative_evaluation import run_innovative_evaluation
from advanced_visualizations import AdvancedVisualizer
from additional_metrics import compute_all_additional_metrics
from parameter_sweep import ParameterSweep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Complete evaluation pipeline for RAG system"""

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        dense_retriever: DenseRetriever,
        sparse_retriever: BM25Retriever
    ):
        """
        Initialize evaluation pipeline

        Args:
            rag_pipeline: RAG pipeline to evaluate
            dense_retriever: Dense retriever for ablation studies
            sparse_retriever: Sparse retriever for ablation studies
        """
        self.rag_pipeline = rag_pipeline
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.metrics = EvaluationMetrics()

    def evaluate_questions(
        self,
        questions: List[Dict],
        top_k: int = 10,
        top_n: int = 5
    ) -> Dict:
        """
        Evaluate RAG system on a set of questions

        Args:
            questions: List of question dictionaries
            top_k: Top-K for retrieval
            top_n: Top-N for generation

        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating {len(questions)} questions...")

        retrieval_results = []
        generated_answers = []
        response_times = []
        errors = []

        for i, question in enumerate(tqdm(questions, desc="Evaluating")):
            try:
                start_time = time.time()

                # Run RAG pipeline
                result = self.rag_pipeline.query(
                    question['question'],
                    top_k=top_k,
                    top_n=top_n
                )

                end_time = time.time()

                retrieval_results.append(result['retrieval_results'])
                generated_answers.append(result['answer'])
                response_times.append(end_time - start_time)

            except Exception as e:
                logger.error(f"Error evaluating question {question.get('question_id', i)}: {e}")
                errors.append({
                    'question_id': question.get('question_id', i),
                    'error': str(e)
                })
                # Add placeholder results
                retrieval_results.append({'fused_results': []})
                generated_answers.append("")
                response_times.append(0.0)

        # Calculate metrics
        logger.info("Calculating evaluation metrics...")

        # Mandatory metric: MRR at URL level
        mrr_results = self.metrics.mean_reciprocal_rank_url(questions, retrieval_results)

        # Custom metric 1: NDCG@K
        ndcg_results = self.metrics.normalized_discounted_cumulative_gain(
            questions, retrieval_results, k=top_n
        )

        # Custom metric 2: BERTScore
        bert_results = self.metrics.bert_score_semantic_similarity(
            questions, generated_answers
        )

        # Additional metrics
        precision_results = self.metrics.precision_at_k(questions, retrieval_results, k=top_n)
        recall_results = self.metrics.recall_at_k(questions, retrieval_results, k=top_n)
        rouge_results = self.metrics.rouge_scores(questions, generated_answers)
        em_results = self.metrics.exact_match(questions, generated_answers)

        # Compile results
        results = {
            'num_questions': len(questions),
            'num_errors': len(errors),
            'errors': errors,
            'avg_response_time': np.mean(response_times),
            'response_times': response_times,
            'metrics': {
                'mrr': mrr_results,
                'ndcg': ndcg_results,
                'bert_score': bert_results,
                'precision_at_k': precision_results,
                'recall_at_k': recall_results,
                'rouge': rouge_results,
                'exact_match': em_results
            },
            'retrieval_results': retrieval_results,
            'generated_answers': generated_answers
        }

        return results

    def ablation_study(
        self,
        questions: List[Dict],
        top_k: int = 10
    ) -> Dict:
        """
        Perform ablation study comparing different retrieval methods

        Args:
            questions: List of questions
            top_k: Top-K for retrieval

        Returns:
            Ablation study results
        """
        logger.info("Performing ablation study...")

        # Test 1: Dense-only
        logger.info("Testing Dense-only retrieval...")
        dense_results = []
        for question in tqdm(questions, desc="Dense-only"):
            try:
                results = self.dense_retriever.search(question['question'], top_k=top_k)
                dense_results.append({'results': results})
            except:
                dense_results.append({'results': []})

        dense_mrr = self.metrics.mean_reciprocal_rank_url(questions, dense_results)
        dense_ndcg = self.metrics.normalized_discounted_cumulative_gain(questions, dense_results, k=top_k)

        # Test 2: Sparse-only
        logger.info("Testing Sparse-only retrieval...")
        sparse_results = []
        for question in tqdm(questions, desc="Sparse-only"):
            try:
                results = self.sparse_retriever.search(question['question'], top_k=top_k)
                sparse_results.append({'results': results})
            except:
                sparse_results.append({'results': []})

        sparse_mrr = self.metrics.mean_reciprocal_rank_url(questions, sparse_results)
        sparse_ndcg = self.metrics.normalized_discounted_cumulative_gain(questions, sparse_results, k=top_k)

        # Test 3: Hybrid (RRF)
        logger.info("Testing Hybrid (RRF) retrieval...")
        hybrid_results = []
        for question in tqdm(questions, desc="Hybrid"):
            try:
                results = self.rag_pipeline.retriever.retrieve(
                    question['question'],
                    top_k=top_k,
                    top_n=top_k
                )
                hybrid_results.append(results)
            except:
                hybrid_results.append({'fused_results': []})

        hybrid_mrr = self.metrics.mean_reciprocal_rank_url(questions, hybrid_results)
        hybrid_ndcg = self.metrics.normalized_discounted_cumulative_gain(questions, hybrid_results, k=top_k)

        ablation_results = {
            'dense_only': {
                'mrr': dense_mrr['mrr'],
                'ndcg': dense_ndcg['ndcg']
            },
            'sparse_only': {
                'mrr': sparse_mrr['mrr'],
                'ndcg': sparse_ndcg['ndcg']
            },
            'hybrid_rrf': {
                'mrr': hybrid_mrr['mrr'],
                'ndcg': hybrid_ndcg['ndcg']
            }
        }

        logger.info("\nAblation Study Results:")
        logger.info(f"Dense-only:  MRR={dense_mrr['mrr']:.4f}, NDCG@{top_k}={dense_ndcg['ndcg']:.4f}")
        logger.info(f"Sparse-only: MRR={sparse_mrr['mrr']:.4f}, NDCG@{top_k}={sparse_ndcg['ndcg']:.4f}")
        logger.info(f"Hybrid (RRF): MRR={hybrid_mrr['mrr']:.4f}, NDCG@{top_k}={hybrid_ndcg['ndcg']:.4f}")

        return ablation_results

    def error_analysis(
        self,
        questions: List[Dict],
        retrieval_results: List[Dict],
        generated_answers: List[str]
    ) -> Dict:
        """
        Perform error analysis

        Args:
            questions: List of questions
            retrieval_results: Retrieval results
            generated_answers: Generated answers

        Returns:
            Error analysis results
        """
        logger.info("Performing error analysis...")

        errors_by_type = {
            'factual': {'retrieval_failures': 0, 'generation_failures': 0, 'total': 0},
            'comparative': {'retrieval_failures': 0, 'generation_failures': 0, 'total': 0},
            'inferential': {'retrieval_failures': 0, 'generation_failures': 0, 'total': 0},
            'multi-hop': {'retrieval_failures': 0, 'generation_failures': 0, 'total': 0}
        }

        failure_examples = []

        for question, retrieval_result, generated in zip(questions, retrieval_results, generated_answers):
            q_type = question.get('question_type', 'unknown')

            if q_type in errors_by_type:
                errors_by_type[q_type]['total'] += 1

            # Check for retrieval failure
            ground_truth_url = question['source_url']
            retrieved_chunks = retrieval_result.get('fused_results', retrieval_result.get('results', []))

            retrieval_failed = not any(
                chunk['chunk']['url'] == ground_truth_url
                for chunk in retrieved_chunks
            )

            if retrieval_failed and q_type in errors_by_type:
                errors_by_type[q_type]['retrieval_failures'] += 1
                failure_examples.append({
                    'question_id': question.get('question_id', ''),
                    'question_type': q_type,
                    'failure_type': 'retrieval',
                    'question': question['question'],
                    'ground_truth_url': ground_truth_url,
                    'retrieved_urls': [chunk['chunk']['url'] for chunk in retrieved_chunks[:5]]
                })

            # Check for generation failure (empty or very short answer)
            if len(generated.strip()) < 10:
                if q_type in errors_by_type:
                    errors_by_type[q_type]['generation_failures'] += 1
                failure_examples.append({
                    'question_id': question.get('question_id', ''),
                    'question_type': q_type,
                    'failure_type': 'generation',
                    'question': question['question'],
                    'generated_answer': generated,
                    'expected_answer': question['answer']
                })

        return {
            'errors_by_type': errors_by_type,
            'failure_examples': failure_examples[:20]  # Top 20 failures
        }

    def run_full_evaluation(
        self,
        questions_file: str,
        output_dir: str,
        top_k: int = 10,
        top_n: int = 5
    ) -> Dict:
        """
        Run complete evaluation pipeline

        Args:
            questions_file: Path to questions JSON file
            output_dir: Directory to save results
            top_k: Top-K for retrieval
            top_n: Top-N for generation

        Returns:
            Complete evaluation results
        """
        # Load questions
        logger.info(f"Loading questions from {questions_file}")
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        # Main evaluation
        logger.info("Starting main evaluation...")
        eval_results = self.evaluate_questions(questions, top_k, top_n)

        # Ablation study
        logger.info("Starting ablation study...")
        ablation_results = self.ablation_study(questions, top_k)
        eval_results['ablation_study'] = ablation_results

        # Error analysis
        logger.info("Starting error analysis...")
        error_analysis = self.error_analysis(
            questions,
            eval_results['retrieval_results'],
            eval_results['generated_answers']
        )
        eval_results['error_analysis'] = error_analysis

        # INNOVATIVE EVALUATION SUITE
        logger.info("\n" + "="*60)
        logger.info("Starting Innovative Evaluation Suite...")
        logger.info("="*60)
        try:
            innovative_results = run_innovative_evaluation(
                self.rag_pipeline,
                questions,
                eval_results['retrieval_results'],
                eval_results['generated_answers'],
                num_llm_judge_samples=30,  # Sample size for expensive LLM-as-Judge
                num_adversarial_samples=20
            )
            eval_results['innovative_metrics'] = innovative_results
            logger.info("Innovative evaluation completed successfully!")
        except Exception as e:
            logger.error(f"Error in innovative evaluation: {e}")
            logger.info("Continuing with standard evaluation...")
            eval_results['innovative_metrics'] = {}

        # ADDITIONAL NOVEL METRICS
        logger.info("\n" + "="*60)
        logger.info("Computing Additional Novel Metrics...")
        logger.info("="*60)
        try:
            additional_metrics = compute_all_additional_metrics(
                questions,
                eval_results['generated_answers'],
                eval_results['retrieval_results']
            )
            eval_results['additional_metrics'] = additional_metrics
            logger.info("Additional metrics computed successfully!")
        except Exception as e:
            logger.error(f"Error computing additional metrics: {e}")
            eval_results['additional_metrics'] = {}

        # PARAMETER SWEEP (on sample)
        logger.info("\n" + "="*60)
        logger.info("Running Parameter Sweep (Sample)...")
        logger.info("="*60)
        try:
            param_sweeper = ParameterSweep(self.rag_pipeline, self.metrics)
            param_results = param_sweeper.full_parameter_sweep(questions, output_dir=str(output_dir))
            eval_results['parameter_sweep'] = param_results
            logger.info("Parameter sweep completed successfully!")
        except Exception as e:
            logger.error(f"Error in parameter sweep: {e}")
            eval_results['parameter_sweep'] = {}

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        results_file = output_path / "evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            # Convert to serializable format
            serializable_results = self._make_serializable(eval_results)
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Saved evaluation results to {results_file}")

        # Generate standard visualizations
        self.generate_visualizations(eval_results, output_path)

        # Generate ADVANCED visualizations
        logger.info("\nGenerating advanced visualizations...")
        try:
            advanced_viz = AdvancedVisualizer(output_dir=str(output_path))
            advanced_viz.create_all_visualizations(eval_results)
        except Exception as e:
            logger.error(f"Error creating advanced visualizations: {e}")

        # Create results CSV
        self.create_results_csv(questions, eval_results, output_path / "results_table.csv")

        # Print summary of innovative metrics
        self._print_innovative_summary(eval_results)

        return eval_results

    def _make_serializable(self, obj):
        """Convert numpy types to native Python types"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj

    def generate_visualizations(self, results: Dict, output_dir: Path):
        """Generate evaluation visualizations"""
        logger.info("Generating visualizations...")

        # Set style
        sns.set_style("whitegrid")

        # 1. Ablation Study Comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ablation = results['ablation_study']
        methods = ['Dense Only', 'Sparse Only', 'Hybrid (RRF)']
        mrr_values = [
            ablation['dense_only']['mrr'],
            ablation['sparse_only']['mrr'],
            ablation['hybrid_rrf']['mrr']
        ]
        ndcg_values = [
            ablation['dense_only']['ndcg'],
            ablation['sparse_only']['ndcg'],
            ablation['hybrid_rrf']['ndcg']
        ]

        axes[0].bar(methods, mrr_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0].set_ylabel('MRR Score')
        axes[0].set_title('Mean Reciprocal Rank by Method')
        axes[0].set_ylim(0, 1)

        axes[1].bar(methods, ndcg_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1].set_ylabel('NDCG Score')
        axes[1].set_title('NDCG by Method')
        axes[1].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(output_dir / 'ablation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Response Time Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(results['response_times'], bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Response Time Distribution')
        plt.axvline(results['avg_response_time'], color='r', linestyle='--',
                    label=f'Mean: {results["avg_response_time"]:.2f}s')
        plt.legend()
        plt.savefig(output_dir / 'response_time_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Error Analysis by Question Type
        error_data = results['error_analysis']['errors_by_type']

        fig, ax = plt.subplots(figsize=(12, 6))
        question_types = list(error_data.keys())
        retrieval_failures = [error_data[qt]['retrieval_failures'] for qt in question_types]
        generation_failures = [error_data[qt]['generation_failures'] for qt in question_types]

        x = np.arange(len(question_types))
        width = 0.35

        ax.bar(x - width/2, retrieval_failures, width, label='Retrieval Failures', color='#d62728')
        ax.bar(x + width/2, generation_failures, width, label='Generation Failures', color='#ff9896')

        ax.set_xlabel('Question Type')
        ax.set_ylabel('Number of Failures')
        ax.set_title('Errors by Question Type')
        ax.set_xticks(x)
        ax.set_xticklabels(question_types)
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved visualizations to {output_dir}")

    def create_results_csv(self, questions: List[Dict], results: Dict, output_file: Path):
        """Create detailed results CSV"""
        rows = []

        for i, question in enumerate(questions):
            row = {
                'Question ID': question.get('question_id', ''),
                'Question': question['question'],
                'Question Type': question.get('question_type', ''),
                'Ground Truth': question['answer'],
                'Generated Answer': results['generated_answers'][i] if i < len(results['generated_answers']) else '',
                'MRR': results['metrics']['mrr']['details'][i]['reciprocal_rank'] if i < len(results['metrics']['mrr']['details']) else 0,
                'NDCG': results['metrics']['ndcg']['details'][i]['ndcg'] if i < len(results['metrics']['ndcg']['details']) else 0,
                'BERTScore F1': results['metrics']['bert_score']['details'][i]['f1'] if i < len(results['metrics']['bert_score']['details']) else 0,
                'Response Time (s)': results['response_times'][i] if i < len(results['response_times']) else 0
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)

        logger.info(f"Saved results table to {output_file}")

    def _print_innovative_summary(self, results: Dict):
        """Print summary of innovative evaluation metrics"""
        logger.info("\n" + "="*60)
        logger.info("INNOVATIVE EVALUATION SUMMARY")
        logger.info("="*60)

        innovative = results.get('innovative_metrics', {})

        if not innovative:
            logger.info("No innovative metrics available")
            return

        # LLM-as-Judge
        if 'llm_as_judge' in innovative:
            llm_judge = innovative['llm_as_judge']
            logger.info("\n[1] LLM-as-Judge Evaluation:")
            logger.info(f"  Overall Score:      {llm_judge.get('avg_overall_score', 0):.4f}")
            logger.info(f"  Factual Accuracy:   {llm_judge.get('avg_factual_accuracy', 0):.4f}")
            logger.info(f"  Completeness:       {llm_judge.get('avg_completeness', 0):.4f}")
            logger.info(f"  Relevance:          {llm_judge.get('avg_relevance', 0):.4f}")
            logger.info(f"  Coherence:          {llm_judge.get('avg_coherence', 0):.4f}")
            logger.info(f"  Sample Size:        {llm_judge.get('sample_size', 0)}")

        # Adversarial Testing
        if 'adversarial_testing' in innovative:
            adv = innovative['adversarial_testing']
            logger.info("\n[2] Adversarial Testing:")

            if 'paraphrasing_robustness' in adv:
                para = adv['paraphrasing_robustness']
                logger.info(f"  Paraphrase Consistency: {para.get('avg_paraphrase_consistency', 0):.4f}")
                logger.info(f"  {para.get('interpretation', '')}")

            if 'unanswerable_detection' in adv:
                unans = adv['unanswerable_detection']
                logger.info(f"  Hallucination Rate (Unanswerable): {unans.get('hallucination_rate', 0)*100:.1f}%")
                logger.info(f"  {unans.get('interpretation', '')}")

        # Confidence Calibration
        if 'confidence_calibration' in innovative:
            calib = innovative['confidence_calibration']
            logger.info("\n[3] Confidence Calibration:")
            logger.info(f"  Brier Score:       {calib.get('brier_score', 0):.4f} (lower is better)")
            logger.info(f"  Calibration Error: {calib.get('expected_calibration_error', 0):.4f} (lower is better)")

        # Hallucination Detection
        if 'hallucination_detection' in innovative:
            hall = innovative['hallucination_detection']
            logger.info("\n[4] Hallucination Detection:")
            logger.info(f"  Avg Hallucination Rate: {hall.get('avg_hallucination_rate', 0)*100:.1f}%")
            logger.info(f"  Hallucinated Answers:   {hall.get('total_hallucinations', 0)}/{len(results.get('generated_answers', []))}")
            logger.info(f"  {hall.get('interpretation', '')}")

        # Contextual Metrics
        if 'contextual_metrics' in innovative:
            ctx = innovative['contextual_metrics']
            logger.info("\n[5] Contextual Metrics:")
            logger.info(f"  Contextual Precision: {ctx.get('avg_contextual_precision', 0):.4f}")
            logger.info(f"  Contextual Recall:    {ctx.get('avg_contextual_recall', 0):.4f}")
            logger.info(f"  Contextual F1:        {ctx.get('contextual_f1', 0):.4f}")

        # Additional Novel Metrics
        additional = results.get('additional_metrics', {})
        if additional:
            logger.info("\n[6] Additional Novel Metrics:")

            if 'entity_coverage' in additional:
                ec = additional['entity_coverage']
                logger.info(f"  Entity Coverage:      {ec.get('avg_coverage', 0):.4f}")
                logger.info(f"    {ec.get('interpretation', '')}")

            if 'answer_diversity' in additional:
                ad = additional['answer_diversity']
                logger.info(f"  Answer Diversity:     {ad.get('diversity_score', 0):.4f}")
                logger.info(f"    {ad.get('interpretation', '')}")

            if 'faithfulness' in additional:
                faith = additional['faithfulness']
                logger.info(f"  Faithfulness:         {faith.get('avg_faithfulness', 0):.4f}")
                logger.info(f"    {faith.get('interpretation', '')}")

            if 'difficulty_analysis' in additional:
                diff = additional['difficulty_analysis']
                logger.info(f"  Question Difficulty:  {diff.get('interpretation', '')}")

        # Parameter Sweep Results
        params = results.get('parameter_sweep', {})
        if params:
            logger.info("\n[7] Parameter Sweep Results:")
            if 'top_k_sweep' in params:
                logger.info(f"  Optimal top-K:        {params['top_k_sweep'].get('best_k', 'N/A')}")
            if 'top_n_sweep' in params:
                logger.info(f"  Optimal top-N:        {params['top_n_sweep'].get('best_n', 'N/A')}")
            if 'rrf_k_sweep' in params:
                logger.info(f"  Optimal RRF k:        {params['rrf_k_sweep'].get('best_rrf_k', 'N/A')}")

        logger.info("\n" + "="*60)


if __name__ == "__main__":
    # Example usage
    from dense_retrieval import DenseRetriever
    from sparse_retrieval import BM25Retriever
    from rrf_fusion import HybridRetriever
    from llm_generation import LLMGenerator, RAGPipeline

    # Load components
    dense_retriever = DenseRetriever()
    dense_retriever.load_index("data/vector_index")

    sparse_retriever = BM25Retriever()
    sparse_retriever.load_index("data/bm25_index.pkl")

    hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)

    generator = LLMGenerator()
    rag_pipeline = RAGPipeline(hybrid_retriever, generator)

    # Create evaluation pipeline
    eval_pipeline = EvaluationPipeline(rag_pipeline, dense_retriever, sparse_retriever)

    # Run evaluation
    results = eval_pipeline.run_full_evaluation(
        questions_file="data/questions.json",
        output_dir="outputs",
        top_k=10,
        top_n=5
    )

    print("\n=== Evaluation Complete ===")
    print(f"MRR: {results['metrics']['mrr']['mrr']:.4f}")
    print(f"NDCG@5: {results['metrics']['ndcg']['ndcg']:.4f}")
    print(f"BERTScore F1: {results['metrics']['bert_score']['bert_score_f1']:.4f}")
