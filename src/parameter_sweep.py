"""
RRF Parameter Sweep Module

Tests different values of K (top-K retrieval), N (top-N for generation),
and RRF k-constant to find optimal parameters.

This demonstrates thorough evaluation and system optimization.
"""

import numpy as np
from typing import List, Dict
import logging
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParameterSweep:
    """Systematic parameter optimization for RAG system"""

    def __init__(self, rag_pipeline, evaluation_metrics):
        """
        Initialize parameter sweep

        Args:
            rag_pipeline: RAG pipeline to test
            evaluation_metrics: EvaluationMetrics instance
        """
        self.rag_pipeline = rag_pipeline
        self.metrics = evaluation_metrics

    def sweep_top_k(
        self,
        questions: List[Dict],
        k_values: List[int] = [3, 5, 10, 15, 20],
        fixed_n: int = 5
    ) -> Dict:
        """
        Sweep over different top-K values for retrieval

        Tests how many chunks to retrieve affects performance.

        Args:
            questions: Test questions
            k_values: Different K values to test
            fixed_n: Fixed N for generation

        Returns:
            Results for each K value
        """
        logger.info(f"Sweeping top-K values: {k_values}")

        results = []

        for k in tqdm(k_values, desc="K values"):
            retrieval_results = []
            generated_answers = []

            # Run RAG with this K value
            for question in questions[:20]:  # Sample 20 for speed
                try:
                    result = self.rag_pipeline.query(
                        question['question'],
                        top_k=k,
                        top_n=min(fixed_n, k)  # N can't exceed K
                    )
                    retrieval_results.append(result['retrieval_results'])
                    generated_answers.append(result['answer'])
                except Exception as e:
                    logger.warning(f"Error with K={k}: {e}")
                    retrieval_results.append({'fused_results': []})
                    generated_answers.append("")

            # Calculate metrics
            mrr = self.metrics.mean_reciprocal_rank_url(questions[:20], retrieval_results)
            precision = self.metrics.precision_at_k(questions[:20], retrieval_results, k=min(fixed_n, k))

            results.append({
                'K': k,
                'MRR': mrr['mrr'],
                'Precision@N': precision['precision_at_k']
            })

        return {
            'parameter': 'top_K',
            'results': results,
            'best_k': max(results, key=lambda x: x['MRR'])['K'],
            'interpretation': f"Optimal K value is {max(results, key=lambda x: x['MRR'])['K']} based on MRR"
        }

    def sweep_top_n(
        self,
        questions: List[Dict],
        n_values: List[int] = [3, 5, 7, 10],
        fixed_k: int = 10
    ) -> Dict:
        """
        Sweep over different top-N values for generation

        Tests how many chunks to use for answer generation affects quality.

        Args:
            questions: Test questions
            n_values: Different N values to test
            fixed_k: Fixed K for retrieval

        Returns:
            Results for each N value
        """
        logger.info(f"Sweeping top-N values: {n_values}")

        results = []

        for n in tqdm(n_values, desc="N values"):
            if n > fixed_k:
                continue  # Skip if N > K

            retrieval_results = []
            generated_answers = []

            # Run RAG with this N value
            for question in questions[:20]:  # Sample 20
                try:
                    result = self.rag_pipeline.query(
                        question['question'],
                        top_k=fixed_k,
                        top_n=n
                    )
                    retrieval_results.append(result['retrieval_results'])
                    generated_answers.append(result['answer'])
                except Exception as e:
                    logger.warning(f"Error with N={n}: {e}")
                    retrieval_results.append({'fused_results': []})
                    generated_answers.append("")

            # Calculate metrics
            mrr = self.metrics.mean_reciprocal_rank_url(questions[:20], retrieval_results)

            # Simple answer quality metric (length)
            avg_length = np.mean([len(ans.split()) for ans in generated_answers])

            results.append({
                'N': n,
                'MRR': mrr['mrr'],
                'Avg_Answer_Length': avg_length
            })

        return {
            'parameter': 'top_N',
            'results': results,
            'best_n': max(results, key=lambda x: x['MRR'])['N'],
            'interpretation': f"Optimal N value is {max(results, key=lambda x: x['MRR'])['N']} based on MRR"
        }

    def sweep_rrf_k(
        self,
        questions: List[Dict],
        rrf_k_values: List[int] = [30, 60, 90, 120],
        top_k: int = 10,
        top_n: int = 5
    ) -> Dict:
        """
        Sweep over different RRF k-constant values

        The RRF formula is: RRF_score(d) = Σ 1/(k + rank_i(d))

        Tests how the k constant affects fusion quality.

        Args:
            questions: Test questions
            rrf_k_values: Different RRF k values to test
            top_k: Fixed top-K
            top_n: Fixed top-N

        Returns:
            Results for each RRF k value
        """
        logger.info(f"Sweeping RRF k-constant values: {rrf_k_values}")

        results = []

        for rrf_k in tqdm(rrf_k_values, desc="RRF k values"):
            # Temporarily change RRF k constant
            original_k = self.rag_pipeline.retriever.rrf.k
            self.rag_pipeline.retriever.rrf.k = rrf_k

            retrieval_results = []
            generated_answers = []

            # Run RAG with this RRF k value
            for question in questions[:20]:  # Sample 20
                try:
                    result = self.rag_pipeline.query(
                        question['question'],
                        top_k=top_k,
                        top_n=top_n
                    )
                    retrieval_results.append(result['retrieval_results'])
                    generated_answers.append(result['answer'])
                except Exception as e:
                    logger.warning(f"Error with RRF k={rrf_k}: {e}")
                    retrieval_results.append({'fused_results': []})
                    generated_answers.append("")

            # Calculate metrics
            mrr = self.metrics.mean_reciprocal_rank_url(questions[:20], retrieval_results)

            results.append({
                'RRF_k': rrf_k,
                'MRR': mrr['mrr']
            })

            # Restore original k
            self.rag_pipeline.retriever.rrf.k = original_k

        return {
            'parameter': 'RRF_k_constant',
            'results': results,
            'best_rrf_k': max(results, key=lambda x: x['MRR'])['RRF_k'],
            'interpretation': f"Optimal RRF k-constant is {max(results, key=lambda x: x['MRR'])['RRF_k']} based on MRR"
        }

    def full_parameter_sweep(
        self,
        questions: List[Dict],
        output_dir: str = "outputs"
    ) -> Dict:
        """
        Run complete parameter sweep over K, N, and RRF k

        Args:
            questions: Test questions
            output_dir: Directory to save results

        Returns:
            Complete sweep results
        """
        logger.info("=" * 60)
        logger.info("RUNNING FULL PARAMETER SWEEP")
        logger.info("=" * 60)

        results = {}

        # Sweep top-K
        results['top_k_sweep'] = self.sweep_top_k(questions)

        # Sweep top-N
        results['top_n_sweep'] = self.sweep_top_n(questions)

        # Sweep RRF k
        results['rrf_k_sweep'] = self.sweep_rrf_k(questions)

        # Generate visualizations
        self.visualize_parameter_sweep(results, output_dir)

        logger.info("\nParameter Sweep Summary:")
        logger.info(f"  Best top-K: {results['top_k_sweep']['best_k']}")
        logger.info(f"  Best top-N: {results['top_n_sweep']['best_n']}")
        logger.info(f"  Best RRF k: {results['rrf_k_sweep']['best_rrf_k']}")

        return results

    def visualize_parameter_sweep(self, results: Dict, output_dir: str):
        """Create visualizations for parameter sweep results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Top-K sweep
        k_data = results['top_k_sweep']['results']
        k_values = [r['K'] for r in k_data]
        k_mrr = [r['MRR'] for r in k_data]

        axes[0].plot(k_values, k_mrr, 'o-', linewidth=2, markersize=8, color='#2E86AB')
        axes[0].set_xlabel('Top-K Value', fontsize=12)
        axes[0].set_ylabel('MRR Score', fontsize=12)
        axes[0].set_title('MRR vs Top-K Retrieval', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(results['top_k_sweep']['best_k'], color='red', linestyle='--',
                        label=f"Best K={results['top_k_sweep']['best_k']}")
        axes[0].legend()

        # Plot 2: Top-N sweep
        n_data = results['top_n_sweep']['results']
        n_values = [r['N'] for r in n_data]
        n_mrr = [r['MRR'] for r in n_data]

        axes[1].plot(n_values, n_mrr, 's-', linewidth=2, markersize=8, color='#A23B72')
        axes[1].set_xlabel('Top-N Value', fontsize=12)
        axes[1].set_ylabel('MRR Score', fontsize=12)
        axes[1].set_title('MRR vs Top-N Generation', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(results['top_n_sweep']['best_n'], color='red', linestyle='--',
                        label=f"Best N={results['top_n_sweep']['best_n']}")
        axes[1].legend()

        # Plot 3: RRF k sweep
        rrf_data = results['rrf_k_sweep']['results']
        rrf_values = [r['RRF_k'] for r in rrf_data]
        rrf_mrr = [r['MRR'] for r in rrf_data]

        axes[2].plot(rrf_values, rrf_mrr, '^-', linewidth=2, markersize=8, color='#F18F01')
        axes[2].set_xlabel('RRF k-constant', fontsize=12)
        axes[2].set_ylabel('MRR Score', fontsize=12)
        axes[2].set_title('MRR vs RRF k-constant', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].axvline(results['rrf_k_sweep']['best_rrf_k'], color='red', linestyle='--',
                        label=f"Best k={results['rrf_k_sweep']['best_rrf_k']}")
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(output_path / 'parameter_sweep_results.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Parameter sweep visualization saved to {output_path / 'parameter_sweep_results.png'}")


if __name__ == "__main__":
    logger.info("Parameter Sweep Module loaded successfully!")
    logger.info("Enables systematic optimization of K, N, and RRF k parameters")
