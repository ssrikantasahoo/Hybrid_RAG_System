"""
Advanced Visualization Module
Creates comprehensive visualizations for innovative evaluation metrics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedVisualizer:
    """Create advanced visualizations for innovative evaluation"""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        sns.set_palette("husl")

    def plot_llm_judge_radar(self, llm_judge_results: Dict, filename: str = "llm_judge_radar.png"):
        """
        Create radar chart for LLM-as-Judge dimensions

        Args:
            llm_judge_results: Results from LLM-as-Judge evaluation
            filename: Output filename
        """
        logger.info("Creating LLM-as-Judge radar chart...")

        categories = ['Factual\nAccuracy', 'Completeness', 'Relevance', 'Coherence']
        values = [
            llm_judge_results.get('avg_factual_accuracy', 0),
            llm_judge_results.get('avg_completeness', 0),
            llm_judge_results.get('avg_relevance', 0),
            llm_judge_results.get('avg_coherence', 0)
        ]

        # Close the plot by appending the first value
        values += values[:1]
        categories_plot = categories + [categories[0]]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label='RAG System', color='#2E86AB')
        ax.fill(angles, values, alpha=0.25, color='#2E86AB')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)

        plt.title('LLM-as-Judge Evaluation Scores', size=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved LLM-as-Judge radar chart to {self.output_dir / filename}")

    def plot_calibration_curve(self, calibration_results: Dict, filename: str = "calibration_curve.png"):
        """
        Create calibration curve plot

        Args:
            calibration_results: Results from confidence calibration
            filename: Output filename
        """
        logger.info("Creating calibration curve...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Calibration curve
        curve_data = calibration_results.get('calibration_curve', {})
        prob_true = curve_data.get('prob_true', [])
        prob_pred = curve_data.get('prob_pred', [])

        if prob_true and prob_pred:
            ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
            ax1.plot(prob_pred, prob_true, 'o-', label='RAG System', linewidth=2, markersize=8, color='#A23B72')
            ax1.set_xlabel('Predicted Confidence', fontsize=12)
            ax1.set_ylabel('Actual Accuracy', fontsize=12)
            ax1.set_title('Calibration Curve', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim([0, 1])
            ax1.set_ylim([0, 1])

        # Confidence distribution
        confidences = calibration_results.get('confidences', [])
        correctness = calibration_results.get('correctness', [])

        if confidences and correctness:
            correct_conf = [c for c, cor in zip(confidences, correctness) if cor == 1.0]
            incorrect_conf = [c for c, cor in zip(confidences, correctness) if cor == 0.0]

            ax2.hist(correct_conf, bins=20, alpha=0.6, label='Correct', color='green', edgecolor='black')
            ax2.hist(incorrect_conf, bins=20, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
            ax2.set_xlabel('Confidence Score', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Confidence Distribution by Correctness', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')

        # Add metrics text
        brier = calibration_results.get('brier_score', 0)
        ece = calibration_results.get('expected_calibration_error', 0)
        fig.text(0.5, 0.02, f'Brier Score: {brier:.4f} | Expected Calibration Error: {ece:.4f}',
                 ha='center', fontsize=11, style='italic')

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved calibration curve to {self.output_dir / filename}")

    def plot_hallucination_analysis(self, hallucination_results: Dict, filename: str = "hallucination_analysis.png"):
        """
        Create hallucination analysis visualization

        Args:
            hallucination_results: Results from hallucination detection
            filename: Output filename
        """
        logger.info("Creating hallucination analysis...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Pie chart: Hallucinated vs Clean
        total = len(hallucination_results.get('details', []))
        hallucinated = hallucination_results.get('total_hallucinations', 0)
        clean = total - hallucinated

        colors = ['#FF6B6B', '#4ECDC4']
        ax1.pie([hallucinated, clean], labels=['Hallucinated', 'Clean'],
                autopct='%1.1f%%', colors=colors, startangle=90,
                textprops={'fontsize': 12, 'weight': 'bold'})
        ax1.set_title('Answer Hallucination Rate', fontsize=14, fontweight='bold')

        # Histogram: Hallucination scores
        details = hallucination_results.get('details', [])
        if details:
            scores = [d.get('hallucination_score', 0) for d in details]
            ax2.hist(scores, bins=20, edgecolor='black', alpha=0.7, color='#FF6B6B')
            ax2.set_xlabel('Hallucination Score', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Distribution of Hallucination Scores', fontsize=14, fontweight='bold')
            ax2.axvline(0.3, color='red', linestyle='--', linewidth=2,
                       label='Threshold (0.3)')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved hallucination analysis to {self.output_dir / filename}")

    def plot_adversarial_results(self, adversarial_results: Dict, filename: str = "adversarial_testing.png"):
        """
        Create adversarial testing visualizations

        Args:
            adversarial_results: Results from adversarial testing
            filename: Output filename
        """
        logger.info("Creating adversarial testing visualizations...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Paraphrasing consistency
        paraphrase_data = adversarial_results.get('paraphrasing_robustness', {})
        consistency_scores = paraphrase_data.get('consistency_scores', [])

        if consistency_scores:
            ax1.hist(consistency_scores, bins=20, edgecolor='black', alpha=0.7, color='#6C5B7B')
            ax1.axvline(np.mean(consistency_scores), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(consistency_scores):.3f}')
            ax1.set_xlabel('Consistency Score', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.set_title('Paraphrasing Robustness', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3, axis='y')

        # Unanswerable detection
        unanswerable_data = adversarial_results.get('unanswerable_detection', {})
        hallucination_rate = unanswerable_data.get('hallucination_rate', 0)
        good_rate = 1 - hallucination_rate

        categories = ['Detected\nUnanswerable', 'Hallucinated\nAnswer']
        values = [good_rate * 100, hallucination_rate * 100]
        colors = ['#4ECDC4', '#FF6B6B']

        bars = ax2.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
        ax2.set_ylabel('Percentage (%)', fontsize=12)
        ax2.set_title('Unanswerable Question Detection', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved adversarial testing results to {self.output_dir / filename}")

    def plot_comprehensive_metrics_dashboard(self, all_results: Dict, filename: str = "comprehensive_dashboard.png"):
        """
        Create comprehensive metrics dashboard

        Args:
            all_results: All evaluation results
            filename: Output filename
        """
        logger.info("Creating comprehensive metrics dashboard...")

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Main Metrics (Top left)
        ax1 = fig.add_subplot(gs[0, 0])
        metrics_data = all_results.get('metrics', {})
        metric_names = ['MRR', 'NDCG@5', 'BERTScore\nF1', 'Precision@5', 'Recall@5']
        metric_values = [
            metrics_data.get('mrr', {}).get('mrr', 0),
            metrics_data.get('ndcg', {}).get('ndcg', 0),
            metrics_data.get('bert_score', {}).get('bert_score_f1', 0),
            metrics_data.get('precision_at_k', {}).get('precision_at_k', 0),
            metrics_data.get('recall_at_k', {}).get('recall_at_k', 0)
        ]

        bars = ax1.barh(metric_names, metric_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
        ax1.set_xlim([0, 1])
        ax1.set_xlabel('Score', fontsize=10)
        ax1.set_title('Core Retrieval & Generation Metrics', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

        # 2. LLM-as-Judge Scores (Top middle)
        ax2 = fig.add_subplot(gs[0, 1], projection='polar')
        llm_judge = all_results.get('innovative_metrics', {}).get('llm_as_judge', {})
        categories = ['Factual', 'Complete', 'Relevant', 'Coherent']
        values = [
            llm_judge.get('avg_factual_accuracy', 0),
            llm_judge.get('avg_completeness', 0),
            llm_judge.get('avg_relevance', 0),
            llm_judge.get('avg_coherence', 0)
        ]
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        ax2.plot(angles, values, 'o-', linewidth=2, color='#2E86AB')
        ax2.fill(angles, values, alpha=0.25, color='#2E86AB')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories, size=9)
        ax2.set_ylim(0, 1)
        ax2.set_title('LLM-as-Judge\nScores', fontsize=11, fontweight='bold', pad=15)
        ax2.grid(True)

        # 3. Ablation Study (Top right)
        ax3 = fig.add_subplot(gs[0, 2])
        ablation = all_results.get('ablation_study', {})
        methods = ['Dense', 'Sparse', 'Hybrid']
        mrr_values = [
            ablation.get('dense_only', {}).get('mrr', 0),
            ablation.get('sparse_only', {}).get('mrr', 0),
            ablation.get('hybrid_rrf', {}).get('mrr', 0)
        ]
        ndcg_values = [
            ablation.get('dense_only', {}).get('ndcg', 0),
            ablation.get('sparse_only', {}).get('ndcg', 0),
            ablation.get('hybrid_rrf', {}).get('ndcg', 0)
        ]

        x = np.arange(len(methods))
        width = 0.35

        bars1 = ax3.bar(x - width/2, mrr_values, width, label='MRR', color='#2E86AB')
        bars2 = ax3.bar(x + width/2, ndcg_values, width, label='NDCG', color='#A23B72')

        ax3.set_ylabel('Score', fontsize=10)
        ax3.set_title('Ablation Study Results', fontsize=11, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(methods)
        ax3.legend(fontsize=9)
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Response Time Distribution (Middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        response_times = all_results.get('response_times', [])
        if response_times:
            ax4.hist(response_times, bins=30, edgecolor='black', alpha=0.7, color='#F18F01')
            ax4.axvline(np.mean(response_times), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(response_times):.2f}s')
            ax4.set_xlabel('Time (seconds)', fontsize=10)
            ax4.set_ylabel('Frequency', fontsize=10)
            ax4.set_title('Response Time Distribution', fontsize=11, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3, axis='y')

        # 5. Calibration Curve (Middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        calibration = all_results.get('innovative_metrics', {}).get('confidence_calibration', {})
        curve_data = calibration.get('calibration_curve', {})
        prob_true = curve_data.get('prob_true', [])
        prob_pred = curve_data.get('prob_pred', [])

        if prob_true and prob_pred:
            ax5.plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=2)
            ax5.plot(prob_pred, prob_true, 'o-', label='System', linewidth=2, markersize=6, color='#A23B72')
            ax5.set_xlabel('Predicted', fontsize=10)
            ax5.set_ylabel('Actual', fontsize=10)
            ax5.set_title('Confidence Calibration', fontsize=11, fontweight='bold')
            ax5.legend(fontsize=9)
            ax5.grid(True, alpha=0.3)
            ax5.set_xlim([0, 1])
            ax5.set_ylim([0, 1])

        # 6. Hallucination Analysis (Middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        hallucination = all_results.get('innovative_metrics', {}).get('hallucination_detection', {})
        hall_rate = hallucination.get('hallucination_percentage', 0)
        clean_rate = 100 - hall_rate

        sizes = [clean_rate, hall_rate]
        colors_pie = ['#4ECDC4', '#FF6B6B']
        labels = ['Clean', 'Hallucinated']
        ax6.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie,
               startangle=90, textprops={'fontsize': 9})
        ax6.set_title('Hallucination Rate', fontsize=11, fontweight='bold')

        # 7. Error Analysis (Bottom left)
        ax7 = fig.add_subplot(gs[2, 0])
        error_analysis = all_results.get('error_analysis', {}).get('errors_by_type', {})
        if error_analysis:
            question_types = list(error_analysis.keys())
            retrieval_failures = [error_analysis[qt]['retrieval_failures'] for qt in question_types]
            generation_failures = [error_analysis[qt]['generation_failures'] for qt in question_types]

            x = np.arange(len(question_types))
            width = 0.35

            ax7.bar(x - width/2, retrieval_failures, width, label='Retrieval', color='#C73E1D')
            ax7.bar(x + width/2, generation_failures, width, label='Generation', color='#FF6B6B')

            ax7.set_xlabel('Question Type', fontsize=10)
            ax7.set_ylabel('Failures', fontsize=10)
            ax7.set_title('Error Analysis by Type', fontsize=11, fontweight='bold')
            ax7.set_xticks(x)
            ax7.set_xticklabels(question_types, fontsize=8, rotation=15)
            ax7.legend(fontsize=9)
            ax7.grid(True, alpha=0.3, axis='y')

        # 8. Adversarial Testing (Bottom middle)
        ax8 = fig.add_subplot(gs[2, 1])
        adversarial = all_results.get('innovative_metrics', {}).get('adversarial_testing', {})
        paraphrase_consistency = adversarial.get('paraphrasing_robustness', {}).get('avg_paraphrase_consistency', 0)
        unanswerable = adversarial.get('unanswerable_detection', {}).get('hallucination_rate', 0)

        categories_adv = ['Paraphrase\nConsistency', 'Unanswerable\nDetection']
        values_adv = [paraphrase_consistency * 100, (1 - unanswerable) * 100]
        colors_adv = ['#6C5B7B', '#4ECDC4']

        bars = ax8.bar(categories_adv, values_adv, color=colors_adv, edgecolor='black', alpha=0.8)
        ax8.set_ylabel('Score (%)', fontsize=10)
        ax8.set_title('Adversarial Testing', fontsize=11, fontweight='bold')
        ax8.set_ylim([0, 100])
        ax8.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 9. Contextual Metrics (Bottom right)
        ax9 = fig.add_subplot(gs[2, 2])
        contextual = all_results.get('innovative_metrics', {}).get('contextual_metrics', {})
        ctx_precision = contextual.get('avg_contextual_precision', 0)
        ctx_recall = contextual.get('avg_contextual_recall', 0)
        ctx_f1 = contextual.get('contextual_f1', 0)

        categories_ctx = ['Precision', 'Recall', 'F1']
        values_ctx = [ctx_precision, ctx_recall, ctx_f1]
        colors_ctx = ['#2E86AB', '#A23B72', '#F18F01']

        bars = ax9.bar(categories_ctx, values_ctx, color=colors_ctx, edgecolor='black', alpha=0.8)
        ax9.set_ylabel('Score', fontsize=10)
        ax9.set_title('Contextual Metrics', fontsize=11, fontweight='bold')
        ax9.set_ylim([0, 1])
        ax9.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Add main title
        fig.suptitle('Hybrid RAG System - Comprehensive Evaluation Dashboard',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved comprehensive dashboard to {self.output_dir / filename}")

    def create_all_visualizations(self, results: Dict):
        """
        Create all visualizations

        Args:
            results: Complete evaluation results
        """
        logger.info("Creating all advanced visualizations...")

        innovative = results.get('innovative_metrics', {})

        # LLM-as-Judge
        if 'llm_as_judge' in innovative:
            self.plot_llm_judge_radar(innovative['llm_as_judge'])

        # Calibration
        if 'confidence_calibration' in innovative:
            self.plot_calibration_curve(innovative['confidence_calibration'])

        # Hallucination
        if 'hallucination_detection' in innovative:
            self.plot_hallucination_analysis(innovative['hallucination_detection'])

        # Adversarial
        if 'adversarial_testing' in innovative:
            self.plot_adversarial_results(innovative['adversarial_testing'])

        # Comprehensive dashboard
        self.plot_comprehensive_metrics_dashboard(results)

        logger.info("All visualizations created successfully!")


if __name__ == "__main__":
    logger.info("Advanced Visualization Module loaded successfully!")
