"""
Evaluation Metrics Module
Implements various metrics for RAG evaluation
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import re
from sklearn.metrics import ndcg_score
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Collection of evaluation metrics for RAG systems"""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # ==================== Mandatory Metric ====================

    def mean_reciprocal_rank_url(
        self,
        questions: List[Dict],
        retrieval_results: List[Dict]
    ) -> Dict:
        """
        Calculate Mean Reciprocal Rank at URL level (Mandatory Metric)

        For each question, finds the rank position of the first correct Wikipedia URL
        in the retrieved results.

        Args:
            questions: List of question dictionaries with 'source_url'
            retrieval_results: List of retrieval results

        Returns:
            Dictionary with MRR score and per-question details
        """
        reciprocal_ranks = []
        details = []

        for question, result in zip(questions, retrieval_results):
            ground_truth_url = question['source_url']
            retrieved_chunks = result.get('fused_results', result.get('results', []))

            # Find the rank of first matching URL
            rank = None
            for i, chunk_result in enumerate(retrieved_chunks, 1):
                retrieved_url = chunk_result['chunk']['url']
                if retrieved_url == ground_truth_url:
                    rank = i
                    break

            if rank is not None:
                rr = 1.0 / rank
            else:
                rr = 0.0

            reciprocal_ranks.append(rr)
            details.append({
                'question_id': question.get('question_id', ''),
                'rank': rank,
                'reciprocal_rank': rr,
                'ground_truth_url': ground_truth_url
            })

        mrr = np.mean(reciprocal_ranks)

        return {
            'mrr': mrr,
            'reciprocal_ranks': reciprocal_ranks,
            'details': details,
            'metric_name': 'Mean Reciprocal Rank (URL Level)',
            'description': 'Average of 1/rank where rank is the position of first correct URL',
            'interpretation': f'MRR of {mrr:.4f} means the correct URL appears on average at position {1/mrr:.1f}' if mrr > 0 else 'No correct URLs found'
        }

    # ==================== Additional Custom Metrics ====================

    def normalized_discounted_cumulative_gain(
        self,
        questions: List[Dict],
        retrieval_results: List[Dict],
        k: int = 10
    ) -> Dict:
        """
        NDCG@K - Normalized Discounted Cumulative Gain (Custom Metric 1)

        Measures ranking quality by considering both relevance and position.
        Higher weights to relevant items appearing earlier in the ranking.

        Mathematical Formulation:
        DCG@k = Σ(i=1 to k) [rel_i / log2(i + 1)]
        NDCG@k = DCG@k / IDCG@k

        Where:
        - rel_i is the relevance of item at position i
        - IDCG is the ideal DCG (perfect ranking)

        Args:
            questions: List of question dictionaries
            retrieval_results: List of retrieval results
            k: Cutoff rank

        Returns:
            Dictionary with NDCG scores and interpretation
        """
        ndcg_scores = []
        details = []

        for question, result in zip(questions, retrieval_results):
            ground_truth_url = question['source_url']
            retrieved_chunks = result.get('fused_results', result.get('results', []))[:k]

            # Create relevance scores (1 for matching URL, 0 otherwise)
            relevance_scores = [
                1.0 if chunk_result['chunk']['url'] == ground_truth_url else 0.0
                for chunk_result in retrieved_chunks
            ]

            # Pad if fewer than k results
            while len(relevance_scores) < k:
                relevance_scores.append(0.0)

            # Calculate DCG
            dcg = sum(
                rel / np.log2(i + 2)  # i+2 because i is 0-indexed
                for i, rel in enumerate(relevance_scores)
            )

            # Calculate ideal DCG (best possible ranking)
            ideal_relevance = sorted(relevance_scores, reverse=True)
            idcg = sum(
                rel / np.log2(i + 2)
                for i, rel in enumerate(ideal_relevance)
            )

            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0

            ndcg_scores.append(ndcg)
            details.append({
                'question_id': question.get('question_id', ''),
                'ndcg': ndcg,
                'dcg': dcg,
                'idcg': idcg
            })

        avg_ndcg = np.mean(ndcg_scores)

        return {
            'ndcg': avg_ndcg,
            'ndcg_scores': ndcg_scores,
            'k': k,
            'details': details,
            'metric_name': f'NDCG@{k}',
            'justification': 'NDCG measures ranking quality by considering both relevance and position. It penalizes relevant documents appearing lower in the ranking, making it ideal for evaluating retrieval systems where rank matters.',
            'calculation': 'DCG = Σ(rel_i / log2(i+1)), NDCG = DCG / IDCG where IDCG is the ideal (perfect) DCG',
            'interpretation': f'NDCG@{k} of {avg_ndcg:.4f} indicates {"excellent" if avg_ndcg > 0.8 else "good" if avg_ndcg > 0.6 else "moderate" if avg_ndcg > 0.4 else "poor"} ranking quality. Score of 1.0 is perfect, 0.0 means no relevant documents retrieved.'
        }

    def bert_score_semantic_similarity(
        self,
        questions: List[Dict],
        generated_answers: List[str],
        model_type: str = "microsoft/deberta-base-mnli"
    ) -> Dict:
        """
        BERTScore - Semantic Similarity (Custom Metric 2)

        Measures semantic similarity between generated and ground truth answers
        using contextual embeddings from BERT-like models.

        Mathematical Formulation:
        For each token in candidate and reference:
        - Compute cosine similarity using BERT embeddings
        - Precision: Average max similarity for each candidate token
        - Recall: Average max similarity for each reference token
        - F1: Harmonic mean of precision and recall

        Args:
            questions: List of question dictionaries with 'answer' field
            generated_answers: List of generated answers
            model_type: BERT model to use for embeddings

        Returns:
            Dictionary with BERTScore metrics and interpretation
        """
        logger.info("Computing BERTScore (this may take a while)...")

        # Extract ground truth answers
        references = [q['answer'] for q in questions]

        # Compute BERTScore
        P, R, F1 = bert_score_fn(
            generated_answers,
            references,
            model_type=model_type,
            verbose=False
        )

        # Convert to numpy
        precision = P.numpy()
        recall = R.numpy()
        f1 = F1.numpy()

        avg_precision = float(np.mean(precision))
        avg_recall = float(np.mean(recall))
        avg_f1 = float(np.mean(f1))

        details = []
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            details.append({
                'question_id': questions[i].get('question_id', ''),
                'precision': float(p),
                'recall': float(r),
                'f1': float(f)
            })

        return {
            'bert_score_precision': avg_precision,
            'bert_score_recall': avg_recall,
            'bert_score_f1': avg_f1,
            'precision_scores': precision.tolist(),
            'recall_scores': recall.tolist(),
            'f1_scores': f1.tolist(),
            'details': details,
            'metric_name': 'BERTScore Semantic Similarity',
            'justification': 'BERTScore captures semantic similarity beyond lexical overlap by using contextual embeddings. It is robust to paraphrasing and better reflects answer quality than traditional metrics like BLEU or ROUGE.',
            'calculation': 'For each token, compute cosine similarity of BERT embeddings. Precision = avg max similarity for candidate tokens, Recall = avg max similarity for reference tokens, F1 = harmonic mean.',
            'interpretation': f'BERTScore F1 of {avg_f1:.4f} indicates {"excellent" if avg_f1 > 0.9 else "good" if avg_f1 > 0.8 else "moderate" if avg_f1 > 0.7 else "poor"} semantic similarity. Higher scores mean generated answers are semantically closer to ground truth.'
        }

    # ==================== Additional Helper Metrics ====================

    def precision_at_k(
        self,
        questions: List[Dict],
        retrieval_results: List[Dict],
        k: int = 5
    ) -> Dict:
        """Calculate Precision@K"""
        precisions = []

        for question, result in zip(questions, retrieval_results):
            ground_truth_url = question['source_url']
            retrieved_chunks = result.get('fused_results', result.get('results', []))[:k]

            relevant_count = sum(
                1 for chunk_result in retrieved_chunks
                if chunk_result['chunk']['url'] == ground_truth_url
            )

            precision = relevant_count / k if k > 0 else 0.0
            precisions.append(precision)

        return {
            'precision_at_k': np.mean(precisions),
            'k': k,
            'precisions': precisions
        }

    def recall_at_k(
        self,
        questions: List[Dict],
        retrieval_results: List[Dict],
        k: int = 5
    ) -> Dict:
        """Calculate Recall@K"""
        recalls = []

        for question, result in zip(questions, retrieval_results):
            ground_truth_url = question['source_url']
            retrieved_chunks = result.get('fused_results', result.get('results', []))[:k]

            # For single URL ground truth, recall is 1 if found, 0 otherwise
            found = any(
                chunk_result['chunk']['url'] == ground_truth_url
                for chunk_result in retrieved_chunks
            )

            recalls.append(1.0 if found else 0.0)

        return {
            'recall_at_k': np.mean(recalls),
            'k': k,
            'recalls': recalls
        }

    def rouge_scores(
        self,
        questions: List[Dict],
        generated_answers: List[str]
    ) -> Dict:
        """Calculate ROUGE scores"""
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for question, generated in zip(questions, generated_answers):
            reference = question['answer']
            scores = self.rouge_scorer.score(reference, generated)

            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores),
            'rouge1_scores': rouge1_scores,
            'rouge2_scores': rouge2_scores,
            'rougeL_scores': rougeL_scores
        }

    def exact_match(
        self,
        questions: List[Dict],
        generated_answers: List[str]
    ) -> Dict:
        """Calculate Exact Match score"""

        def normalize_answer(s):
            """Normalize answer for comparison"""
            s = s.lower()
            s = re.sub(r'\b(a|an|the)\b', ' ', s)
            s = re.sub(r'[^\w\s]', '', s)
            s = ' '.join(s.split())
            return s

        matches = []
        for question, generated in zip(questions, generated_answers):
            reference = question['answer']
            match = normalize_answer(reference) == normalize_answer(generated)
            matches.append(1.0 if match else 0.0)

        return {
            'exact_match': np.mean(matches),
            'matches': matches
        }


if __name__ == "__main__":
    # Example usage
    metrics = EvaluationMetrics()

    # Sample data
    questions = [
        {
            'question_id': 'Q001',
            'question': 'What is ML?',
            'answer': 'Machine learning is a subset of AI.',
            'source_url': 'https://en.wikipedia.org/wiki/Machine_learning'
        }
    ]

    retrieval_results = [
        {
            'fused_results': [
                {'chunk': {'url': 'https://en.wikipedia.org/wiki/Machine_learning', 'text': '...'}},
                {'chunk': {'url': 'https://en.wikipedia.org/wiki/Artificial_intelligence', 'text': '...'}}
            ]
        }
    ]

    generated_answers = ['Machine learning is a type of artificial intelligence.']

    # Calculate metrics
    mrr = metrics.mean_reciprocal_rank_url(questions, retrieval_results)
    print(f"MRR: {mrr['mrr']:.4f}")

    ndcg = metrics.normalized_discounted_cumulative_gain(questions, retrieval_results, k=5)
    print(f"NDCG@5: {ndcg['ndcg']:.4f}")
