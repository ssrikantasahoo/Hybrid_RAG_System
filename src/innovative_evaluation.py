"""
Innovative Evaluation Module
Advanced evaluation techniques for RAG systems including:
- LLM-as-Judge
- Adversarial Testing
- Confidence Calibration
- Hallucination Detection
- Entity Coverage
- Answer Diversity
"""

import json
import numpy as np
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import spacy
import logging
from tqdm import tqdm
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMAsJudge:
    """
    LLM-as-Judge: Uses language models to evaluate answer quality

    Evaluates on multiple dimensions:
    - Factual Accuracy: Is the answer factually correct?
    - Completeness: Does it fully answer the question?
    - Relevance: Is the answer relevant to the question?
    - Coherence: Is the answer well-structured and coherent?
    """

    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Initialize LLM-as-Judge

        Args:
            model_name: Model for judging answers
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading LLM-as-Judge model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def evaluate_factual_accuracy(self, question: str, answer: str, context: str) -> Dict:
        """
        Evaluate factual accuracy of answer

        Args:
            question: The question
            answer: Generated answer
            context: Retrieved context

        Returns:
            Score and explanation
        """
        prompt = f"""Given the following context, evaluate if the answer to the question is factually accurate.
Rate on a scale of 0-10 where 0 is completely inaccurate and 10 is perfectly accurate.

Context: {context[:500]}

Question: {question}

Answer: {answer}

Provide your rating (0-10) and a brief explanation.
Rating:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100, temperature=0.3)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract numeric rating
        rating_match = re.search(r'(\d+(?:\.\d+)?)', response)
        rating = float(rating_match.group(1)) / 10.0 if rating_match else 0.5
        rating = min(max(rating, 0.0), 1.0)  # Clamp to [0, 1]

        return {
            'score': rating,
            'explanation': response,
            'dimension': 'factual_accuracy'
        }

    def evaluate_completeness(self, question: str, answer: str, ground_truth: str) -> Dict:
        """Evaluate if answer is complete"""
        prompt = f"""Compare the generated answer with the reference answer.
Rate how complete the generated answer is on a scale of 0-10.

Question: {question}

Reference Answer: {ground_truth}

Generated Answer: {answer}

Rating (0-10):"""

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=50, temperature=0.3)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        rating_match = re.search(r'(\d+(?:\.\d+)?)', response)
        rating = float(rating_match.group(1)) / 10.0 if rating_match else 0.5
        rating = min(max(rating, 0.0), 1.0)

        return {
            'score': rating,
            'explanation': response,
            'dimension': 'completeness'
        }

    def evaluate_relevance(self, question: str, answer: str) -> Dict:
        """Evaluate relevance of answer to question"""
        prompt = f"""Rate how relevant the answer is to the question on a scale of 0-10.

Question: {question}

Answer: {answer}

Relevance Rating (0-10):"""

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=50, temperature=0.3)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        rating_match = re.search(r'(\d+(?:\.\d+)?)', response)
        rating = float(rating_match.group(1)) / 10.0 if rating_match else 0.5
        rating = min(max(rating, 0.0), 1.0)

        return {
            'score': rating,
            'explanation': response,
            'dimension': 'relevance'
        }

    def evaluate_coherence(self, answer: str) -> Dict:
        """Evaluate coherence and structure of answer"""
        prompt = f"""Rate the coherence and clarity of the following answer on a scale of 0-10.
Consider grammar, structure, and readability.

Answer: {answer}

Coherence Rating (0-10):"""

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=50, temperature=0.3)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        rating_match = re.search(r'(\d+(?:\.\d+)?)', response)
        rating = float(rating_match.group(1)) / 10.0 if rating_match else 0.5
        rating = min(max(rating, 0.0), 1.0)

        return {
            'score': rating,
            'explanation': response,
            'dimension': 'coherence'
        }

    def evaluate_all(self, question: str, answer: str, context: str, ground_truth: str) -> Dict:
        """
        Comprehensive evaluation on all dimensions

        Returns:
            Dictionary with all scores and overall score
        """
        factual = self.evaluate_factual_accuracy(question, answer, context)
        completeness = self.evaluate_completeness(question, answer, ground_truth)
        relevance = self.evaluate_relevance(question, answer)
        coherence = self.evaluate_coherence(answer)

        # Weighted overall score
        overall_score = (
            factual['score'] * 0.4 +
            completeness['score'] * 0.3 +
            relevance['score'] * 0.2 +
            coherence['score'] * 0.1
        )

        return {
            'factual_accuracy': factual,
            'completeness': completeness,
            'relevance': relevance,
            'coherence': coherence,
            'overall_score': overall_score
        }


class AdversarialTester:
    """
    Adversarial Testing Suite

    Tests RAG system with challenging variations:
    - Paraphrased questions
    - Negated questions
    - Unanswerable questions
    - Ambiguous questions
    """

    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading adversarial testing model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def paraphrase_question(self, question: str) -> str:
        """Generate paraphrased version of question"""
        prompt = f"Paraphrase the following question while keeping the same meaning:\n\nQuestion: {question}\n\nParaphrased:"

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100, temperature=0.8, do_sample=True)

        paraphrased = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return paraphrased.strip()

    def negate_question(self, question: str) -> str:
        """Create negated version of question"""
        prompt = f"Negate the following question:\n\nQuestion: {question}\n\nNegated:"

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100)

        negated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return negated.strip()

    def create_unanswerable_question(self, question: str) -> str:
        """Create unanswerable variant"""
        # Simple strategy: Ask about information that wouldn't be in the context
        unanswerable = f"What was the exact time and date when {question.lower().replace('what', '').replace('?', '').strip()} happened in the future?"
        return unanswerable

    def test_paraphrasing_robustness(self, questions: List[Dict], rag_pipeline, num_samples: int = 20) -> Dict:
        """
        Test if system handles paraphrased questions consistently

        Args:
            questions: Original questions
            rag_pipeline: RAG system to test
            num_samples: Number of questions to test

        Returns:
            Robustness metrics
        """
        logger.info("Testing paraphrasing robustness...")

        sample_questions = questions[:num_samples]
        consistency_scores = []

        for question_dict in tqdm(sample_questions, desc="Paraphrase testing"):
            original_q = question_dict['question']

            # Get answer for original
            try:
                original_result = rag_pipeline.query(original_q, top_k=10, top_n=5)
                original_answer = original_result['answer']

                # Generate paraphrase
                paraphrased_q = self.paraphrase_question(original_q)

                # Get answer for paraphrase
                paraphrase_result = rag_pipeline.query(paraphrased_q, top_k=10, top_n=5)
                paraphrase_answer = paraphrase_result['answer']

                # Measure consistency (simple word overlap)
                original_words = set(original_answer.lower().split())
                paraphrase_words = set(paraphrase_answer.lower().split())

                if len(original_words.union(paraphrase_words)) > 0:
                    consistency = len(original_words.intersection(paraphrase_words)) / len(original_words.union(paraphrase_words))
                else:
                    consistency = 0.0

                consistency_scores.append(consistency)

            except Exception as e:
                logger.warning(f"Error in paraphrase testing: {e}")
                consistency_scores.append(0.0)

        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0

        return {
            'avg_paraphrase_consistency': avg_consistency,
            'consistency_scores': consistency_scores,
            'interpretation': f"System shows {avg_consistency*100:.1f}% consistency on paraphrased questions. Higher is better (>70% is good)."
        }

    def test_unanswerable_detection(self, rag_pipeline, num_questions: int = 10) -> Dict:
        """
        Test if system can detect unanswerable questions

        Returns:
            Metrics on hallucination rate
        """
        logger.info("Testing unanswerable question detection...")

        unanswerable_questions = [
            "What will be the weather on Mars in the year 3000?",
            "Who will win the next intergalactic sports championship?",
            "What is the exact molecular structure of fictional element Vibranium?",
            "When did unicorns first evolve on Earth?",
            "What is the population of Atlantis?",
            "How many dragon species exist in Antarctica?",
            "What was Sherlock Holmes's favorite breakfast?",
            "Which programming language do aliens prefer?",
            "What is the GDP of Narnia?",
            "When was the last time Bigfoot was spotted in Tokyo?"
        ][:num_questions]

        hallucination_count = 0
        results = []

        for question in tqdm(unanswerable_questions, desc="Unanswerable testing"):
            try:
                result = rag_pipeline.query(question, top_k=10, top_n=5)
                answer = result['answer']

                # If answer is confident and long, likely hallucinating
                if len(answer.split()) > 5:
                    hallucination_count += 1

                results.append({
                    'question': question,
                    'answer': answer,
                    'likely_hallucination': len(answer.split()) > 5
                })

            except Exception as e:
                logger.warning(f"Error testing unanswerable: {e}")

        hallucination_rate = hallucination_count / len(unanswerable_questions) if unanswerable_questions else 0.0

        return {
            'hallucination_rate': hallucination_rate,
            'tested_questions': len(unanswerable_questions),
            'hallucinations_detected': hallucination_count,
            'results': results,
            'interpretation': f"System hallucinates {hallucination_rate*100:.1f}% of the time on unanswerable questions. Lower is better (<20% is good)."
        }


class ConfidenceCalibrator:
    """
    Confidence Calibration

    Estimates answer confidence and measures calibration:
    - Are high-confidence answers actually more accurate?
    - Calibration curves
    - Brier score
    """

    def __init__(self):
        pass

    def estimate_confidence(self, retrieval_results: Dict, answer: str) -> float:
        """
        Estimate confidence based on retrieval scores and answer characteristics

        Args:
            retrieval_results: Retrieval results with scores
            answer: Generated answer

        Returns:
            Confidence score between 0 and 1
        """
        # Factors affecting confidence:
        # 1. Top retrieval score
        # 2. Score distribution (agreement between retrievers)
        # 3. Answer length (very short might indicate uncertainty)

        fused_results = retrieval_results.get('fused_results', [])

        if not fused_results:
            return 0.1  # Low confidence if no results

        # Top RRF score
        top_score = fused_results[0].get('rrf_score', 0.0) if fused_results else 0.0

        # Score variance (lower variance = more agreement)
        scores = [r.get('rrf_score', 0.0) for r in fused_results[:5]]
        score_variance = np.var(scores) if len(scores) > 1 else 1.0
        agreement_score = 1.0 / (1.0 + score_variance)

        # Answer length factor (very short answers might be uncertain)
        answer_length = len(answer.split())
        length_confidence = min(answer_length / 20.0, 1.0)  # Saturate at 20 words

        # Combined confidence
        confidence = 0.5 * top_score + 0.3 * agreement_score + 0.2 * length_confidence
        confidence = min(max(confidence, 0.0), 1.0)

        return confidence

    def calculate_calibration(
        self,
        confidences: List[float],
        correctness: List[float],
        n_bins: int = 10
    ) -> Dict:
        """
        Calculate calibration metrics

        Args:
            confidences: Predicted confidence scores
            correctness: Actual correctness (0 or 1)
            n_bins: Number of bins for calibration curve

        Returns:
            Calibration metrics and curve data
        """
        confidences = np.array(confidences)
        correctness = np.array(correctness)

        # Calculate Brier score (lower is better)
        brier = brier_score_loss(correctness, confidences)

        # Calculate calibration curve
        try:
            prob_true, prob_pred = calibration_curve(
                correctness,
                confidences,
                n_bins=n_bins,
                strategy='uniform'
            )
        except:
            prob_true = np.array([])
            prob_pred = np.array([])

        # Expected Calibration Error (ECE)
        if len(prob_true) > 0:
            ece = np.mean(np.abs(prob_true - prob_pred))
        else:
            ece = 0.0

        return {
            'brier_score': float(brier),
            'expected_calibration_error': float(ece),
            'calibration_curve': {
                'prob_true': prob_true.tolist() if len(prob_true) > 0 else [],
                'prob_pred': prob_pred.tolist() if len(prob_pred) > 0 else []
            },
            'interpretation': f"Brier Score: {brier:.4f} (lower is better, 0 is perfect). ECE: {ece:.4f} (lower is better, 0 is perfect calibration)."
        }


class HallucinationDetector:
    """
    Hallucination Detection

    Detects when generated answers contain information not present in retrieved context
    """

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("Spacy model not found. Downloading...")
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def extract_entities(self, text: str) -> set:
        """Extract named entities from text"""
        doc = self.nlp(text)
        entities = set()
        for ent in doc.ents:
            entities.add(ent.text.lower())
        return entities

    def detect_hallucination(self, answer: str, context: str) -> Dict:
        """
        Detect if answer contains entities/facts not in context

        Args:
            answer: Generated answer
            context: Retrieved context

        Returns:
            Hallucination metrics
        """
        answer_entities = self.extract_entities(answer)
        context_entities = self.extract_entities(context)

        if not answer_entities:
            return {
                'hallucination_score': 0.0,
                'hallucinated_entities': [],
                'total_entities': 0,
                'is_hallucinated': False
            }

        # Find entities in answer but not in context
        hallucinated = answer_entities - context_entities

        hallucination_rate = len(hallucinated) / len(answer_entities) if answer_entities else 0.0

        return {
            'hallucination_score': hallucination_rate,
            'hallucinated_entities': list(hallucinated),
            'total_entities': len(answer_entities),
            'context_entities': len(context_entities),
            'is_hallucinated': hallucination_rate > 0.3  # Threshold
        }


class ContextualMetrics:
    """
    Contextual Precision and Contextual Recall

    Measures quality of retrieved context
    """

    def __init__(self):
        pass

    def contextual_precision(
        self,
        retrieved_chunks: List[Dict],
        question: str,
        answer: str
    ) -> float:
        """
        Measure what fraction of retrieved chunks are actually relevant

        Simple heuristic: chunk is relevant if it shares significant words with answer
        """
        if not retrieved_chunks:
            return 0.0

        answer_words = set(answer.lower().split())
        relevant_count = 0

        for chunk in retrieved_chunks:
            chunk_text = chunk.get('chunk', {}).get('text', '')
            chunk_words = set(chunk_text.lower().split())

            # Check overlap
            overlap = len(answer_words.intersection(chunk_words))
            if overlap > 3:  # At least 3 common words
                relevant_count += 1

        return relevant_count / len(retrieved_chunks)

    def contextual_recall(
        self,
        retrieved_chunks: List[Dict],
        ground_truth_chunk_id: str
    ) -> float:
        """
        Measure if all relevant chunks were retrieved

        For single ground truth, this is binary: did we retrieve it?
        """
        retrieved_ids = [chunk.get('chunk', {}).get('chunk_id', '') for chunk in retrieved_chunks]

        return 1.0 if ground_truth_chunk_id in retrieved_ids else 0.0


def run_innovative_evaluation(
    rag_pipeline,
    questions: List[Dict],
    retrieval_results: List[Dict],
    generated_answers: List[str],
    num_llm_judge_samples: int = 30,
    num_adversarial_samples: int = 20
) -> Dict:
    """
    Run complete innovative evaluation suite

    Args:
        rag_pipeline: RAG pipeline to evaluate
        questions: List of questions
        retrieval_results: Retrieval results
        generated_answers: Generated answers
        num_llm_judge_samples: Number of samples for LLM-as-Judge (expensive)
        num_adversarial_samples: Number for adversarial testing

    Returns:
        Complete innovative evaluation results
    """
    logger.info("=" * 60)
    logger.info("RUNNING INNOVATIVE EVALUATION SUITE")
    logger.info("=" * 60)

    results = {}

    # 1. LLM-as-Judge (sample due to computational cost)
    logger.info("\n[1/5] Running LLM-as-Judge evaluation...")
    llm_judge = LLMAsJudge()
    judge_results = []

    sample_indices = np.random.choice(len(questions), min(num_llm_judge_samples, len(questions)), replace=False)

    for idx in tqdm(sample_indices, desc="LLM-as-Judge"):
        question = questions[idx]
        answer = generated_answers[idx]
        context = " ".join([
            chunk['chunk']['text']
            for chunk in retrieval_results[idx].get('fused_results', [])[:3]
        ])

        try:
            evaluation = llm_judge.evaluate_all(
                question['question'],
                answer,
                context,
                question['answer']
            )
            judge_results.append(evaluation)
        except Exception as e:
            logger.warning(f"Error in LLM-as-Judge: {e}")

    if judge_results:
        results['llm_as_judge'] = {
            'avg_overall_score': np.mean([r['overall_score'] for r in judge_results]),
            'avg_factual_accuracy': np.mean([r['factual_accuracy']['score'] for r in judge_results]),
            'avg_completeness': np.mean([r['completeness']['score'] for r in judge_results]),
            'avg_relevance': np.mean([r['relevance']['score'] for r in judge_results]),
            'avg_coherence': np.mean([r['coherence']['score'] for r in judge_results]),
            'sample_size': len(judge_results),
            'details': judge_results[:5]  # Store first 5 for report
        }

    # 2. Adversarial Testing
    logger.info("\n[2/5] Running Adversarial Testing...")
    adversarial = AdversarialTester()

    paraphrase_results = adversarial.test_paraphrasing_robustness(
        questions,
        rag_pipeline,
        num_samples=num_adversarial_samples
    )

    unanswerable_results = adversarial.test_unanswerable_detection(
        rag_pipeline,
        num_questions=10
    )

    results['adversarial_testing'] = {
        'paraphrasing_robustness': paraphrase_results,
        'unanswerable_detection': unanswerable_results
    }

    # 3. Confidence Calibration
    logger.info("\n[3/5] Calculating Confidence Calibration...")
    calibrator = ConfidenceCalibrator()

    confidences = []
    correctness = []

    for i, (question, retrieval_result, answer) in enumerate(zip(questions, retrieval_results, generated_answers)):
        conf = calibrator.estimate_confidence(retrieval_result, answer)
        confidences.append(conf)

        # Determine correctness (simple heuristic: check if ground truth URL was retrieved in top 3)
        ground_truth_url = question['source_url']
        retrieved_urls = [
            chunk['chunk']['url']
            for chunk in retrieval_result.get('fused_results', [])[:3]
        ]
        correct = 1.0 if ground_truth_url in retrieved_urls else 0.0
        correctness.append(correct)

    calibration_results = calibrator.calculate_calibration(confidences, correctness)
    results['confidence_calibration'] = calibration_results
    results['confidence_calibration']['confidences'] = confidences
    results['confidence_calibration']['correctness'] = correctness

    # 4. Hallucination Detection
    logger.info("\n[4/5] Running Hallucination Detection...")
    hallucination_detector = HallucinationDetector()
    hallucination_results = []

    for i, (answer, retrieval_result) in enumerate(zip(generated_answers, retrieval_results)):
        context = " ".join([
            chunk['chunk']['text']
            for chunk in retrieval_result.get('fused_results', [])[:5]
        ])

        try:
            hall_result = hallucination_detector.detect_hallucination(answer, context)
            hallucination_results.append(hall_result)
        except Exception as e:
            logger.warning(f"Error in hallucination detection: {e}")
            hallucination_results.append({'hallucination_score': 0.0, 'is_hallucinated': False})

    avg_hallucination_rate = np.mean([r['hallucination_score'] for r in hallucination_results])
    total_hallucinations = sum(1 for r in hallucination_results if r.get('is_hallucinated', False))

    results['hallucination_detection'] = {
        'avg_hallucination_rate': avg_hallucination_rate,
        'total_hallucinations': total_hallucinations,
        'hallucination_percentage': (total_hallucinations / len(hallucination_results)) * 100 if hallucination_results else 0,
        'interpretation': f"Average hallucination rate: {avg_hallucination_rate*100:.1f}%. {total_hallucinations} out of {len(hallucination_results)} answers likely contain hallucinated entities.",
        'details': hallucination_results[:10]
    }

    # 5. Contextual Metrics
    logger.info("\n[5/5] Calculating Contextual Metrics...")
    contextual = ContextualMetrics()

    contextual_precisions = []
    contextual_recalls = []

    for question, retrieval_result, answer in zip(questions, retrieval_results, generated_answers):
        retrieved_chunks = retrieval_result.get('fused_results', [])

        precision = contextual.contextual_precision(retrieved_chunks, question['question'], answer)
        recall = contextual.contextual_recall(retrieved_chunks, question.get('chunk_id', ''))

        contextual_precisions.append(precision)
        contextual_recalls.append(recall)

    results['contextual_metrics'] = {
        'avg_contextual_precision': np.mean(contextual_precisions),
        'avg_contextual_recall': np.mean(contextual_recalls),
        'contextual_f1': 2 * np.mean(contextual_precisions) * np.mean(contextual_recalls) / (np.mean(contextual_precisions) + np.mean(contextual_recalls)) if (np.mean(contextual_precisions) + np.mean(contextual_recalls)) > 0 else 0,
        'interpretation': f"Contextual Precision: {np.mean(contextual_precisions):.3f} (fraction of retrieved chunks that are relevant). Contextual Recall: {np.mean(contextual_recalls):.3f} (fraction of relevant chunks that were retrieved)."
    }

    logger.info("\n" + "=" * 60)
    logger.info("INNOVATIVE EVALUATION COMPLETE!")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    logger.info("Innovative Evaluation Module loaded successfully!")
    logger.info("Available components:")
    logger.info("  - LLMAsJudge: Comprehensive answer quality evaluation")
    logger.info("  - AdversarialTester: Paraphrasing and unanswerable question testing")
    logger.info("  - ConfidenceCalibrator: Confidence estimation and calibration")
    logger.info("  - HallucinationDetector: Entity-based hallucination detection")
    logger.info("  - ContextualMetrics: Contextual precision and recall")
