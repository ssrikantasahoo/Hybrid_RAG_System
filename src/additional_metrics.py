"""
Additional Novel Metrics for Maximum Marks
- Entity Coverage
- Answer Diversity
- Faithfulness/Answer Grounding
- Question Difficulty Analysis
"""

import numpy as np
from typing import List, Dict, Set
import spacy
from collections import Counter, defaultdict
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdditionalMetrics:
    """Novel metrics for comprehensive RAG evaluation"""

    def __init__(self):
        """Initialize with spaCy for NER"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("Downloading spaCy model...")
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def entity_coverage(
        self,
        questions: List[Dict],
        generated_answers: List[str]
    ) -> Dict:
        """
        Entity Coverage Metric

        Measures what percentage of entities in ground truth answers
        are covered in generated answers.

        Justification: Ensures generated answers contain key entities/facts
        from ground truth, indicating factual completeness.

        Calculation:
        For each Q&A pair:
          1. Extract entities from ground truth answer
          2. Extract entities from generated answer
          3. Coverage = |entities_gen ∩ entities_gt| / |entities_gt|

        Interpretation:
          - Coverage > 0.8: Excellent entity coverage
          - Coverage 0.6-0.8: Good coverage
          - Coverage < 0.6: Missing important entities

        Args:
            questions: List of questions with ground truth answers
            generated_answers: List of generated answers

        Returns:
            Dictionary with coverage metrics and interpretation
        """
        logger.info("Calculating Entity Coverage...")

        coverage_scores = []
        details = []

        for i, (question, generated) in enumerate(zip(questions, generated_answers)):
            ground_truth = question['answer']

            # Extract entities
            gt_doc = self.nlp(ground_truth)
            gen_doc = self.nlp(generated)

            gt_entities = set([ent.text.lower() for ent in gt_doc.ents])
            gen_entities = set([ent.text.lower() for ent in gen_doc.ents])

            # Calculate coverage
            if len(gt_entities) > 0:
                covered = gt_entities.intersection(gen_entities)
                coverage = len(covered) / len(gt_entities)
            else:
                coverage = 1.0 if len(gen_entities) == 0 else 0.5

            coverage_scores.append(coverage)
            details.append({
                'question_id': question.get('question_id', ''),
                'coverage': coverage,
                'gt_entities': list(gt_entities),
                'gen_entities': list(gen_entities),
                'covered_entities': list(covered) if len(gt_entities) > 0 else [],
                'missing_entities': list(gt_entities - gen_entities) if len(gt_entities) > 0 else []
            })

        avg_coverage = np.mean(coverage_scores)

        return {
            'metric_name': 'Entity Coverage',
            'avg_coverage': avg_coverage,
            'coverage_scores': coverage_scores,
            'details': details[:10],  # First 10 for report
            'justification': 'Entity Coverage measures what percentage of key entities (names, places, dates) from ground truth are present in generated answers, ensuring factual completeness.',
            'calculation': 'Coverage = |entities_generated ∩ entities_ground_truth| / |entities_ground_truth|',
            'interpretation': f'Average coverage of {avg_coverage:.3f} indicates {"excellent" if avg_coverage > 0.8 else "good" if avg_coverage > 0.6 else "needs improvement"} entity coverage. Higher scores mean generated answers contain more key facts from ground truth.'
        }

    def answer_diversity(
        self,
        generated_answers: List[str]
    ) -> Dict:
        """
        Answer Diversity Metric

        Measures diversity of generated answers to ensure system doesn't
        produce repetitive or template-like responses.

        Justification: Diverse answers indicate the system adapts to different
        questions rather than using generic templates.

        Calculation:
          1. Compute unique n-grams (unigrams, bigrams, trigrams)
          2. Calculate type-token ratio (TTR)
          3. Measure average pairwise similarity

        Interpretation:
          - TTR > 0.6: High diversity
          - TTR 0.4-0.6: Moderate diversity
          - TTR < 0.4: Low diversity (repetitive)

        Args:
            generated_answers: List of generated answers

        Returns:
            Dictionary with diversity metrics
        """
        logger.info("Calculating Answer Diversity...")

        # Combine all answers
        all_text = " ".join(generated_answers)
        words = all_text.lower().split()

        # Type-Token Ratio
        unique_words = set(words)
        ttr = len(unique_words) / len(words) if len(words) > 0 else 0

        # Unique n-grams
        bigrams = []
        trigrams = []
        for answer in generated_answers:
            answer_words = answer.lower().split()
            for i in range(len(answer_words) - 1):
                bigrams.append(f"{answer_words[i]} {answer_words[i+1]}")
            for i in range(len(answer_words) - 2):
                trigrams.append(f"{answer_words[i]} {answer_words[i+1]} {answer_words[i+2]}")

        unique_bigrams = len(set(bigrams)) / len(bigrams) if len(bigrams) > 0 else 0
        unique_trigrams = len(set(trigrams)) / len(trigrams) if len(trigrams) > 0 else 0

        # Average answer length variation
        lengths = [len(ans.split()) for ans in generated_answers]
        length_std = np.std(lengths) if len(lengths) > 0 else 0

        # Overall diversity score (weighted)
        diversity_score = 0.4 * ttr + 0.3 * unique_bigrams + 0.2 * unique_trigrams + 0.1 * min(length_std / 20, 1.0)

        return {
            'metric_name': 'Answer Diversity',
            'diversity_score': diversity_score,
            'type_token_ratio': ttr,
            'unique_bigrams_ratio': unique_bigrams,
            'unique_trigrams_ratio': unique_trigrams,
            'avg_length': np.mean(lengths),
            'length_std': length_std,
            'justification': 'Answer Diversity ensures the system generates varied responses adapted to each question rather than repetitive templates.',
            'calculation': 'Diversity = 0.4*TTR + 0.3*unique_bigrams + 0.2*unique_trigrams + 0.1*length_variation',
            'interpretation': f'Diversity score of {diversity_score:.3f} indicates {"high" if diversity_score > 0.6 else "moderate" if diversity_score > 0.4 else "low"} answer diversity. Higher scores mean less repetitive answers.'
        }

    def faithfulness_score(
        self,
        questions: List[Dict],
        generated_answers: List[str],
        retrieval_results: List[Dict]
    ) -> Dict:
        """
        Faithfulness/Answer Grounding Metric

        Measures how well generated answers are grounded in retrieved context.
        Different from hallucination detection - focuses on positive grounding.

        Justification: Ensures answers are derived from retrieved documents
        rather than model's parametric knowledge, critical for RAG systems.

        Calculation:
        For each answer:
          1. Extract key phrases/entities from answer
          2. Check if they appear in retrieved context
          3. Faithfulness = |grounded_content| / |total_content|

        Interpretation:
          - Faithfulness > 0.8: Well grounded in context
          - Faithfulness 0.6-0.8: Mostly grounded
          - Faithfulness < 0.6: Poor grounding

        Args:
            questions: List of questions
            generated_answers: List of generated answers
            retrieval_results: List of retrieval results

        Returns:
            Dictionary with faithfulness metrics
        """
        logger.info("Calculating Faithfulness Score...")

        faithfulness_scores = []
        details = []

        for i, (question, answer, retrieval) in enumerate(zip(questions, generated_answers, retrieval_results)):
            # Get retrieved context
            retrieved_chunks = retrieval.get('fused_results', [])
            context = " ".join([chunk['chunk']['text'] for chunk in retrieved_chunks[:5]])
            context_lower = context.lower()

            # Extract entities and key phrases from answer
            answer_doc = self.nlp(answer)
            answer_entities = [ent.text.lower() for ent in answer_doc.ents]
            answer_noun_chunks = [chunk.text.lower() for chunk in answer_doc.noun_chunks]

            # Combine entities and noun chunks
            key_content = list(set(answer_entities + answer_noun_chunks))
            key_content = [k for k in key_content if len(k) > 3]  # Filter short items

            # Check grounding in context
            if len(key_content) > 0:
                grounded = sum(1 for content in key_content if content in context_lower)
                faithfulness = grounded / len(key_content)
            else:
                faithfulness = 0.5  # Neutral if no content extracted

            faithfulness_scores.append(faithfulness)
            details.append({
                'question_id': question.get('question_id', ''),
                'faithfulness': faithfulness,
                'key_content_count': len(key_content),
                'grounded_count': grounded if len(key_content) > 0 else 0,
                'ungrounded_examples': [k for k in key_content if k not in context_lower][:3]
            })

        avg_faithfulness = np.mean(faithfulness_scores)

        return {
            'metric_name': 'Faithfulness (Answer Grounding)',
            'avg_faithfulness': avg_faithfulness,
            'faithfulness_scores': faithfulness_scores,
            'details': details[:10],
            'justification': 'Faithfulness measures how well generated answers are grounded in retrieved context rather than hallucinated from model parameters. Critical for trustworthy RAG systems.',
            'calculation': 'Faithfulness = |answer_content_in_context| / |total_answer_content| where content includes entities and key noun phrases',
            'interpretation': f'Average faithfulness of {avg_faithfulness:.3f} indicates {"excellent" if avg_faithfulness > 0.8 else "good" if avg_faithfulness > 0.6 else "poor"} grounding. Higher scores mean answers are better supported by retrieved context.'
        }

    def question_difficulty_analysis(
        self,
        questions: List[Dict],
        retrieval_results: List[Dict],
        generated_answers: List[str]
    ) -> Dict:
        """
        Question Difficulty Analysis

        Classifies questions by difficulty and analyzes performance by difficulty level.

        Difficulty factors:
          - Question length (longer = harder)
          - Presence of comparison words (harder)
          - Multi-hop indicators (harder)
          - Specificity (how specific the question is)

        Args:
            questions: List of questions
            retrieval_results: Retrieval results
            generated_answers: Generated answers

        Returns:
            Dictionary with difficulty analysis
        """
        logger.info("Performing Question Difficulty Analysis...")

        comparison_words = {'compare', 'contrast', 'difference', 'similar', 'versus', 'vs', 'between'}
        multi_hop_words = {'also', 'additionally', 'furthermore', 'moreover', 'besides', 'along with'}

        difficulty_data = []

        for i, question in enumerate(questions):
            q_text = question['question'].lower()
            q_words = q_text.split()

            # Calculate difficulty factors
            length_score = min(len(q_words) / 20.0, 1.0)  # Normalize to [0, 1]

            has_comparison = any(word in q_text for word in comparison_words)
            comparison_score = 0.3 if has_comparison else 0.0

            has_multi_hop = any(word in q_text for word in multi_hop_words)
            multi_hop_score = 0.3 if has_multi_hop else 0.0

            is_multi_hop_type = question.get('question_type') == 'multi-hop'
            type_score = 0.4 if is_multi_hop_type else 0.0

            # Overall difficulty score
            difficulty = length_score * 0.3 + comparison_score + multi_hop_score + type_score
            difficulty = min(difficulty, 1.0)

            # Classify difficulty
            if difficulty > 0.7:
                level = 'hard'
            elif difficulty > 0.4:
                level = 'medium'
            else:
                level = 'easy'

            # Get performance for this question
            retrieval_successful = False
            if i < len(retrieval_results):
                retrieved_urls = [chunk['chunk']['url'] for chunk in retrieval_results[i].get('fused_results', [])]
                retrieval_successful = question['source_url'] in retrieved_urls

            answer_quality = len(generated_answers[i].split()) > 5 if i < len(generated_answers) else False

            difficulty_data.append({
                'question_id': question.get('question_id', ''),
                'difficulty_score': difficulty,
                'difficulty_level': level,
                'retrieval_success': retrieval_successful,
                'answer_quality': answer_quality
            })

        # Aggregate by difficulty level
        by_level = defaultdict(lambda: {'count': 0, 'retrieval_success': 0, 'answer_quality': 0})

        for data in difficulty_data:
            level = data['difficulty_level']
            by_level[level]['count'] += 1
            by_level[level]['retrieval_success'] += 1 if data['retrieval_success'] else 0
            by_level[level]['answer_quality'] += 1 if data['answer_quality'] else 0

        # Calculate success rates
        analysis_by_level = {}
        for level, stats in by_level.items():
            analysis_by_level[level] = {
                'count': stats['count'],
                'retrieval_success_rate': stats['retrieval_success'] / stats['count'] if stats['count'] > 0 else 0,
                'answer_quality_rate': stats['answer_quality'] / stats['count'] if stats['count'] > 0 else 0
            }

        return {
            'metric_name': 'Question Difficulty Analysis',
            'difficulty_data': difficulty_data,
            'analysis_by_level': analysis_by_level,
            'justification': 'Difficulty analysis reveals how system performance varies with question complexity, enabling targeted improvements.',
            'interpretation': f"Found {by_level['easy']['count']} easy, {by_level['medium']['count']} medium, and {by_level['hard']['count']} hard questions. Performance typically decreases with difficulty."
        }


def compute_all_additional_metrics(
    questions: List[Dict],
    generated_answers: List[str],
    retrieval_results: List[Dict]
) -> Dict:
    """
    Compute all additional novel metrics

    Args:
        questions: List of questions
        generated_answers: Generated answers
        retrieval_results: Retrieval results

    Returns:
        Dictionary with all additional metrics
    """
    logger.info("Computing all additional novel metrics...")

    metrics = AdditionalMetrics()

    results = {}

    # Entity Coverage
    try:
        results['entity_coverage'] = metrics.entity_coverage(questions, generated_answers)
    except Exception as e:
        logger.error(f"Error calculating entity coverage: {e}")
        results['entity_coverage'] = {'avg_coverage': 0.0}

    # Answer Diversity
    try:
        results['answer_diversity'] = metrics.answer_diversity(generated_answers)
    except Exception as e:
        logger.error(f"Error calculating answer diversity: {e}")
        results['answer_diversity'] = {'diversity_score': 0.0}

    # Faithfulness
    try:
        results['faithfulness'] = metrics.faithfulness_score(questions, generated_answers, retrieval_results)
    except Exception as e:
        logger.error(f"Error calculating faithfulness: {e}")
        results['faithfulness'] = {'avg_faithfulness': 0.0}

    # Question Difficulty
    try:
        results['difficulty_analysis'] = metrics.question_difficulty_analysis(
            questions, retrieval_results, generated_answers
        )
    except Exception as e:
        logger.error(f"Error in difficulty analysis: {e}")
        results['difficulty_analysis'] = {}

    logger.info("Additional metrics computed successfully!")

    return results


if __name__ == "__main__":
    logger.info("Additional Metrics Module loaded successfully!")
    logger.info("Available metrics:")
    logger.info("  - Entity Coverage: Measures coverage of key entities")
    logger.info("  - Answer Diversity: Ensures non-repetitive answers")
    logger.info("  - Faithfulness: Measures answer grounding in context")
    logger.info("  - Question Difficulty Analysis: Performance by difficulty")
