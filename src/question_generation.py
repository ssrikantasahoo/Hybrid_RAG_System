"""
Question Generation Module
Generates evaluation questions from Wikipedia corpus
"""

import json
import random
from typing import List, Dict, Tuple
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Generates questions from text corpus"""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Initialize question generator

        Args:
            model_name: Model for question generation
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading question generation model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded on {self.device}")

    def generate_factual_question(self, text: str, context: Dict) -> Dict:
        """Generate factual question from text"""

        prompt = f"""Generate a specific factual question that can be answered using the following text. The question should ask about facts, names, dates, or specific details.

Text: {text}

Question:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                top_p=0.9,
                do_sample=True
            )

        question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Generate answer
        answer = self.generate_answer(question, text)

        return {
            'question': question.strip(),
            'answer': answer,
            'question_type': 'factual',
            'source_url': context['url'],
            'source_title': context['title'],
            'chunk_id': context.get('chunk_id', ''),
            'context_text': text
        }

    def generate_comparative_question(self, text: str, context: Dict) -> Dict:
        """Generate comparative question"""

        prompt = f"""Generate a question that compares or contrasts concepts, ideas, or entities mentioned in the following text.

Text: {text}

Question:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                top_p=0.9,
                do_sample=True
            )

        question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = self.generate_answer(question, text)

        return {
            'question': question.strip(),
            'answer': answer,
            'question_type': 'comparative',
            'source_url': context['url'],
            'source_title': context['title'],
            'chunk_id': context.get('chunk_id', ''),
            'context_text': text
        }

    def generate_inferential_question(self, text: str, context: Dict) -> Dict:
        """Generate inferential question that requires reasoning"""

        prompt = f"""Generate a question that requires inference or reasoning based on the following text. The answer should not be explicitly stated but can be inferred.

Text: {text}

Question:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                top_p=0.9,
                do_sample=True
            )

        question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = self.generate_answer(question, text)

        return {
            'question': question.strip(),
            'answer': answer,
            'question_type': 'inferential',
            'source_url': context['url'],
            'source_title': context['title'],
            'chunk_id': context.get('chunk_id', ''),
            'context_text': text
        }

    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer for a question given context"""

        prompt = f"""Answer the following question based on the context. Be concise.

Context: {context}

Question: {question}

Answer:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

    def generate_questions_from_chunks(
        self,
        chunks: List[Dict],
        num_questions: int = 100,
        distribution: Dict = None
    ) -> List[Dict]:
        """
        Generate questions from chunks

        Args:
            chunks: List of text chunks
            num_questions: Total number of questions to generate
            distribution: Distribution of question types

        Returns:
            List of question dictionaries
        """
        if distribution is None:
            distribution = {
                'factual': 0.4,
                'comparative': 0.3,
                'inferential': 0.3
            }

        # Calculate number of questions per type
        num_factual = int(num_questions * distribution['factual'])
        num_comparative = int(num_questions * distribution['comparative'])
        num_inferential = num_questions - num_factual - num_comparative

        logger.info(f"Generating {num_questions} questions:")
        logger.info(f"  - Factual: {num_factual}")
        logger.info(f"  - Comparative: {num_comparative}")
        logger.info(f"  - Inferential: {num_inferential}")

        questions = []

        # Select random chunks for question generation
        # Ensure we have enough chunks with sufficient text
        valid_chunks = [
            chunk for chunk in chunks
            if len(chunk['text'].split()) >= 50  # At least 50 words
        ]

        if len(valid_chunks) < num_questions:
            logger.warning(f"Only {len(valid_chunks)} valid chunks available for {num_questions} questions")

        random.shuffle(valid_chunks)

        # Generate factual questions
        logger.info("Generating factual questions...")
        for i in tqdm(range(num_factual), desc="Factual"):
            if i < len(valid_chunks):
                chunk = valid_chunks[i]
                try:
                    qa = self.generate_factual_question(chunk['text'], chunk)
                    qa['question_id'] = f"Q{len(questions) + 1:03d}"
                    questions.append(qa)
                except Exception as e:
                    logger.warning(f"Error generating factual question: {e}")

        # Generate comparative questions
        logger.info("Generating comparative questions...")
        for i in tqdm(range(num_comparative), desc="Comparative"):
            idx = num_factual + i
            if idx < len(valid_chunks):
                chunk = valid_chunks[idx]
                try:
                    qa = self.generate_comparative_question(chunk['text'], chunk)
                    qa['question_id'] = f"Q{len(questions) + 1:03d}"
                    questions.append(qa)
                except Exception as e:
                    logger.warning(f"Error generating comparative question: {e}")

        # Generate inferential questions
        logger.info("Generating inferential questions...")
        for i in tqdm(range(num_inferential), desc="Inferential"):
            idx = num_factual + num_comparative + i
            if idx < len(valid_chunks):
                chunk = valid_chunks[idx]
                try:
                    qa = self.generate_inferential_question(chunk['text'], chunk)
                    qa['question_id'] = f"Q{len(questions) + 1:03d}"
                    questions.append(qa)
                except Exception as e:
                    logger.warning(f"Error generating inferential question: {e}")

        logger.info(f"Generated {len(questions)} questions successfully")

        return questions

    def save_questions(self, questions: List[Dict], output_file: str):
        """Save questions to JSON file"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(questions)} questions to {output_file}")

    def load_questions(self, file_path: str) -> List[Dict]:
        """Load questions from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def create_multi_hop_questions_enhanced(
    chunks: List[Dict],
    num_questions: int = 10,
    model_name: str = "google/flan-t5-base"
) -> List[Dict]:
    """
    Create ENHANCED multi-hop questions using LLM

    Multi-hop questions require reasoning across multiple chunks/sections.
    This uses an LLM to generate realistic multi-hop questions.

    Args:
        chunks: List of chunks
        num_questions: Number of multi-hop questions
        model_name: Model for generation

    Returns:
        List of multi-hop question dictionaries
    """
    logger.info(f"Creating {num_questions} ENHANCED multi-hop questions with LLM...")

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    questions = []

    # Group chunks by title
    chunks_by_title = {}
    for chunk in chunks:
        title = chunk['title']
        if title not in chunks_by_title:
            chunks_by_title[title] = []
        chunks_by_title[title].append(chunk)

    # Find articles with multiple chunks
    multi_chunk_articles = {
        title: chunks_list
        for title, chunks_list in chunks_by_title.items()
        if len(chunks_list) >= 2
    }

    if len(multi_chunk_articles) == 0:
        logger.warning("No articles with multiple chunks found for multi-hop questions")
        return []

    articles_list = list(multi_chunk_articles.keys())
    random.shuffle(articles_list)

    for i in range(min(num_questions, len(articles_list))):
        title = articles_list[i]
        article_chunks = multi_chunk_articles[title]

        # Select two chunks
        chunk1, chunk2 = random.sample(article_chunks, 2)

        # Combine chunks
        combined_text = chunk1['text'][:300] + " " + chunk2['text'][:300]

        # Generate multi-hop question using LLM
        prompt = f"""Generate a question that requires information from BOTH of these text segments to answer:

Segment 1: {chunk1['text'][:200]}

Segment 2: {chunk2['text'][:200]}

Generate a complex question that cannot be answered using only one segment:"""

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.9,
                    top_p=0.95,
                    do_sample=True
                )

            question_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Generate answer
            answer_prompt = f"""Answer this question using the following information:

{combined_text}

Question: {question_text}

Answer:"""

            answer_inputs = tokenizer(answer_prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

            with torch.no_grad():
                answer_outputs = model.generate(
                    **answer_inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True
                )

            answer_text = tokenizer.decode(answer_outputs[0], skip_special_tokens=True)

            questions.append({
                'question_id': f"MH{i + 1:03d}",
                'question': question_text,
                'answer': answer_text,
                'question_type': 'multi-hop',
                'source_url': chunk1['url'],
                'source_title': title,
                'required_chunks': [chunk1['chunk_id'], chunk2['chunk_id']],
                'context_text': combined_text,
                'multi_hop_complexity': 'high'
            })

            logger.info(f"Created multi-hop question {i+1}/{num_questions}")

        except Exception as e:
            logger.warning(f"Error generating multi-hop question: {e}")
            # Fallback to simple template
            questions.append({
                'question_id': f"MH{i + 1:03d}",
                'question': f"How do the different aspects of {title} discussed in various sections relate to each other?",
                'answer': f"Answer requires synthesizing information from multiple sections of {title}",
                'question_type': 'multi-hop',
                'source_url': chunk1['url'],
                'source_title': title,
                'required_chunks': [chunk1['chunk_id'], chunk2['chunk_id']],
                'context_text': combined_text,
                'multi_hop_complexity': 'medium'
            })

    logger.info(f"Created {len(questions)} enhanced multi-hop questions")

    return questions


def create_multi_hop_questions(chunks: List[Dict], num_questions: int = 10) -> List[Dict]:
    """
    Create multi-hop questions (backward compatibility wrapper)

    Calls enhanced version for better quality.

    Args:
        chunks: List of chunks
        num_questions: Number of multi-hop questions

    Returns:
        List of multi-hop question dictionaries
    """
    return create_multi_hop_questions_enhanced(chunks, num_questions)


if __name__ == "__main__":
    # Example usage
    # Load chunks
    with open("data/processed_chunks.json", 'r') as f:
        chunks = json.load(f)

    # Generate questions
    generator = QuestionGenerator()
    questions = generator.generate_questions_from_chunks(chunks, num_questions=90)

    # Add multi-hop questions
    multi_hop_questions = create_multi_hop_questions(chunks, num_questions=10)
    questions.extend(multi_hop_questions)

    # Save questions
    generator.save_questions(questions, "data/questions.json")

    print(f"\nGenerated {len(questions)} questions:")
    print(f"  - Regular: {len(questions) - len(multi_hop_questions)}")
    print(f"  - Multi-hop: {len(multi_hop_questions)}")
