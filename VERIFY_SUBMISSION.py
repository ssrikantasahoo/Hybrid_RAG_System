#!/usr/bin/env python3
"""
SUBMISSION VERIFICATION SCRIPT

Checks all required deliverables before submission.
Run this AFTER running RUN_COMPLETE_SYSTEM.py

Usage: python VERIFY_SUBMISSION.py
"""

import json
from pathlib import Path
import sys

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def check_file(filepath, min_size=None):
    """Check if file exists and optionally meets size requirement"""
    path = Path(filepath)
    if not path.exists():
        return False, "DOES NOT EXIST"

    if path.is_file():
        size = path.stat().st_size
        if min_size and size < min_size:
            return False, f"EXISTS but too small ({size} bytes < {min_size} bytes)"
        return True, f"EXISTS ({size:,} bytes)"
    else:
        # Directory
        return True, "EXISTS (directory)"

def verify_dataset():
    """Verify dataset requirements"""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}DATASET REQUIREMENTS{Colors.END}")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}")

    checks = []

    # Fixed URLs
    exists, msg = check_file('data/fixed_urls.json', min_size=1000)
    checks.append(('D1', 'Fixed 200 Wikipedia URLs (fixed_urls.json)', exists))
    if exists:
        with open('data/fixed_urls.json', 'r') as f:
            urls = json.load(f)
        print(f"{'✓' if len(urls) == 200 else '✗'} D1: Fixed URLs - {len(urls)} URLs {msg}")
        if len(urls) != 200:
            print(f"   {Colors.RED}WARNING: Expected 200 URLs, found {len(urls)}{Colors.END}")
            checks[-1] = ('D1', checks[-1][1], False)
    else:
        print(f"✗ D1: Fixed URLs - {Colors.RED}{msg}{Colors.END}")

    # Processed chunks (verifies chunking)
    exists, msg = check_file('data/processed_chunks.json', min_size=10000)
    checks.append(('D4', 'Text chunking (200-400 tokens, 50 overlap)', exists))
    if exists:
        with open('data/processed_chunks.json', 'r') as f:
            chunks = json.load(f)
        print(f"✓ D4: Chunking - {len(chunks)} chunks {msg}")
        # Verify chunk structure
        if chunks:
            sample = chunks[0]
            has_metadata = all(k in sample for k in ['url', 'title', 'chunk_id'])
            if has_metadata:
                print(f"✓ D5: Metadata (URL, title, chunk_id) present")
                checks.append(('D5', 'Chunk metadata', True))
            else:
                print(f"✗ D5: {Colors.RED}Missing metadata{Colors.END}")
                checks.append(('D5', 'Chunk metadata', False))
    else:
        print(f"✗ D4: Chunking - {Colors.RED}{msg}{Colors.END}")
        checks.append(('D5', 'Chunk metadata', False))

    return checks

def verify_part1():
    """Verify Part 1: Hybrid RAG System"""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}PART 1: HYBRID RAG SYSTEM (10 Marks){Colors.END}")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}")

    checks = []

    # Dense retrieval
    exists, msg = check_file('data/vector_index')
    checks.append(('1.1', 'Dense Vector Retrieval (FAISS index)', exists))
    print(f"{'✓' if exists else '✗'} 1.1: Dense Retrieval - {msg}")

    # Sparse retrieval
    exists, msg = check_file('data/bm25_index.pkl', min_size=1000)
    checks.append(('1.2', 'Sparse BM25 Retrieval', exists))
    print(f"{'✓' if exists else '✗'} 1.2: Sparse Retrieval - {msg}")

    # RRF (verified by evaluation results)
    exists, msg = check_file('outputs/evaluation_results.json')
    if exists:
        with open('outputs/evaluation_results.json', 'r') as f:
            results = json.load(f)
        has_ablation = 'ablation_study' in results
        has_hybrid = has_ablation and 'hybrid_rrf' in results.get('ablation_study', {})
        checks.append(('1.3', 'RRF Implementation', has_hybrid))
        print(f"{'✓' if has_hybrid else '✗'} 1.3: RRF - {'Verified in ablation study' if has_hybrid else 'Not verified'}")
    else:
        checks.append(('1.3', 'RRF Implementation', False))
        print(f"✗ 1.3: RRF - Cannot verify (no evaluation results)")

    # Generation (verified by answers)
    if exists:
        has_answers = 'generated_answers' in results and len(results['generated_answers']) > 0
        checks.append(('1.4', 'Response Generation', has_answers))
        print(f"{'✓' if has_answers else '✗'} 1.4: Generation - {len(results.get('generated_answers', []))} answers generated")
    else:
        checks.append(('1.4', 'Response Generation', False))
        print(f"✗ 1.4: Generation - Cannot verify")

    # UI (code check)
    ui_exists = Path('src/streamlit_app.py').exists()
    checks.append(('1.5', 'User Interface', ui_exists))
    print(f"{'✓' if ui_exists else '✗'} 1.5: UI - Code exists (needs manual testing)")
    if ui_exists:
        print(f"   {Colors.YELLOW}ACTION REQUIRED: Launch UI and take screenshots{Colors.END}")
        print(f"   Run: streamlit run src/streamlit_app.py")

    return checks

def verify_part2():
    """Verify Part 2: Automated Evaluation"""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}PART 2: AUTOMATED EVALUATION (10 Marks){Colors.END}")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}")

    checks = []

    # Question generation
    exists, msg = check_file('data/questions.json', min_size=10000)
    if exists:
        with open('data/questions.json', 'r') as f:
            questions = json.load(f)
        num_q = len(questions)
        checks.append(('2.1', 'Question Generation (100 Q&A)', num_q == 100))
        print(f"{'✓' if num_q == 100 else '✗'} 2.1: Questions - {num_q} generated {msg}")

        # Count by type
        types = {}
        for q in questions:
            qtype = q.get('question_type', 'unknown')
            types[qtype] = types.get(qtype, 0) + 1
        print(f"   Types: {types}")
    else:
        checks.append(('2.1', 'Question Generation', False))
        print(f"✗ 2.1: Questions - {Colors.RED}{msg}{Colors.END}")

    # Evaluation results
    exists, msg = check_file('outputs/evaluation_results.json', min_size=10000)
    if exists:
        with open('outputs/evaluation_results.json', 'r') as f:
            results = json.load(f)

        # MRR
        has_mrr = 'metrics' in results and 'mrr' in results['metrics']
        checks.append(('2.2.1', 'MRR at URL Level (Mandatory)', has_mrr))
        if has_mrr:
            mrr_value = results['metrics']['mrr'].get('mrr', 0)
            print(f"✓ 2.2.1: MRR - {mrr_value:.4f}")
        else:
            print(f"✗ 2.2.1: MRR - {Colors.RED}Missing{Colors.END}")

        # NDCG
        has_ndcg = 'metrics' in results and 'ndcg' in results['metrics']
        checks.append(('2.2.2', 'NDCG@K (Custom Metric 1)', has_ndcg))
        if has_ndcg:
            ndcg_value = results['metrics']['ndcg'].get('ndcg', 0)
            has_justification = 'justification' in results['metrics']['ndcg']
            print(f"✓ 2.2.2: NDCG - {ndcg_value:.4f} (Justification: {'✓' if has_justification else '✗'})")
        else:
            print(f"✗ 2.2.2: NDCG - {Colors.RED}Missing{Colors.END}")

        # BERTScore
        has_bert = 'metrics' in results and 'bert_score' in results['metrics']
        checks.append(('2.2.3', 'BERTScore (Custom Metric 2)', has_bert))
        if has_bert:
            bert_value = results['metrics']['bert_score'].get('bert_score_f1', 0)
            has_justification = 'justification' in results['metrics']['bert_score']
            print(f"✓ 2.2.3: BERTScore - {bert_value:.4f} (Justification: {'✓' if has_justification else '✗'})")
        else:
            print(f"✗ 2.2.3: BERTScore - {Colors.RED}Missing{Colors.END}")

        # Innovation
        innovation_count = 0
        if 'innovative_metrics' in results:
            innovation_count += len(results['innovative_metrics'])
        if 'additional_metrics' in results:
            innovation_count += len(results['additional_metrics'])
        if 'ablation_study' in results:
            innovation_count += 1
        if 'error_analysis' in results:
            innovation_count += 1

        checks.append(('2.3', 'Innovative Evaluation', innovation_count >= 4))
        print(f"{'✓' if innovation_count >= 4 else '✗'} 2.3: Innovation - {innovation_count} techniques implemented")

        # Ablation
        has_ablation = 'ablation_study' in results
        print(f"{'✓' if has_ablation else '✗'}   - Ablation Study: {'Present' if has_ablation else 'Missing'}")

        # Error analysis
        has_error = 'error_analysis' in results
        print(f"{'✓' if has_error else '✗'}   - Error Analysis: {'Present' if has_error else 'Missing'}")

    else:
        print(f"✗ 2.2.1-2.3: {Colors.RED}Evaluation results missing{Colors.END}")
        checks.append(('2.2.1', 'MRR', False))
        checks.append(('2.2.2', 'NDCG', False))
        checks.append(('2.2.3', 'BERTScore', False))
        checks.append(('2.3', 'Innovation', False))

    # PDF Report
    exists, msg = check_file('outputs/evaluation_report.pdf', min_size=10000)
    checks.append(('2.5', 'PDF Report', exists))
    print(f"{'✓' if exists else '✗'} 2.5: PDF Report - {msg}")

    return checks

def verify_visualizations():
    """Verify visualization files"""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}VISUALIZATIONS{Colors.END}")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}")

    required_viz = [
        'ablation_comparison.png',
        'response_time_distribution.png',
        'error_analysis.png',
        'llm_judge_radar.png',
        'calibration_curve.png',
        'hallucination_analysis.png',
        'adversarial_testing.png',
        'comprehensive_dashboard.png',
        'parameter_sweep_results.png'
    ]

    found = 0
    for viz in required_viz:
        exists, msg = check_file(f'outputs/{viz}', min_size=1000)
        if exists:
            print(f"✓ {viz}")
            found += 1
        else:
            print(f"✗ {viz} - {Colors.RED}{msg}{Colors.END}")

    print(f"\n{'✓' if found >= 9 else '✗'} Total: {found}/9 visualizations")
    return found >= 9

def verify_submission_package():
    """Verify submission package requirements"""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}SUBMISSION PACKAGE REQUIREMENTS{Colors.END}")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}")

    required = {
        'Code': [
            'src/data_collection.py',
            'src/preprocessing.py',
            'src/dense_retrieval.py',
            'src/sparse_retrieval.py',
            'src/rrf_fusion.py',
            'src/llm_generation.py',
            'src/question_generation.py',
            'src/evaluation_metrics.py',
            'src/evaluation_pipeline.py',
            'src/streamlit_app.py'
        ],
        'Data': [
            'data/fixed_urls.json',
            'data/questions.json'
        ],
        'Outputs': [
            'outputs/evaluation_results.json',
            'outputs/results_table.csv',
            'outputs/evaluation_report.pdf'
        ],
        'Documentation': [
            'README.md',
            'requirements.txt',
            'config.yaml'
        ]
    }

    all_exist = True
    for category, files in required.items():
        print(f"\n{category}:")
        for filepath in files:
            exists, msg = check_file(filepath)
            if exists:
                print(f"  ✓ {filepath}")
            else:
                print(f"  ✗ {filepath} - {Colors.RED}{msg}{Colors.END}")
                all_exist = False

    return all_exist

def calculate_score(all_checks):
    """Calculate estimated score"""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}ESTIMATED SCORE{Colors.END}")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}")

    # Count passes
    passed = sum(1 for _, _, status in all_checks if status)
    total = len(all_checks)

    print(f"Requirements passed: {passed}/{total}")

    # Estimate score
    score_part1 = 0
    part1_checks = [c for c in all_checks if c[0].startswith('1.')]
    if all(c[2] for c in part1_checks):
        score_part1 = 10
    else:
        score_part1 = sum(2 for c in part1_checks if c[2])

    score_part2 = 0
    # MRR, NDCG, BERTScore = 2 marks each
    for check_id in ['2.2.1', '2.2.2', '2.2.3']:
        if any(c[0] == check_id and c[2] for c in all_checks):
            score_part2 += 2

    # Innovation = 4 marks
    if any(c[0] == '2.3' and c[2] for c in all_checks):
        score_part2 += 4

    total_score = score_part1 + score_part2

    print(f"\nPart 1 (Hybrid RAG): {score_part1}/10")
    print(f"Part 2 (Evaluation): {score_part2}/10")
    print(f"\n{Colors.GREEN if total_score == 20 else Colors.YELLOW}TOTAL ESTIMATED SCORE: {total_score}/20{Colors.END}")

    return total_score

def main():
    """Main verification function"""
    print(f"\n{Colors.GREEN}{'='*70}{Colors.END}")
    print(f"{Colors.GREEN}HYBRID RAG SYSTEM - SUBMISSION VERIFICATION{Colors.END}")
    print(f"{Colors.GREEN}{'='*70}{Colors.END}")

    all_checks = []

    # Verify each section
    all_checks.extend(verify_dataset())
    all_checks.extend(verify_part1())
    all_checks.extend(verify_part2())
    viz_ok = verify_visualizations()
    package_ok = verify_submission_package()

    # Calculate score
    score = calculate_score(all_checks)

    # Final verdict
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}FINAL VERDICT{Colors.END}")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}")

    if score == 20 and viz_ok and package_ok:
        print(f"{Colors.GREEN}✓✓✓ READY FOR SUBMISSION - 100% COMPLETE ✓✓✓{Colors.END}")
        print(f"\nNext steps:")
        print(f"1. Take UI screenshots: streamlit run src/streamlit_app.py")
        print(f"2. Take dashboard screenshots: streamlit run src/evaluation_dashboard.py")
        print(f"3. Create submission ZIP")
    elif score >= 17:
        print(f"{Colors.YELLOW}⚠️ ALMOST READY - Minor issues to fix ⚠️{Colors.END}")
        print(f"\nMissing requirements:")
        for check_id, desc, status in all_checks:
            if not status:
                print(f"  ✗ {check_id}: {desc}")
    else:
        print(f"{Colors.RED}✗ NOT READY - Critical issues ✗{Colors.END}")
        print(f"\nRun: python RUN_COMPLETE_SYSTEM.py")

    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}\n")

    return score == 20

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
