"""
Interactive Evaluation Dashboard

Separate Streamlit app for viewing evaluation results interactively.
Shows real-time metrics, question breakdowns, retrieval visualizations, and method comparisons.

This demonstrates innovation through interactive dashboard as mentioned in assignment.

Run with: streamlit run src/evaluation_dashboard.py
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

# Page config
st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_results(results_file):
    """Load evaluation results"""
    try:
        path = Path(results_file)
        # Verify file exists, if not try to resolve relative to project root
        if not path.exists():
            # Try resolving relative to this script's parent (src) -> parent (root)
            root_path = Path(__file__).parent.parent / results_file
            if root_path.exists():
                path = root_path
            else:
                # Try assuming results_file is just the filename and it's in root/outputs
                alt_path = Path(__file__).parent.parent / "outputs" / path.name
                if alt_path.exists():
                    path = alt_path
        
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None


def create_radar_chart(llm_judge_data):
    """Create radar chart for LLM-as-Judge scores"""
    categories = ['Factual<br>Accuracy', 'Completeness', 'Relevance', 'Coherence']
    values = [
        llm_judge_data.get('avg_factual_accuracy', 0),
        llm_judge_data.get('avg_completeness', 0),
        llm_judge_data.get('avg_relevance', 0),
        llm_judge_data.get('avg_coherence', 0)
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='RAG System',
        line_color='#2E86AB'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title="LLM-as-Judge Quality Assessment",
        height=400
    )

    return fig


def create_metrics_comparison(results):
    """Create comparison of all main metrics"""
    metrics_data = results.get('metrics', {})

    metric_names = ['MRR', 'NDCG@5', 'BERTScore F1', 'Precision@5', 'Recall@5']
    values = [
        metrics_data.get('mrr', {}).get('mrr', 0),
        metrics_data.get('ndcg', {}).get('ndcg', 0),
        metrics_data.get('bert_score', {}).get('bert_score_f1', 0),
        metrics_data.get('precision_at_k', {}).get('precision_at_k', 0),
        metrics_data.get('recall_at_k', {}).get('recall_at_k', 0)
    ]

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metric_names,
        y=values,
        marker_color=colors,
        text=[f'{v:.3f}' for v in values],
        textposition='outside'
    ))

    fig.update_layout(
        title="Core Evaluation Metrics",
        yaxis_title="Score",
        yaxis_range=[0, 1.1],
        height=400
    )

    return fig


def create_ablation_chart(ablation_data):
    """Create ablation study comparison"""
    methods = ['Dense Only', 'Sparse Only', 'Hybrid (RRF)']
    mrr_values = [
        ablation_data.get('dense_only', {}).get('mrr', 0),
        ablation_data.get('sparse_only', {}).get('mrr', 0),
        ablation_data.get('hybrid_rrf', {}).get('mrr', 0)
    ]
    ndcg_values = [
        ablation_data.get('dense_only', {}).get('ndcg', 0),
        ablation_data.get('sparse_only', {}).get('ndcg', 0),
        ablation_data.get('hybrid_rrf', {}).get('ndcg', 0)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='MRR', x=methods, y=mrr_values, marker_color='#2E86AB'))
    fig.add_trace(go.Bar(name='NDCG', x=methods, y=ndcg_values, marker_color='#A23B72'))

    fig.update_layout(
        title="Ablation Study: Dense vs Sparse vs Hybrid",
        yaxis_title="Score",
        barmode='group',
        height=400
    )

    return fig


def create_difficulty_chart(difficulty_data):
    """Create question difficulty analysis chart"""
    analysis = difficulty_data.get('analysis_by_level', {})

    levels = list(analysis.keys())
    retrieval_rates = [analysis[level]['retrieval_success_rate'] * 100 for level in levels]
    answer_rates = [analysis[level]['answer_quality_rate'] * 100 for level in levels]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Retrieval Success', x=levels, y=retrieval_rates, marker_color='#4ECDC4'))
    fig.add_trace(go.Bar(name='Answer Quality', x=levels, y=answer_rates, marker_color='#F18F01'))

    fig.update_layout(
        title="Performance by Question Difficulty",
        yaxis_title="Success Rate (%)",
        barmode='group',
        height=400
    )

    return fig


def main():
    """Main dashboard function"""

    # Header
    st.title("📊 Hybrid RAG System - Evaluation Dashboard")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        results_file = st.text_input(
            "Results File Path",
            value="outputs/evaluation_results.json"
        )

        if st.button("🔄 Reload Results"):
            st.cache_data.clear()

        st.markdown("---")
        st.header("📖 Navigation")
        page = st.radio(
            "Select Page",
            ["Overview", "Detailed Metrics", "Innovative Evaluation", "Question Analysis", "Parameter Sweep"]
        )

    # Load results
    results = load_results(results_file)

    if results is None:
        st.error("⚠️ Could not load results. Please check the file path.")
        st.info("Make sure you've run the evaluation first: `python build_system.py`")
        return

    # PAGE: Overview
    if page == "Overview":
        st.header("📈 Evaluation Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Questions",
                results.get('num_questions', 0)
            )

        with col2:
            st.metric(
                "Avg Response Time",
                f"{results.get('avg_response_time', 0):.2f}s"
            )

        with col3:
            mrr = results.get('metrics', {}).get('mrr', {}).get('mrr', 0)
            st.metric(
                "MRR Score",
                f"{mrr:.4f}"
            )

        with col4:
            ndcg = results.get('metrics', {}).get('ndcg', {}).get('ndcg', 0)
            st.metric(
                "NDCG@5",
                f"{ndcg:.4f}"
            )

        st.markdown("---")

        # Main metrics chart
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                create_metrics_comparison(results),
                use_container_width=True
            )

        with col2:
            ablation = results.get('ablation_study', {})
            if ablation:
                st.plotly_chart(
                    create_ablation_chart(ablation),
                    use_container_width=True
                )

    # PAGE: Detailed Metrics
    elif page == "Detailed Metrics":
        st.header("📊 Detailed Metrics")

        metrics_data = results.get('metrics', {})

        # MRR
        with st.expander("📍 Mean Reciprocal Rank (MRR) - Mandatory Metric", expanded=True):
            mrr_data = metrics_data.get('mrr', {})
            st.metric("MRR Score", f"{mrr_data.get('mrr', 0):.4f}")
            st.info(mrr_data.get('interpretation', ''))

        # NDCG
        with st.expander("📊 NDCG@K - Custom Metric 1", expanded=True):
            ndcg_data = metrics_data.get('ndcg', {})
            st.metric("NDCG@5 Score", f"{ndcg_data.get('ndcg', 0):.4f}")
            st.info(ndcg_data.get('justification', ''))
            st.caption(f"**Calculation**: {ndcg_data.get('calculation', '')}")
            st.info(ndcg_data.get('interpretation', ''))

        # BERTScore
        with st.expander("🤖 BERTScore - Custom Metric 2", expanded=True):
            bert_data = metrics_data.get('bert_score', {})
            cols = st.columns(3)
            cols[0].metric("Precision", f"{bert_data.get('bert_score_precision', 0):.4f}")
            cols[1].metric("Recall", f"{bert_data.get('bert_score_recall', 0):.4f}")
            cols[2].metric("F1", f"{bert_data.get('bert_score_f1', 0):.4f}")
            st.info(bert_data.get('justification', ''))
            st.caption(f"**Calculation**: {bert_data.get('calculation', '')}")
            st.info(bert_data.get('interpretation', ''))

        # Other metrics
        with st.expander("📈 Additional Standard Metrics"):
            rouge_data = metrics_data.get('rouge', {})
            cols = st.columns(3)
            cols[0].metric("ROUGE-1", f"{rouge_data.get('rouge1', 0):.4f}")
            cols[1].metric("ROUGE-2", f"{rouge_data.get('rouge2', 0):.4f}")
            cols[2].metric("ROUGE-L", f"{rouge_data.get('rougeL', 0):.4f}")

            em_data = metrics_data.get('exact_match', {})
            st.metric("Exact Match", f"{em_data.get('exact_match', 0):.4f}")

    # PAGE: Innovative Evaluation
    elif page == "Innovative Evaluation":
        st.header("🚀 Innovative Evaluation Metrics")

        innovative = results.get('innovative_metrics', {})
        additional = results.get('additional_metrics', {})

        if not innovative and not additional:
            st.warning("⚠️ No innovative metrics found. Run evaluation with innovative features enabled.")
            return

        # LLM-as-Judge
        if 'llm_as_judge' in innovative:
            st.subheader("🤖 LLM-as-Judge Evaluation")
            llm_judge = innovative['llm_as_judge']

            col1, col2 = st.columns([1, 1])

            with col1:
                st.plotly_chart(
                    create_radar_chart(llm_judge),
                    use_container_width=True
                )

            with col2:
                st.metric("Overall Score", f"{llm_judge.get('avg_overall_score', 0):.4f}")
                st.metric("Factual Accuracy", f"{llm_judge.get('avg_factual_accuracy', 0):.4f}")
                st.metric("Completeness", f"{llm_judge.get('avg_completeness', 0):.4f}")
                st.metric("Relevance", f"{llm_judge.get('avg_relevance', 0):.4f}")
                st.metric("Coherence", f"{llm_judge.get('avg_coherence', 0):.4f}")
                st.caption(f"Sample size: {llm_judge.get('sample_size', 0)} questions")

        st.markdown("---")

        # Adversarial Testing
        if 'adversarial_testing' in innovative:
            st.subheader("⚔️ Adversarial Testing")
            adv = innovative['adversarial_testing']

            col1, col2 = st.columns(2)

            if 'paraphrasing_robustness' in adv:
                with col1:
                    para = adv['paraphrasing_robustness']
                    st.metric(
                        "Paraphrase Consistency",
                        f"{para.get('avg_paraphrase_consistency', 0):.3f}"
                    )
                    st.info(para.get('interpretation', ''))

            if 'unanswerable_detection' in adv:
                with col2:
                    unans = adv['unanswerable_detection']
                    st.metric(
                        "Hallucination Rate (Unanswerable)",
                        f"{unans.get('hallucination_rate', 0)*100:.1f}%"
                    )
                    st.info(unans.get('interpretation', ''))

        st.markdown("---")

        # Additional Novel Metrics
        if additional:
            st.subheader("✨ Additional Novel Metrics")

            col1, col2, col3 = st.columns(3)

            if 'entity_coverage' in additional:
                with col1:
                    ec = additional['entity_coverage']
                    st.metric("Entity Coverage", f"{ec.get('avg_coverage', 0):.3f}")
                    with st.expander("ℹ️ Details"):
                        st.caption(f"**Justification**: {ec.get('justification', '')}")
                        st.caption(f"**Calculation**: {ec.get('calculation', '')}")
                        st.info(ec.get('interpretation', ''))

            if 'answer_diversity' in additional:
                with col2:
                    ad = additional['answer_diversity']
                    st.metric("Answer Diversity", f"{ad.get('diversity_score', 0):.3f}")
                    with st.expander("ℹ️ Details"):
                        st.caption(f"**Justification**: {ad.get('justification', '')}")
                        st.caption(f"**Calculation**: {ad.get('calculation', '')}")
                        st.info(ad.get('interpretation', ''))

            if 'faithfulness' in additional:
                with col3:
                    faith = additional['faithfulness']
                    st.metric("Faithfulness", f"{faith.get('avg_faithfulness', 0):.3f}")
                    with st.expander("ℹ️ Details"):
                        st.caption(f"**Justification**: {faith.get('justification', '')}")
                        st.caption(f"**Calculation**: {faith.get('calculation', '')}")
                        st.info(faith.get('interpretation', ''))

    # PAGE: Question Analysis
    elif page == "Question Analysis":
        st.header("❓ Question Analysis")

        # Difficulty Analysis
        additional = results.get('additional_metrics', {})
        if 'difficulty_analysis' in additional:
            diff_data = additional['difficulty_analysis']

            st.subheader("📊 Performance by Question Difficulty")
            st.plotly_chart(
                create_difficulty_chart(diff_data),
                use_container_width=True
            )

            analysis = diff_data.get('analysis_by_level', {})
            for level, stats in analysis.items():
                with st.expander(f"{level.upper()} Questions ({stats['count']} total)"):
                    col1, col2 = st.columns(2)
                    col1.metric(
                        "Retrieval Success Rate",
                        f"{stats['retrieval_success_rate']*100:.1f}%"
                    )
                    col2.metric(
                        "Answer Quality Rate",
                        f"{stats['answer_quality_rate']*100:.1f}%"
                    )

        # Error Analysis
        error_data = results.get('error_analysis', {})
        if error_data:
            st.subheader("🔍 Error Analysis")

            errors_by_type = error_data.get('errors_by_type', {})

            df_errors = pd.DataFrame([
                {
                    'Question Type': qt,
                    'Total': stats['total'],
                    'Retrieval Failures': stats['retrieval_failures'],
                    'Generation Failures': stats['generation_failures']
                }
                for qt, stats in errors_by_type.items()
            ])

            st.dataframe(df_errors, use_container_width=True)

    # PAGE: Parameter Sweep
    elif page == "Parameter Sweep":
        st.header("🔧 Parameter Sweep Results")

        param_sweep = results.get('parameter_sweep', {})

        if not param_sweep:
            st.warning("⚠️ No parameter sweep results found.")
            return

        st.info("📊 Parameter sweep tests different values of K, N, and RRF k to find optimal settings")

        col1, col2, col3 = st.columns(3)

        if 'top_k_sweep' in param_sweep:
            with col1:
                st.metric("Optimal top-K", param_sweep['top_k_sweep'].get('best_k', 'N/A'))

        if 'top_n_sweep' in param_sweep:
            with col2:
                st.metric("Optimal top-N", param_sweep['top_n_sweep'].get('best_n', 'N/A'))

        if 'rrf_k_sweep' in param_sweep:
            with col3:
                st.metric("Optimal RRF k", param_sweep['rrf_k_sweep'].get('best_rrf_k', 'N/A'))

        st.markdown("---")

        # Show parameter sweep visualization if available
        param_img = Path("outputs/parameter_sweep_results.png")
        if param_img.exists():
            st.image(str(param_img), caption="Parameter Sweep Results", use_column_width=True)

    # Footer
    st.markdown("---")
    st.caption("Built with ❤️ for Hybrid RAG System Assignment")


if __name__ == "__main__":
    main()
