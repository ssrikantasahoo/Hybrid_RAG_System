"""
Streamlit User Interface for Hybrid RAG System
"""

import streamlit as st
import sys
import time
import json
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent))

from dense_retrieval import DenseRetriever
from sparse_retrieval import BM25Retriever
from rrf_fusion import HybridRetriever
from llm_generation import LLMGenerator, RAGPipeline


# Page configuration
st.set_page_config(
    page_title="Hybrid RAG System - Wikipedia Q&A",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_system(config_path="config.yaml"):
    """Load and cache the RAG system components"""

    # Resolve config path
    if not Path(config_path).exists():
        # Try finding it in the project root (parent of src)
        root_config = Path(__file__).parent.parent / "config.yaml"
        if root_config.exists():
            config_path = str(root_config)
        else:
            st.error(f"Config file not found. Checked: {config_path} and {root_config}")
            return None

    with st.spinner("Loading RAG system components..."):
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Resolve paths relative to config file location
        config_dir = Path(config_path).parent.resolve()
        for key, path_str in config.get('paths', {}).items():
            # If path is relative, make it absolute relative to config directory
            path = Path(path_str)
            if not path.is_absolute():
                config['paths'][key] = str(config_dir / path)

        # Load dense retriever
        st.info("Loading Dense Retriever (FAISS)...")
        dense_retriever = DenseRetriever(
            model_name=config['dense_retrieval']['model_name']
        )

        vector_index_dir = config['paths']['vector_index_dir']
        # Use str() for compatibility if libraries expect string
        if Path(vector_index_dir).exists():
            dense_retriever.load_index(vector_index_dir)
        else:
            st.error(f"Vector index not found at {vector_index_dir}. Please build the index first.")
            return None

        # Load sparse retriever
        st.info("Loading Sparse Retriever (BM25)...")
        sparse_retriever = BM25Retriever(
            k1=config['sparse_retrieval']['k1'],
            b=config['sparse_retrieval']['b']
        )

        bm25_index_file = config['paths']['bm25_index_file']
        if Path(bm25_index_file).exists():
            sparse_retriever.load_index(bm25_index_file)
        else:
            st.error(f"BM25 index not found at {bm25_index_file}. Please build the index first.")
            return None

        # Create hybrid retriever
        st.info("Initializing Hybrid Retriever with RRF...")
        hybrid_retriever = HybridRetriever(
            dense_retriever,
            sparse_retriever,
            rrf_k=config['rrf']['k_constant']
        )

        # Load LLM generator
        st.info(f"Loading LLM: {config['llm']['model_name']}...")
        llm_generator = LLMGenerator(
            model_name=config['llm']['model_name'],
            max_length=config['llm']['max_length'],
            temperature=config['llm']['temperature'],
            top_p=config['llm']['top_p']
        )

        # Create RAG pipeline
        rag_pipeline = RAGPipeline(hybrid_retriever, llm_generator)

        st.success("✅ System loaded successfully!")

        return rag_pipeline, config


def display_retrieval_results(retrieval_results, method="fused"):
    """Display retrieval results in a formatted way"""

    if method == "fused":
        results = retrieval_results['fused_results']
        st.subheader("🔗 Fused Results (RRF)")
    elif method == "dense":
        results = retrieval_results['dense_results']
        st.subheader("🧮 Dense Retrieval Results")
    else:
        results = retrieval_results['sparse_results']
        st.subheader("📝 Sparse Retrieval Results (BM25)")

    for i, result in enumerate(results, 1):
        chunk = result['chunk']

        with st.expander(f"#{i} - {chunk['title']} (Score: {result.get('rrf_score', result.get('score', 0)):.4f})"):
            # Chunk metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rank", i)
            with col2:
                if method == "fused":
                    st.metric("RRF Score", f"{result['rrf_score']:.4f}")
                else:
                    st.metric("Score", f"{result['score']:.4f}")
            with col3:
                st.metric("Tokens", chunk['token_count'])

            # Source information
            st.markdown(f"**Source:** [{chunk['title']}]({chunk['url']})")
            st.markdown(f"**Chunk ID:** `{chunk['chunk_id']}`")

            # Chunk text
            st.markdown("**Text:**")
            st.text_area(
                "Content",
                chunk['text'],
                height=150,
                key=f"{method}_{i}",
                label_visibility="collapsed"
            )

            # Method details for fused results
            if method == "fused" and 'method_details' in result:
                st.markdown("**Method Contributions:**")
                details = result['method_details']

                detail_cols = st.columns(len(details))
                for idx, (method_name, method_info) in enumerate(details.items()):
                    with detail_cols[idx]:
                        st.caption(f"**{method_name.upper()}**")
                        st.caption(f"Score: {method_info['score']:.4f}")
                        st.caption(f"Rank: {method_info['rank']}")
                        if 'rrf_contribution' in method_info:
                            st.caption(f"RRF: {method_info['rrf_contribution']:.4f}")


def main():
    """Main Streamlit application"""

    # Title and description
    st.title("🔍 Hybrid RAG System - Wikipedia Q&A")
    st.markdown("""
    This system combines **Dense Vector Retrieval**, **Sparse Keyword Retrieval (BM25)**,
    and **Reciprocal Rank Fusion (RRF)** to answer questions from Wikipedia articles.
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Retrieval settings
        st.subheader("Retrieval Settings")
        top_k = st.slider(
            "Top-K per method",
            min_value=1,
            max_value=20,
            value=10,
            help="Number of chunks to retrieve from each method"
        )

        top_n = st.slider(
            "Top-N final chunks",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of chunks to use for answer generation"
        )

        # Generation settings
        st.subheader("Generation Settings")
        max_tokens = st.slider(
            "Max generation tokens",
            min_value=50,
            max_value=500,
            value=256,
            help="Maximum tokens to generate"
        )

        # Display options
        st.subheader("Display Options")
        show_dense = st.checkbox("Show Dense Results", value=False)
        show_sparse = st.checkbox("Show Sparse Results", value=False)
        show_timing = st.checkbox("Show Timing Details", value=True)

    # Load system
    system_data = load_system()

    if system_data is None:
        st.error("Failed to load system. Please check the configuration and ensure indices are built.")
        return

    rag_pipeline, config = system_data

    # Main query interface
    st.header("💬 Ask a Question")

    # Example questions
    example_questions = [
        "What is machine learning?",
        "Who invented the telephone?",
        "What are the causes of climate change?",
        "How does photosynthesis work?",
        "What is the theory of relativity?"
    ]

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Enter your question:",
            placeholder="Type your question here...",
            label_visibility="collapsed"
        )
    with col2:
        example_selected = st.selectbox(
            "Or select an example:",
            [""] + example_questions,
            label_visibility="collapsed"
        )

    # Use example if selected
    if example_selected:
        query = example_selected

    # Query button
    if st.button("🔍 Search and Answer", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter a question!")
            return

        # Process query
        with st.spinner("Retrieving information and generating answer..."):
            start_time = time.time()

            try:
                # Run RAG pipeline
                result = rag_pipeline.query(
                    query,
                    top_k=top_k,
                    top_n=top_n,
                    max_new_tokens=max_tokens
                )

                end_time = time.time()
                response_time = end_time - start_time

                # Display answer
                st.success("✅ Answer Generated!")

                # Answer section
                st.header("📝 Answer")
                st.markdown(f"### {result['answer']}")

                # Metrics
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Response Time", f"{response_time:.2f}s")
                with metric_cols[1]:
                    st.metric(
                        "Chunks Used",
                        len(result['retrieval_results'].get(
                            'generation_chunks',
                            result['retrieval_results']['fused_results']
                        ))
                    )
                with metric_cols[2]:
                    st.metric("Unique Chunks Retrieved", result['retrieval_results']['num_unique_chunks'])
                with metric_cols[3]:
                    st.metric("Input Tokens", result['generation_metadata']['input_tokens'])

                # Timing breakdown
                if show_timing:
                    st.info(f"⏱️ Total Response Time: {response_time:.3f} seconds")

                # Retrieval results
                st.markdown("---")

                # Display fused results
                display_retrieval_results(result['retrieval_results'], method="fused")

                # Optional: Show individual method results
                if show_dense:
                    st.markdown("---")
                    display_retrieval_results(result['retrieval_results'], method="dense")

                if show_sparse:
                    st.markdown("---")
                    display_retrieval_results(result['retrieval_results'], method="sparse")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Hybrid RAG System | Dense + Sparse + RRF | Powered by Wikipedia</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
