import glob
import numpy as np
import pandas as pd
import os
import logging
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
from wiki_loader import fetch_wikipedia_articles, save_metadata, search_wikipedia_articles, fetch_wikipedia_articles_from_titles
from text_processor import process_sections
from embedding_generator import generate_and_save_embeddings, generate_embeddings
from functools import lru_cache
import re
import wikipedia
from bs4 import BeautifulSoup
import structlog
import time
import traceback

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

def _wiki_request(*args, **kwargs):
    """Monkey patch for Wikipedia's _wiki_request to use lxml parser"""
    html = wikipedia._wiki_request(*args, **kwargs)
    if isinstance(html, str):
        return BeautifulSoup(html, 'lxml')
    return html

wikipedia._wiki_request = _wiki_request

@lru_cache(maxsize=100)
def get_embeddings(query: str) -> list[float]:
    return generate_embeddings([query])[0]

def load_embeddings(embeddings_file: str) -> pd.DataFrame:
    """Loads embeddings from a CSV file."""
    try:
        if not os.path.exists(embeddings_file):
            logger.info(f"No existing embeddings file found. Will create new one at: {embeddings_file}")
            return pd.DataFrame(columns=['text', 'embedding'])
        
        df = pd.read_csv(embeddings_file)
        # Convert string representation of embeddings back to list
        df['embedding'] = df['embedding'].apply(lambda x: list(map(float, x.strip('[]').split(','))))
        return df
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        raise

def load_metadata(metadata_file: str) -> pd.DataFrame:
    """Loads metadata from a CSV file."""
    try:
        if not os.path.exists(metadata_file):
            logger.info(f"No existing metadata file found. Will create new one at: {metadata_file}")
            return pd.DataFrame(columns=['title', 'url', 'summary', 'last_modified'])
        return pd.read_csv(metadata_file)
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        raise

def search_embeddings(
    query_embedding: np.ndarray,
    embeddings_df: pd.DataFrame,
    metadata: pd.DataFrame,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Finds the top_k most similar embeddings using cosine similarity.
    Returns the corresponding metadata and similarity scores.
    """
    try:
        # Ensure query_embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Convert embeddings from list to numpy array for calculation
        embeddings = np.vstack(embeddings_df['embedding'].tolist())
        
        # Perform similarity calculation
        similarity_scores = cosine_similarity(query_embedding, embeddings)[0]
        
        # Handle empty results
        if len(similarity_scores) == 0:
            return pd.DataFrame()  # Return empty DataFrame if no results
            
        # Ensure we don't request more results than we have
        top_k = min(top_k, len(similarity_scores))
        
        # Get indices of top k scores
        top_indices = similarity_scores.argsort()[::-1][:top_k]
        top_scores = similarity_scores[top_indices]
        
        # Create results DataFrame from embeddings_df first
        results = embeddings_df.iloc[top_indices].copy()
        
        # Add metadata columns if they exist
        if not metadata.empty:
            for col in metadata.columns:
                if len(metadata) > max(top_indices):
                    results[col] = metadata.iloc[top_indices][col].values
        
        results['similarity_score'] = top_scores
        return results
        
    except Exception as e:
        logger.error(f"Error in search_embeddings: {str(e)}")
        raise  # Re-raise the exception to see the full traceback

def format_results(results_df):
    """Enhanced results formatting with better readability and metadata."""
    formatted_results = []
    for _, row in results_df.iterrows():
        result = {
            "title": row.get('title', 'Untitled'),
            "url": row.get('url', ''),
            "summary": row.get('text', '')[:300] + "..." if len(row.get('text', '')) > 300 else row.get('text', ''),
            "similarity": f"{row.get('similarity_score', 0):.2f}",
            "metadata": {
                "category": row.get('category', 'Uncategorized'),
                "last_updated": row.get('last_updated', 'Unknown'),
                "word_count": len(row.get('text', '').split())
            }
        }
        formatted_results.append(result)
    return {"results": formatted_results, "total_count": len(formatted_results)}

def clean_wiki_text(text: str) -> str:
    """Remove Wikipedia template tags and clean up the text."""
    # Remove template tags {{...}}
    text = re.sub(r'\{\{[^}]+\}\}', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def format_wiki_results(status: str, articles: list) -> tuple[str, dict, str]:
    if not articles:
        return status, {"results": []}, ""
        
    formatted_articles = []
    for article in articles:
        # Get summary and clean it
        summary = article.get('summary', '')
        cleaned_summary = clean_wiki_text(summary)
        
        formatted_articles.append({
            "title": article['title'],
            "url": article['url'],
            "summary": cleaned_summary[:200] + "..." if cleaned_summary else "",
            "relevance": "High"  # Add relevance indicator
        })
    
    return status, {"results": formatted_articles}, ""

try:
    # Add directory creation
    os.makedirs('data/embeddings', exist_ok=True)
    os.makedirs('data/metadata', exist_ok=True)
    
    # Load data
    logger.info("Loading embeddings and metadata...")
    embeddings_file = 'data/embeddings/embeddings.csv'
    metadata_file = 'data/metadata.csv'
    
    # Initialize with empty arrays if no data exists
    embeddings_df = load_embeddings(embeddings_file)
    metadata = load_metadata(metadata_file)
    
    if embeddings_df.empty:
        logger.info("No existing embeddings found. System will create new embeddings on first search.")
    else:
        logger.info(f"Loaded embeddings and metadata successfully")

except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
    raise

def search_wikipedia(query: str, top_k: int = 5) -> tuple[str, list[dict]]:
    """Searches Wikipedia for articles related to the query."""
    start_time = time.time()
    logger.info("search_started", 
                query=query, 
                top_k=top_k,
                timestamp=time.time())
    try:
        # Search for articles
        search_results = search_wikipedia_articles(query, limit=top_k)
        if not search_results:
            logger.info("No results found for query: %s", query)
            return "No results found.", []

        # Fetch full articles
        article_titles = [article['title'] for article in search_results]
        articles = fetch_wikipedia_articles_from_titles(article_titles)
        
        duration = time.time() - start_time
        logger.info("search_completed",
                   query=query,
                   results_count=len(articles),
                   duration=duration)
        
        return f"Found {len(articles)} results in {duration:.2f} seconds", articles

    except Exception as e:
        logger.error("search_failed",
                    query=query,
                    error=str(e),
                    traceback=traceback.format_exc())
        return f"Error: {str(e)}", []

def create_gradio_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Semantic Wikipedia Search")
        
        # State variables for progress updates
        status_message = gr.State("")
        is_processing = gr.State(False)
        
        with gr.Row():
            with gr.Column(scale=3):
                query = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter your search query here...",
                    lines=3
                )
                
            with gr.Column(scale=1):
                num_results = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of Results"
                )
                
        with gr.Row():
            search_button = gr.Button("Search", variant="primary")
            clear_button = gr.Button("Clear")
            
        with gr.Accordion("Advanced Options", open=False):
            similarity_threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.7,
                label="Similarity Threshold"
            )
        
        # Progress and status indicators
        with gr.Row():
            status_box = gr.Textbox(
                label="Status",
                value="Ready",
                interactive=False
            )
            
        # Progress tracking
        progress_tracker = gr.Textbox(
            label="Progress",
            value="",
            interactive=False,
            visible=False
        )
            
        # Results area
        with gr.Row():
            output_box = gr.JSON(label="Search Results")
            
        # Error messages area
        error_box = gr.Textbox(
            label="Errors",
            visible=False,
            interactive=False
        )

        def search_with_progress(query, num_results, progress=gr.Progress()):
            try:
                # Update status
                progress(0, desc="Initializing search...")
                status, articles = search_wikipedia(query, top_k=num_results)
                
                # Simulate progress steps (you can replace with actual progress)
                progress(0.3, desc="Fetching articles...")
                time.sleep(0.5)  # Simulate processing time
                
                progress(0.6, desc="Processing results...")
                time.sleep(0.5)  # Simulate processing time
                
                progress(0.9, desc="Formatting output...")
                status, results, _ = format_wiki_results(status, articles)
                
                progress(1.0, desc="Complete!")
                return {
                    status_box: "Search completed",
                    output_box: results,
                    error_box: gr.update(visible=False),
                    progress_tracker: gr.update(visible=False)
                }
                
            except Exception as e:
                error_msg = f"Error during search: {str(e)}"
                return {
                    status_box: "Error occurred",
                    output_box: {"results": []},
                    error_box: gr.update(visible=True, value=error_msg),
                    progress_tracker: gr.update(visible=False)
                }

        def clear_outputs():
            return {
                query: "",
                status_box: "Ready",
                output_box: {"results": []},
                error_box: gr.update(visible=False),
                progress_tracker: gr.update(visible=False)
            }

        # Event handlers
        search_button.click(
            fn=search_with_progress,
            inputs=[query, num_results],
            outputs=[status_box, output_box, error_box, progress_tracker],
            show_progress=True
        )
        
        clear_button.click(
            fn=clear_outputs,
            inputs=[],
            outputs=[query, status_box, output_box, error_box, progress_tracker]
        )
        
        # Add example queries
        gr.Examples(
            examples=[
                ["What is quantum computing?", 5],
                ["Explain artificial intelligence", 3],
                ["History of space exploration", 4]
            ],
            inputs=[query, num_results]
        )
        
        return demo

if __name__ == "__main__":
    logger.info("Starting Gradio interface...")
    iface = create_gradio_interface()
    iface.launch(share=False)