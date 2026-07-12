import glob
import numpy as np
import pandas as pd
import os
import logging
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
from google import genai
from dotenv import load_dotenv
from wiki_loader import fetch_wikipedia_articles, save_metadata, search_wikipedia_articles, fetch_wikipedia_articles_from_titles
from text_processor import process_sections
from embedding_generator import generate_embeddings
from functools import lru_cache
import re
import wikipedia
import structlog
import time
import traceback
import mwparserfromhell

# FIX: Set unique User-Agent to comply with Wikipedia API rules and prevent 429/JSONDecodeError
wikipedia.set_user_agent("SemanticSearchApp/1.0 (contact:akshay@example.com)")

load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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

@lru_cache(maxsize=100)
def get_embeddings(query: str) -> list[float]:
    """Retrieves embedding vector for the single incoming search string."""
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )
    return response.embeddings[0].values

def load_embeddings(embeddings_file: str) -> pd.DataFrame:
    """Loads embeddings from a CSV file."""
    try:
        if not os.path.exists(embeddings_file):
            logger.info(f"No existing embeddings file found. Will create new one at: {embeddings_file}")
            return pd.DataFrame(columns=['text', 'embedding'])
        
        df = pd.read_csv(embeddings_file)
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
    """Finds the top_k most similar embeddings using cosine similarity."""
    try:
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        embeddings = np.vstack(embeddings_df['embedding'].tolist())
        similarity_scores = cosine_similarity(query_embedding, embeddings)[0]
        
        if len(similarity_scores) == 0:
            return pd.DataFrame()
            
        top_k = min(top_k, len(similarity_scores))
        top_indices = similarity_scores.argsort()[::-1][:top_k]
        top_scores = similarity_scores[top_indices]
        
        results = embeddings_df.iloc[top_indices].copy()
        
        if not metadata.empty:
            for col in metadata.columns:
                if len(metadata) > max(top_indices):
                    results[col] = metadata.iloc[top_indices][col].values
        
        results['similarity_score'] = top_scores
        return results
    except Exception as e:
        logger.error(f"Error in search_embeddings: {str(e)}")
        raise

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
    """Strip Wikipedia markup (templates, links, files, refs) down to plain readable text."""
    wikicode = mwparserfromhell.parse(text)

    for template in wikicode.filter_templates(recursive=False):
        try:
            wikicode.remove(template)
        except ValueError:
            pass  # already removed as part of a parent node

    for link in wikicode.filter_wikilinks(recursive=False):
        if link.title.strip().lower().startswith(('file:', 'image:')):
            try:
                wikicode.remove(link)
            except ValueError:
                pass

    plain = wikicode.strip_code()
    plain = re.sub(r'<ref.*?</ref>', '', plain, flags=re.DOTALL)
    plain = re.sub(r'\s+', ' ', plain)
    return plain.strip()

try:
    os.makedirs('data/embeddings', exist_ok=True)
    os.makedirs('data/metadata', exist_ok=True)
    
    logger.info("Loading embeddings and metadata...")
    embeddings_file = 'data/embeddings/embeddings.csv'
    metadata_file = 'data/metadata.csv'
    
    embeddings_df = load_embeddings(embeddings_file)
    metadata = load_metadata(metadata_file)

except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
    raise

def search_wikipedia(query: str, top_k: int = 5) -> tuple[str, pd.DataFrame]:
    global embeddings_df, metadata
    start_time = time.time()
    logger.info("search_started", query=query, top_k=top_k, timestamp=time.time())
    
    try:
        logger.info("Gathering live articles to embed via Gemini...")
        search_results = search_wikipedia_articles(query, limit=top_k)
        if search_results:
            article_titles = [art['title'] for art in search_results]
            articles = fetch_wikipedia_articles_from_titles(article_titles)
            
            texts, titles, urls = [], [], []
            for art in articles:
                try:
                    clean_text = clean_wiki_text(art.get('summary', ''))
                    if clean_text:
                        texts.append(clean_text)
                        titles.append(art['title'])
                        urls.append(art['url'])
                except Exception as e:
                    logger.warning("skipping_article", title=art.get('title'), error=str(e))
                    continue
            
            if texts:
                vectors = generate_embeddings(texts)
                embeddings_df = pd.DataFrame({'text': texts, 'embedding': vectors})
                metadata = pd.DataFrame({'title': titles, 'url': urls})

        if embeddings_df.empty:
            return "No text corpus available to process semantic lookup.", pd.DataFrame()

        query_embedding = np.array(get_embeddings(query))
        results_df = search_embeddings(query_embedding, embeddings_df, metadata, top_k=top_k)
        duration = time.time() - start_time
        
        return f"Found {len(results_df)} semantic matches in {duration:.2f} seconds", results_df

    except Exception as e:
        logger.error("search_failed", query=query, error=str(e), traceback=traceback.format_exc())
        return f"Error: {str(e)}", pd.DataFrame()

def create_gradio_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Semantic Wikipedia Search (Powered by Gemini)")
        
        with gr.Row():
            with gr.Column(scale=3):
                query = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter your search query here...",
                    lines=3
                )
            with gr.Column(scale=1):
                num_results = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1, label="Number of Results"
                )
                
        with gr.Row():
            search_button = gr.Button("Search", variant="primary")
            clear_button = gr.Button("Clear")
            
        with gr.Accordion("Advanced Options", open=False):
            similarity_threshold = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.7, label="Similarity Threshold"
            )
        
        status_box = gr.Textbox(label="Status", value="Ready", interactive=False)
        output_box = gr.JSON(label="Search Results")
        error_box = gr.Textbox(label="Errors", visible=False, interactive=False)

        def search_with_progress(query, num_results, progress=gr.Progress()):
            try:
                progress(0, desc="Querying vector space...")
                status, results_df = search_wikipedia(query, top_k=num_results)
                
                if results_df.empty:
                    return {status_box: status, output_box: {"results": [], "total_count": 0}, error_box: gr.update(visible=False)}
                
                progress(0.7, desc="Formatting parameters...")
                formatted_json = format_results(results_df)
                progress(1.0, desc="Complete!")
                return {status_box: status, output_box: formatted_json, error_box: gr.update(visible=False)}
            except Exception as e:
                return {status_box: "Error occurred", output_box: {"results": []}, error_box: gr.update(visible=True, value=str(e))}

        def clear_outputs():
            return {query: "", status_box: "Ready", output_box: {"results": []}, error_box: gr.update(visible=False)}

        search_button.click(fn=search_with_progress, inputs=[query, num_results], outputs=[status_box, output_box, error_box], show_progress=True)
        clear_button.click(fn=clear_outputs, outputs=[query, status_box, output_box, error_box])
        
        return demo

if __name__ == "__main__":
    iface = create_gradio_interface()
    iface.launch(share=False)