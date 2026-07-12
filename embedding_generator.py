from google import genai
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from text_processor import process_sections
from wiki_loader import fetch_wikipedia_articles
from utils import save_embeddings_to_csv, save_metadata

load_dotenv()

# Initialize the official Gemini Client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

EMBEDDING_MODEL = "gemini-embedding-001"
BATCH_SIZE = 100

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using Gemini API."""
    embeddings = []
    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = texts[batch_start:batch_end]
        print(f"Generating Gemini embeddings for batch {batch_start} to {batch_end - 1}")
        
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch
        )
        
        batch_embeddings = [e.values for e in response.embeddings]
        embeddings.extend(batch_embeddings)
    return embeddings

def generate_and_save_embeddings(base_category, related_titles=None, max_depth=2, embeddings_dir='data/embeddings', metadata_file='data/metadata.csv'):
    embeddings_dir = os.path.join(embeddings_dir, base_category.replace(" ", "_"))
    os.makedirs(embeddings_dir, exist_ok=True)

    articles = fetch_wikipedia_articles(base_category, max_depth)
    titles = [article['title'] for article in articles]
    texts = process_sections(titles)
    embeddings = generate_embeddings(texts)

    df = pd.DataFrame({"text": texts, "embedding": embeddings})
    embeddings_file = os.path.join(embeddings_dir, "embeddings.csv")
    save_embeddings_to_csv(df, embeddings_file)

    metadata = pd.DataFrame([{
        'title': article['title'],
        'url': article['url'],
        'summary': article.get('summary', ''),
        'last_modified': article.get('last_modified', '')
    } for article in articles])
    save_metadata(metadata, metadata_file)