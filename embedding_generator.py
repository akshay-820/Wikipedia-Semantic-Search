# This file generates text embeddings using the OpenAI API. 
# It processes text in batches and retrieves their vector representations.
from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from text_processor import process_sections
from wiki_loader import fetch_wikipedia_articles
from utils import save_embeddings_to_csv, save_metadata

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 1000

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using OpenAI API."""
    embeddings = []
    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = texts[batch_start:batch_end]
        print(f"Generating embeddings for batch {batch_start} to {batch_end - 1}")
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for i, be in enumerate(response.data):
            assert i == be.index
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)
    return embeddings

def generate_and_save_embeddings(base_category, related_titles=None, max_depth=2, embeddings_dir='data/embeddings', metadata_file='data/metadata.csv'):
    """
    Generates embeddings for Wikipedia articles and saves them along with metadata.

    Args:
        base_category (str): The base Wikipedia category to start the search from.
        related_titles (list, optional): A list of related Wikipedia titles to include. Defaults to None.
        max_depth (int, optional): The maximum depth for recursive search. Defaults to 2.
        embeddings_dir (str, optional): Directory to save the embeddings. Defaults to 'data/embeddings'.
        metadata_file (str, optional): File path to save the metadata. Defaults to 'data/metadata.csv'.
    """
    embeddings_dir = os.path.join(embeddings_dir, base_category.replace(" ", "_"))
    os.makedirs(embeddings_dir, exist_ok=True)

    # Fetch articles
    articles = fetch_wikipedia_articles(base_category, max_depth)

    # Process articles into sections
    titles = [article['title'] for article in articles]
    texts = process_sections(titles)

    # Generate embeddings
    embeddings = generate_embeddings(texts)

    # Create DataFrame
    df = pd.DataFrame({"text": texts, "embedding": embeddings})

    # Save embeddings to CSV
    embeddings_file = os.path.join(embeddings_dir, "embeddings.csv")
    save_embeddings_to_csv(df, embeddings_file)

    # Save metadata
    metadata = pd.DataFrame([{
        'title': article['title'],
        'url': article['url'],
        'summary': article.get('summary', ''),
        'last_modified': article.get('last_modified', '')
    } for article in articles])
    save_metadata(metadata, metadata_file) 