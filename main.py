import pandas as pd
import logging
from wiki_loader import fetch_wikipedia_articles, save_metadata
from text_processor import process_sections
from embedding_generator import generate_and_save_embeddings
from utils import save_embeddings_to_csv
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    # Embeddings and metadata will now be generated dynamically in app.py
    # generate_and_save_embeddings(category_title)

    logger.info("Embeddings and metadata generation is now handled dynamically in app.py.")

if __name__ == "__main__":
    main()