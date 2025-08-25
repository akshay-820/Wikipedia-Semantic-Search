import mwclient
import pandas as pd
import os
import wikipedia
import time
from typing import Optional
from time import sleep

WIKI_SITE = "en.wikipedia.org"

def fetch_wikipedia_articles(category_title: str, max_depth: int = 1, max_articles: int = 5) -> list[dict]:
    """
    Fetch Wikipedia articles based on a category title.

    Args:
        category_title: Title of the Wikipedia category
        max_depth: Maximum depth to search for subcategories
        max_articles: Maximum number of articles to fetch (default: 5)
    """
    try:
        site = mwclient.Site(WIKI_SITE)
        titles = titles_from_category(site.categories[category_title], max_depth, max_articles)
        
        # Use max_articles parameter instead of hard-coded 5
        titles = list(titles)[:max_articles]
        print(f"Processing {len(titles)} articles from {category_title}.")

        articles = []
        for title in titles:
            page = site.pages[title]
            if page.exists:
                text = page.text()
                summary = text.split('\n\n')[0] if text else ''
                
                article = {
                    'title': page.name,
                    'text': text,
                    'url': f"https://{WIKI_SITE}/wiki/{page.name.replace(' ', '_')}",
                    'summary': summary,
                    'last_modified': ''
                }
                articles.append(article)

        print(f"Successfully processed {len(articles)} articles")
        return articles

    except Exception as e:
        print(f"Error fetching Wikipedia articles: {e}")
        return []

def titles_from_category(category: mwclient.listing.Category, max_depth: int, max_articles: int = 5) -> set[str]:
    """Return a set of page titles in a given Wiki category and its subcategories."""
    titles = set()
    try:
        site = mwclient.Site(WIKI_SITE)
        # Get category members only if the category object is valid and has 'members' attribute
        if hasattr(category, 'members'):
            for cm in category.members():
                # Break if we already have 5 titles
                if len(titles) >= max_articles:
                    break
                    
                if isinstance(cm, mwclient.page.Page):
                    titles.add(cm.name)
                elif isinstance(cm, mwclient.listing.Category) and max_depth > 0:
                    # Only get more titles if we haven't reached 5 yet
                    if len(titles) < max_articles:
                        deeper_titles = titles_from_category(cm, max_depth=max_depth - 1, max_articles=max_articles)
                        titles.update(list(deeper_titles)[:max_articles - len(titles)])

    except Exception as e:
        print(f"Error while fetching titles: {e}")

    # Final safety check to ensure we never return more than 5 titles
    return set(list(titles)[:max_articles])

def load_wikipedia_articles(category_title: str, max_depth: int) -> list[str]:
    """Load Wikipedia articles based on a category title."""
    site = mwclient.Site(WIKI_SITE)
    category_page = site.categories[category_title]  # Fetch category using site.categories
    titles = titles_from_category(category_page, max_depth=max_depth)
    print(f"Found {len(titles)} article titles related to {category_title}.")
    return list(titles)

def save_metadata(articles, output_file='data/metadata.csv'):
    """
    Save metadata for extracted Wikipedia articles
    
    Args:
        articles: List of dictionaries containing article info
        output_file: Path to save the metadata CSV
    """
    # Create metadata directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    metadata = []
    for article in articles:
        metadata.append({
            'title': article['title'],
            'url': article['url'],
            'summary': article.get('summary', ''),  # Optional summary
            'last_modified': article.get('last_modified', ''),  # Optional timestamp
        })
    
    df = pd.DataFrame(metadata)
    df.to_csv(output_file, index=False)
    return df 

class WikiAPIError(Exception):
    """Custom exception for Wikipedia API errors"""
    pass

def with_retry(func):
    """Decorator to retry API calls with exponential backoff"""
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise WikiAPIError(f"Failed after {max_retries} attempts: {str(e)}")
                time.sleep(retry_delay)
                retry_delay *= 2
    return wrapper

def rate_limit(min_time_between_calls: float = 1.0):
    """Decorator to enforce minimum time between API calls"""
    last_call_time = {}
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_time = time.time()
            if func.__name__ in last_call_time:
                elapsed = current_time - last_call_time[func.__name__]
                if elapsed < min_time_between_calls:
                    sleep(min_time_between_calls - elapsed)
            
            result = func(*args, **kwargs)
            last_call_time[func.__name__] = time.time()
            return result
        return wrapper
    return decorator

@rate_limit(1.0)
@with_retry
def search_wikipedia_articles(query: str, limit: Optional[int] = 5) -> list[dict]:
    """
    Search Wikipedia for relevant pages based on a query.
    
    Args:
        query: Search query
        limit: Maximum number of results to return
    Returns:
        List of article dictionaries
    """
    try:
        search_results = wikipedia.search(query, results=limit)
        articles = []
        
        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                articles.append({
                    'title': page.title,
                    'url': page.url,
                    'content': page.content,
                    'summary': page.summary
                })
            except (wikipedia.exceptions.DisambiguationError,
                   wikipedia.exceptions.PageError):
                continue
                
        return articles
        
    except Exception as e:
        raise WikiAPIError(f"Error searching Wikipedia: {str(e)}")

def fetch_wikipedia_articles_from_titles(titles: list[str]) -> list[dict]:
    """
    Fetch full Wikipedia articles from a list of titles.
    
    Args:
        titles: List of Wikipedia article titles
    Returns:
        List of dictionaries containing article information
    """
    try:
        site = mwclient.Site(WIKI_SITE)
        articles = []
        
        for title in titles:
            page = site.pages[title]
            if page.exists:
                article = {
                    'title': page.name,
                    'text': page.text(),
                    'url': f"https://{WIKI_SITE}/wiki/{page.name.replace(' ', '_')}",
                    'summary': page.text().split('\n\n')[0] if page.text() else ''
                }
                articles.append(article)
                
        return articles
        
    except Exception as e:
        print(f"Error fetching articles from titles: {e}")
        return [] 