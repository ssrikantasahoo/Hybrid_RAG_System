"""
Data Collection Module for Hybrid RAG System
Handles Wikipedia article collection, scraping, and preprocessing
"""

import json
import random
import time
from typing import List, Dict, Tuple
from pathlib import Path
import wikipediaapi
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging
import urllib3
import ssl
import os
from unittest.mock import patch

# Disable SSL warnings and verification for corporate environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''

# Monkey patch ssl to disable verification globally
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Monkey patch requests to disable SSL verification globally
original_request = requests.Session.request

def patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return original_request(self, method, url, **kwargs)

requests.Session.request = patched_request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WikipediaDataCollector:
    """Collects and processes Wikipedia articles"""

    def __init__(self, min_words: int = 200):
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='HybridRAGSystem/1.0 (Educational Project)'
        )
        self.min_words = min_words

    def get_random_wikipedia_urls(self, count: int, exclude_urls: List[str] = None) -> List[Dict]:
        """
        Get random Wikipedia URLs with minimum word count

        Args:
            count: Number of URLs to collect
            exclude_urls: URLs to exclude from collection

        Returns:
            List of dictionaries with URL, title, and content
        """
        exclude_urls = exclude_urls or []
        collected_urls = []
        attempts = 0
        max_attempts = count * 10  # Prevent infinite loops

        logger.info(f"Collecting {count} random Wikipedia URLs...")

        # Use Wikipedia API to get random articles
        while len(collected_urls) < count and attempts < max_attempts:
            attempts += 1
            try:
                # Use Wikipedia API's random generator
                api_url = "https://en.wikipedia.org/w/api.php"
                params = {
                    'action': 'query',
                    'format': 'json',
                    'list': 'random',
                    'rnnamespace': 0,  # Main namespace only
                    'rnlimit': 1
                }

                headers = {
                    'User-Agent': 'HybridRAGSystem/1.0 (Educational Project; Python/requests)'
                }

                response = requests.get(api_url, params=params, headers=headers, verify=False)
                data = response.json()

                if 'query' not in data or 'random' not in data['query']:
                    continue

                random_page = data['query']['random'][0]
                title = random_page['title']
                page_id = random_page['id']
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

                # Skip if already collected or excluded
                if url in exclude_urls or any(u['url'] == url for u in collected_urls):
                    continue

                # Get article content
                page = self.wiki.page(title)

                if not page.exists():
                    continue

                # Check word count
                text = page.text
                word_count = len(text.split())

                if word_count >= self.min_words:
                    collected_urls.append({
                        'url': url,
                        'title': page.title,
                        'text': text,
                        'word_count': word_count
                    })
                    logger.info(f"Collected {len(collected_urls)}/{count}: {page.title} ({word_count} words)")

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                logger.warning(f"Error collecting article: {e}")
                continue

        if len(collected_urls) < count:
            logger.warning(f"Only collected {len(collected_urls)} out of {count} requested URLs")

        return collected_urls

    def get_article_from_url(self, url: str) -> Dict:
        """
        Fetch article content from a specific Wikipedia URL

        Args:
            url: Wikipedia article URL

        Returns:
            Dictionary with article data
        """
        try:
            # Extract title from URL
            title = url.split("/wiki/")[-1].replace("_", " ")

            # Get article content
            page = self.wiki.page(title)

            if not page.exists():
                logger.error(f"Page does not exist: {url}")
                return None

            text = page.text
            word_count = len(text.split())

            if word_count < self.min_words:
                logger.warning(f"Article too short ({word_count} words): {title}")
                return None

            return {
                'url': url,
                'title': page.title,
                'text': text,
                'word_count': word_count
            }

        except Exception as e:
            logger.error(f"Error fetching article {url}: {e}")
            return None

    def create_fixed_urls_set(self, count: int, output_file: str) -> List[str]:
        """
        Create and save a fixed set of Wikipedia URLs

        Args:
            count: Number of URLs to collect
            output_file: Path to save the URLs

        Returns:
            List of URLs
        """
        logger.info(f"Creating fixed URL set of {count} articles...")

        articles = self.get_random_wikipedia_urls(count)

        # Extract just the URLs
        urls = [article['url'] for article in articles]

        # Save to JSON file
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(urls, f, indent=2)

        logger.info(f"Saved {len(urls)} fixed URLs to {output_file}")

        return urls

    def load_fixed_urls(self, file_path: str) -> List[str]:
        """Load fixed URLs from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def collect_corpus(self, fixed_urls_file: str, random_count: int) -> List[Dict]:
        """
        Collect complete corpus: fixed URLs + random URLs

        Args:
            fixed_urls_file: Path to fixed URLs JSON file
            random_count: Number of additional random URLs

        Returns:
            List of article dictionaries
        """
        corpus = []

        # Load and fetch fixed URLs
        logger.info("Loading fixed URLs...")
        if Path(fixed_urls_file).exists():
            fixed_urls = self.load_fixed_urls(fixed_urls_file)
        else:
            logger.info("Fixed URLs file not found. Creating new one...")
            fixed_urls = self.create_fixed_urls_set(200, fixed_urls_file)

        logger.info(f"Fetching {len(fixed_urls)} fixed articles...")
        for url in tqdm(fixed_urls, desc="Fixed URLs"):
            article = self.get_article_from_url(url)
            if article:
                article['source_type'] = 'fixed'
                corpus.append(article)
            time.sleep(0.5)

        # Collect random URLs
        logger.info(f"Collecting {random_count} random articles...")
        random_articles = self.get_random_wikipedia_urls(
            random_count,
            exclude_urls=fixed_urls
        )

        for article in random_articles:
            article['source_type'] = 'random'
            corpus.append(article)

        logger.info(f"Total corpus size: {len(corpus)} articles")

        return corpus

    def save_corpus(self, corpus: List[Dict], output_file: str):
        """Save corpus to JSON file"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved corpus to {output_file}")

    def load_corpus(self, file_path: str) -> List[Dict]:
        """Load corpus from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)


if __name__ == "__main__":
    # Example usage
    collector = WikipediaDataCollector(min_words=200)

    # Create fixed URLs if needed
    fixed_urls_file = "data/fixed_urls.json"
    if not Path(fixed_urls_file).exists():
        collector.create_fixed_urls_set(200, fixed_urls_file)

    # Collect full corpus
    corpus = collector.collect_corpus(fixed_urls_file, random_count=300)

    # Save corpus
    collector.save_corpus(corpus, "data/raw_corpus.json")
