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
import yaml

# Load configuration
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    SSL_VERIFY = config.get('network', {}).get('ssl_verify', True)
except Exception:
    SSL_VERIFY = True  # Default to secure

# Only disable SSL warnings if SSL verification is disabled
if not SSL_VERIFY:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logging.warning("SSL verification is DISABLED - This is not recommended for production!")
    logging.warning("Set network.ssl_verify: true in config.yaml to enable SSL verification")

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
        self.ssl_verify = SSL_VERIFY

    def get_random_wikipedia_urls(self, count: int, exclude_urls: List[str] = None) -> List[Dict]:
        """
        Get random Wikipedia URLs with minimum word count

        Args:
            count: Number of URLs to collect
            exclude_urls: URLs to exclude from collection

        Returns:
            List of dictionaries with URL, title, and text
        """
        exclude_urls = exclude_urls or []
        collected_urls = []
        attempts = 0
        max_attempts = count * 10  # Prevent infinite loops

        logger.info(f"Collecting {count} random Wikipedia URLs...")
        logger.info(f"SSL Verification: {'ENABLED' if self.ssl_verify else 'DISABLED'}")

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

                response = requests.get(
                    api_url,
                    params=params,
                    headers=headers,
                    verify=self.ssl_verify,
                    timeout=10
                )
                data = response.json()

                if 'query' not in data or 'random' not in data['query']:
                    continue

                random_page = data['query']['random'][0]
                page_title = random_page['title']

                # Get full page content
                page = self.wiki.page(page_title)

                if not page.exists():
                    continue

                # Get text content
                text = page.text
                word_count = len(text.split())

                if word_count < self.min_words:
                    continue

                page_url = page.fullurl

                # Skip if in exclude list
                if page_url in exclude_urls or page_url in [u['url'] for u in collected_urls]:
                    continue

                collected_urls.append({
                    'url': page_url,
                    'title': page_title,
                    'text': text,
                    # Backward-compatible alias for any older code paths.
                    'content': text,
                    'word_count': word_count
                })

                if len(collected_urls) % 50 == 0:
                    logger.info(f"  Progress: {len(collected_urls)}/{count}")

            except Exception as e:
                logger.debug(f"Error fetching random article: {e}")
                continue

        logger.info(f"[OK] Collected {len(collected_urls)} random URLs")
        return collected_urls

    def get_article_from_url(self, url: str) -> Dict:
        """
        Get article content from Wikipedia URL

        Args:
            url: Wikipedia URL

        Returns:
            Dictionary with URL, title, and text
        """
        try:
            # Extract title from URL
            title = url.split('/wiki/')[-1].replace('_', ' ')

            # Get page
            page = self.wiki.page(title)

            if not page.exists():
                logger.warning(f"Page does not exist: {title}")
                return None

            text = page.text
            word_count = len(text.split())

            if word_count < self.min_words:
                logger.warning(f"Page too short ({word_count} words): {title}")
                return None

            return {
                'url': url,
                'title': title,
                'text': text,
                # Backward-compatible alias for any older code paths.
                'content': text,
                'word_count': word_count
            }

        except Exception as e:
            logger.error(f"Error fetching article from {url}: {e}")
            return None

    def create_fixed_urls_set(self, count: int = 200, output_file: str = None) -> List[str]:
        """
        Create a fixed set of Wikipedia URLs

        Args:
            count: Number of URLs to generate
            output_file: Optional file to save URLs

        Returns:
            List of Wikipedia URLs
        """
        logger.info(f"Generating {count} fixed Wikipedia URLs...")

        articles = self.get_random_wikipedia_urls(count=count)
        urls = [article['url'] for article in articles]

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(urls, f, indent=2, ensure_ascii=False)
            logger.info(f"[OK] Saved {len(urls)} URLs to {output_file}")

        return urls


def main():
    """Test data collection"""
    collector = WikipediaDataCollector(min_words=200)

    # Test with a few random URLs
    logger.info("Testing data collection...")
    urls = collector.get_random_wikipedia_urls(count=5)

    for url_data in urls:
        logger.info(f"  {url_data['title']}: {url_data['word_count']} words")


if __name__ == "__main__":
    main()
