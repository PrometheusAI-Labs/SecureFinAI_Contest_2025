#!/usr/bin/env python3
"""
XBRL Data Collector
Collects XBRL specifications and documentation from xbrl.org
Based on xbrl_webcrawl.ipynb
"""

import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from pathlib import Path
import logging
from typing import List, Dict, Set, Optional
import argparse

logger = logging.getLogger(__name__)

class XBRLDataCollector:
    """Collects XBRL specifications and documentation from xbrl.org"""
    
    def __init__(self, output_dir: str = "data/raw/xbrl"):
        self.base_url = "https://www.xbrl.org"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Headers to mimic browser request
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_delay = 1.0  # Minimum delay between requests
        
        # Track visited URLs
        self.visited_urls: Set[str] = set()
        self.internal_urls: Set[str] = set()
    
    def _rate_limit(self):
        """Implement rate limiting to be respectful to xbrl.org servers"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay:
            time.sleep(self.min_delay - time_since_last)
        self.last_request_time = time.time()
    
    def is_valid(self, url: str, allow_external: bool = True) -> bool:
        """Check if URL is valid and allowed"""
        parsed_url = urlparse(url)
        if allow_external:
            # Define allowed external domains
            allowed_domains = ['xbrl.org', 'specifications.xbrl.org']
            return any(parsed_url.netloc.endswith(domain) for domain in allowed_domains)
        else:
            return bool(parsed_url.netloc) and bool(parsed_url.scheme)
    
    def get_all_website_links(self, url: str) -> Set[str]:
        """Get all valid links from a webpage"""
        try:
            urls = set()
            domain_name = urlparse(url).netloc
            session = requests.Session()
            session.headers.update(self.headers)
            
            self._rate_limit()
            response = session.get(url)
            if response.status_code != 200:
                return urls  # Return empty set if failed to fetch the page
            
            soup = BeautifulSoup(response.text, 'html.parser')
            for a_tag in soup.findAll("a"):
                href = a_tag.attrs.get("href")
                if href == "" or href is None:
                    continue
                href = urljoin(url, href)
                parsed_href = urlparse(href)
                href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
                if not self.is_valid(href):
                    continue
                if href in self.internal_urls:
                    continue
                if domain_name not in href:
                    continue
                urls.add(href)
                self.internal_urls.add(href)
            return urls
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return set()
    
    def extract_details(self, url: str) -> Optional[Dict]:
        """Extract details from a webpage"""
        try:
            self._rate_limit()
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                logger.warning(f'Failed to retrieve the webpage. URL: {url}, Status code: {response.status_code}')
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('title').text if soup.find('title') else ''
            text = soup.get_text(separator=' ', strip=True)
            
            # Extract XBRL-specific content
            xbrl_content = self.extract_xbrl_content(soup)
            
            return {
                "title": title,
                "url": url,
                "text": text,
                "xbrl_content": xbrl_content,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return None
    
    def extract_xbrl_content(self, soup: BeautifulSoup) -> Dict:
        """Extract XBRL-specific content from the page"""
        xbrl_content = {
            "specifications": [],
            "taxonomies": [],
            "examples": [],
            "definitions": []
        }
        
        # Look for XBRL specifications
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)
            
            if 'specification' in href.lower() or 'specification' in text.lower():
                xbrl_content["specifications"].append({
                    "title": text,
                    "url": href
                })
            elif 'taxonomy' in href.lower() or 'taxonomy' in text.lower():
                xbrl_content["taxonomies"].append({
                    "title": text,
                    "url": href
                })
            elif 'example' in href.lower() or 'example' in text.lower():
                xbrl_content["examples"].append({
                    "title": text,
                    "url": href
                })
        
        # Look for XBRL definitions and concepts
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
            text = element.get_text(strip=True)
            if any(keyword in text.lower() for keyword in ['xbrl', 'extensible business reporting language', 'tag', 'element']):
                xbrl_content["definitions"].append(text)
        
        return xbrl_content
    
    def crawl(self, url: str, results: List[Dict], max_depth: int = 3) -> None:
        """Crawl XBRL website starting from given URL"""
        if url in self.visited_urls or max_depth <= 0:
            return
        
        self.visited_urls.add(url)
        logger.info(f"Visiting: {url}")
        
        details = self.extract_details(url)
        if details:
            details['id'] = len(results)
            results.append(details)
        
        links = self.get_all_website_links(url)
        if links:
            for link in links:
                self.crawl(link, results, max_depth - 1)
    
    def create_training_data(self, collected_data: List[Dict]) -> List[Dict]:
        """Convert collected XBRL data to training format"""
        training_data = []
        
        for item in collected_data:
            # Create examples for different XBRL tasks
            xbrl_content = item.get('xbrl_content', {})
            
            # Tag extraction examples
            for spec in xbrl_content.get('specifications', []):
                training_data.append({
                    'instruction': 'Extract XBRL tag information from the specification.',
                    'input': f"XBRL Specification: {spec['title']}\nURL: {spec['url']}",
                    'output': f"XBRL Tag: {spec['title']}",
                    'task': 'xbrl_tag_extraction',
                    'source': 'xbrl_specifications'
                })
            
            # Value extraction examples
            for taxonomy in xbrl_content.get('taxonomies', []):
                training_data.append({
                    'instruction': 'Extract XBRL taxonomy values and definitions.',
                    'input': f"XBRL Taxonomy: {taxonomy['title']}\nURL: {taxonomy['url']}",
                    'output': f"Taxonomy Value: {taxonomy['title']}",
                    'task': 'xbrl_value_extraction',
                    'source': 'xbrl_taxonomies'
                })
            
            # Formula construction examples
            for example in xbrl_content.get('examples', []):
                training_data.append({
                    'instruction': 'Construct XBRL formula based on the example.',
                    'input': f"XBRL Example: {example['title']}\nURL: {example['url']}",
                    'output': f"Formula: Based on {example['title']}",
                    'task': 'xbrl_formula_construction',
                    'source': 'xbrl_examples'
                })
            
            # Definition extraction
            for definition in xbrl_content.get('definitions', []):
                if len(definition) > 50:  # Only use substantial definitions
                    training_data.append({
                        'instruction': 'Extract XBRL concept definition.',
                        'input': f"XBRL Text: {definition[:200]}...",
                        'output': f"Definition: {definition}",
                        'task': 'xbrl_definition_extraction',
                        'source': 'xbrl_definitions'
                    })
        
        return training_data
    
    def collect_xbrl_data(self, max_pages: int = 50) -> Dict:
        """Collect XBRL data from xbrl.org"""
        logger.info("Starting XBRL data collection")
        
        results = []
        
        # Start crawling from main XBRL pages
        start_urls = [
            "https://www.xbrl.org",
            "https://www.xbrl.org/specifications/",
            "https://www.xbrl.org/taxonomies/",
            "https://www.xbrl.org/examples/"
        ]
        
        for url in start_urls:
            if len(results) >= max_pages:
                break
            self.crawl(url, results, max_depth=2)
        
        # Create training data
        training_data = self.create_training_data(results)
        
        # Save collected data
        collected_data = {
            'raw_data': results,
            'training_data': training_data,
            'collection_stats': {
                'total_pages': len(results),
                'training_examples': len(training_data),
                'visited_urls': len(self.visited_urls),
                'internal_urls': len(self.internal_urls)
            }
        }
        
        # Save to files
        raw_file = self.output_dir / "xbrl_collected_data.json"
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(collected_data, f, indent=2, ensure_ascii=False)
        
        # Save training data in JSONL format
        training_file = self.output_dir / "xbrl_training_data.jsonl"
        with open(training_file, 'w', encoding='utf-8') as f:
            for example in training_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Collected {len(results)} pages and created {len(training_data)} training examples")
        logger.info(f"Data saved to {raw_file} and {training_file}")
        
        return collected_data

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Collect XBRL data from xbrl.org')
    parser.add_argument('--output-dir', default='data/raw/xbrl', help='Output directory')
    parser.add_argument('--max-pages', type=int, default=50, help='Maximum pages to collect')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Collect data
    collector = XBRLDataCollector(args.output_dir)
    data = collector.collect_xbrl_data(args.max_pages)
    
    print(f"Collection completed!")
    print(f"Total pages: {data['collection_stats']['total_pages']}")
    print(f"Training examples: {data['collection_stats']['training_examples']}")

if __name__ == "__main__":
    main()