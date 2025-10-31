"""
SEC EDGAR Data Collector
Collects XBRL filings and financial data from SEC EDGAR database
"""

import os
import time
import requests
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import json
from pathlib import Path
import logging
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class SECDataCollector:
    """Collects SEC EDGAR filings and XBRL data"""
    
    def __init__(self, output_dir: str = "data/raw/sec"):
        self.base_url = "https://www.sec.gov"
        self.search_url = "https://www.sec.gov/edgar/searchedgar/companysearch"
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
        self.min_delay = 0.1  # Minimum delay between requests
    
    def _rate_limit(self):
        """Implement rate limiting to be respectful to SEC servers"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay:
            time.sleep(self.min_delay - time_since_last)
        self.last_request_time = time.time()
    
    def search_company(self, company_name: str, ticker: str = None) -> List[Dict]:
        """Search for a company in SEC EDGAR database"""
        self._rate_limit()
        
        params = {
            'company': company_name,
            'ticker': ticker or '',
            'type': '',
            'dateb': '',
            'owner': 'exclude',
            'start': '0',
            'count': '100'
        }
        
        try:
            response = requests.get(self.search_url, params=params, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            companies = []
            
            # Parse company search results
            for row in soup.find_all('tr', class_='company'):
                cells = row.find_all('td')
                if len(cells) >= 3:
                    company_info = {
                        'name': cells[0].get_text(strip=True),
                        'cik': cells[1].get_text(strip=True),
                        'state': cells[2].get_text(strip=True),
                        'business': cells[3].get_text(strip=True) if len(cells) > 3 else '',
                        'incorporated': cells[4].get_text(strip=True) if len(cells) > 4 else ''
                    }
                    companies.append(company_info)
            
            return companies
            
        except Exception as e:
            logger.error(f"Error searching for company {company_name}: {e}")
            return []
    
    def get_company_filings(self, cik: str, filing_types: List[str] = None) -> List[Dict]:
        """Get filings for a specific company by CIK"""
        if filing_types is None:
            filing_types = ['10-K', '10-Q', '8-K']
        
        self._rate_limit()
        
        filings_url = f"{self.base_url}/cgi-bin/browse-edgar"
        params = {
            'action': 'getcompany',
            'CIK': cik,
            'type': '|'.join(filing_types),
            'dateb': '',
            'owner': 'exclude',
            'start': '0',
            'count': '100'
        }
        
        try:
            response = requests.get(filings_url, params=params, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            filings = []
            
            # Parse filings table
            for row in soup.find_all('tr')[1:]:  # Skip header row
                cells = row.find_all('td')
                if len(cells) >= 4:
                    filing_info = {
                        'filing_type': cells[0].get_text(strip=True),
                        'filing_date': cells[3].get_text(strip=True),
                        'filing_url': urljoin(self.base_url, cells[1].find('a')['href']) if cells[1].find('a') else '',
                        'accession_number': cells[2].get_text(strip=True)
                    }
                    filings.append(filing_info)
            
            return filings
            
        except Exception as e:
            logger.error(f"Error getting filings for CIK {cik}: {e}")
            return []
    
    def get_xbrl_files(self, filing_url: str) -> List[Dict]:
        """Get XBRL files from a specific filing"""
        self._rate_limit()
        
        try:
            response = requests.get(filing_url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            xbrl_files = []
            
            # Look for XBRL-related files
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True)
                
                # Check for XBRL file extensions
                if any(href.endswith(ext) for ext in ['.xml', '.xsd', '.xbrl']):
                    xbrl_files.append({
                        'filename': text,
                        'url': urljoin(self.base_url, href),
                        'file_type': href.split('.')[-1]
                    })
                elif 'Interactive Data' in text or 'XBRL' in text:
                    xbrl_files.append({
                        'filename': text,
                        'url': urljoin(self.base_url, href),
                        'file_type': 'interactive'
                    })
            
            return xbrl_files
            
        except Exception as e:
            logger.error(f"Error getting XBRL files from {filing_url}: {e}")
            return []
    
    def download_xbrl_file(self, file_url: str, filename: str) -> bool:
        """Download an XBRL file"""
        self._rate_limit()
        
        try:
            response = requests.get(file_url, headers=self.headers)
            response.raise_for_status()
            
            file_path = self.output_dir / filename
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return False
    
    def parse_xbrl_xml(self, xml_file_path: str) -> Dict:
        """Parse XBRL XML file and extract financial data"""
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            
            # Define XBRL namespaces
            namespaces = {
                'xbrli': 'http://www.xbrl.org/2003/instance',
                'us-gaap': 'http://fasb.org/us-gaap/2023-01-31',
                'dei': 'http://xbrl.sec.gov/dei/2023-01-31'
            }
            
            financial_data = {}
            
            # Extract basic company information
            entity_name = root.find('.//dei:EntityRegistrantName', namespaces)
            if entity_name is not None:
                financial_data['entity_name'] = entity_name.text
            
            # Extract financial facts
            for fact in root.findall('.//xbrli:item', namespaces):
                context_ref = fact.get('contextRef')
                unit_ref = fact.get('unitRef')
                value = fact.text
                
                if value and context_ref:
                    # Get the concept name (local name without namespace)
                    concept_name = fact.tag.split('}')[-1] if '}' in fact.tag else fact.tag
                    
                    financial_data[concept_name] = {
                        'value': value,
                        'context_ref': context_ref,
                        'unit_ref': unit_ref
                    }
            
            return financial_data
            
        except Exception as e:
            logger.error(f"Error parsing XBRL file {xml_file_path}: {e}")
            return {}
    
    def collect_company_data(self, company_name: str, ticker: str = None, 
                           max_filings: int = 10) -> Dict:
        """Collect comprehensive data for a company"""
        logger.info(f"Collecting data for {company_name} ({ticker})")
        
        # Search for company
        companies = self.search_company(company_name, ticker)
        if not companies:
            logger.warning(f"No companies found for {company_name}")
            return {}
        
        # Use first match
        company = companies[0]
        cik = company['cik']
        
        # Get filings
        filings = self.get_company_filings(cik)
        if not filings:
            logger.warning(f"No filings found for {company_name}")
            return {}
        
        # Collect XBRL data from recent filings
        collected_data = {
            'company_info': company,
            'filings': filings[:max_filings],
            'xbrl_data': []
        }
        
        for filing in filings[:max_filings]:
            xbrl_files = self.get_xbrl_files(filing['filing_url'])
            
            for xbrl_file in xbrl_files:
                if xbrl_file['file_type'] in ['xml', 'xbrl']:
                    # Download and parse XBRL file
                    filename = f"{filing['accession_number']}_{xbrl_file['filename']}"
                    if self.download_xbrl_file(xbrl_file['url'], filename):
                        file_path = self.output_dir / filename
                        parsed_data = self.parse_xbrl_xml(str(file_path))
                        if parsed_data:
                            collected_data['xbrl_data'].append({
                                'filing': filing,
                                'file': xbrl_file,
                                'data': parsed_data
                            })
        
        return collected_data

def main():
    """Example usage"""
    logging.basicConfig(level=logging.INFO)
    
    collector = SECDataCollector()
    
    # Example: Collect data for Apple Inc.
    apple_data = collector.collect_company_data("Apple Inc.", "AAPL", max_filings=5)
    
    # Save collected data
    output_file = collector.output_dir / "apple_data.json"
    with open(output_file, 'w') as f:
        json.dump(apple_data, f, indent=2)
    
    print(f"Collected data saved to {output_file}")

if __name__ == "__main__":
    main()