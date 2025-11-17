import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import json
from datetime import datetime
import re
from urllib.parse import urljoin
import logging
from typing import List, Dict, Optional
import pickle
import random
from requests.adapters import HTTPAdapter

class LitigationCrawler:
    """
    A web crawler for collecting litigation releases from SEC, FBI, and US SDNY.
    Based on the methodology described in "A Deep Learning Approach to the Detection of Illegal Insider Trading"
    """
    
    def __init__(self, output_dir: str = "litigation_data", delay: float = 5.0):

        self.output_dir = output_dir
        self.delay = delay  

        self.session = requests.Session()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{output_dir}/crawler.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.sessionHeads = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36'
        ]
        
        self.sec_data = []
        self.fbi_data = []
        self.sdny_data = []

    def setup_session(self) -> Optional[requests.Session]:
        self.session.headers.update({
        'Connection': 'keep-alive',
        'Keep-Alive': 'timeout=20, max=1000'
        })
    
        self.session.mount('https://', HTTPAdapter(pool_connections=1, pool_maxsize=1))
         

    def refresh_session(self):
        self.session.close()
        self.session = requests.Session()
        self.setup_session()

        


    def safe_request(self, url: str, max_retries: int = 5) -> Optional[requests.Response]:
        """Make a safe HTTP request with retries and error handling."""
        for attempt in range(max_retries):
            try:
                headers = {'User-Agent': random.choice(self.sessionHeads)}
                response = self.session.get(url, timeout=120, headers=headers)                
                response.raise_for_status()
                time.sleep(self.delay)  
                if attempt > 0:
                    self.refresh_session()
                return response
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                self.refresh_session()
                if attempt < max_retries - 1:
                    time.sleep(self.delay + random.uniform(5.0, 10.0)) 
        return None

    def save_raw_data(self, data: List[Dict], source: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = f"{self.output_dir}/raw/{source}_raw_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        pickle_file = f"{self.output_dir}/raw/{source}_raw_{timestamp}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Saved {len(data)} {source} records to {json_file}")
    def fetch_filing_text(self, filing_url: str) -> str:
        response = self.safe_request(filing_url)
        if response and response.status_code == 200:
            return response.text.strip()
        else:
            return ""
    def crawl_sec_data(self, start_year: int = 2022, end_year: int = 2024) -> List[Dict]:
        self.logger.info("Starting EDGAR master index crawl...")

        base_url = "https://www.sec.gov/Archives/edgar/full-index"
        insider_forms = {'3', '4', '5'}

        all_records = []

        for year in range(start_year, end_year + 1):
            for quarter in ['QTR1', 'QTR2', 'QTR3', 'QTR4']:
                index_url = f"{base_url}/{year}/{quarter}/master.idx"

                self.logger.info(f"Fetching: {index_url}")
                response = self.safe_request(index_url)
                if not response:
                    self.logger.warning(f"Failed to retrieve {index_url}")
                    continue

                lines = response.text.splitlines()
                if len(lines) <= 11:
                    continue
                data_lines = lines[11:]

                for line in data_lines:
                    parts = line.strip().split('|')
                    if len(parts) != 5:
                        continue

                    cik, company_name, form_type, date_filed, filename = parts
                    if form_type.upper() not in insider_forms:
                        continue

                    filing_url = f"https://www.sec.gov/Archives/{filename}"

                    filing_text = self.fetch_filing_text(filing_url)

                    record = {
                        'year': year,
                        'quarter': quarter,
                        'cik': cik,
                        'company_name': company_name,
                        'form_type': form_type,
                        'date_filed': date_filed,
                        'url': filing_url,
                        'scraped_date': datetime.now().isoformat(),
                        'filing_text': filing_text
                    }

                    all_records.append(record)

                self.logger.info(f"Collected {len(all_records)} records so far...")
        self.save_json(all_records)
        return all_records

    def save_json(self, data: List[Dict]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"edgar_index_raw_{timestamp}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved {len(data)} records to {output_file}")

    def crawl_fbi_white_collar(self) -> List[Dict]:
        self.logger.info("Starting FBI white collar crime crawl...")
        fbi_data = []
        
        base_url = "https://www.fbi.gov/investigate/white-collar-crime/news"
        
        response = self.safe_request(base_url)
        if not response:
            return fbi_data
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles = soup.find_all(['article', 'div'], class_=re.compile(r'news|press|release|story'))
        
        for article in articles:
            article_data = self.parse_fbi_article(article)
            if article_data:
                fbi_data.append(article_data)
        
        for page in range(2, 11): 
            page_url = f"{base_url}?page={page}"
            response = self.safe_request(page_url)
            
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                articles = soup.find_all(['article', 'div'], class_=re.compile(r'news|press|release|story'))
                
                if not articles:  
                    break
                
                for article in articles:
                    article_data = self.parse_fbi_article(article)
                    if article_data:
                        fbi_data.append(article_data)
        
        self.fbi_data = fbi_data
        self.save_raw_data(fbi_data, "fbi")
        return fbi_data


    def create_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Convert raw scraped data into organized pandas DataFrames.
        """
        self.logger.info("Creating organized DataFrames...")
        
        dataframes = {}
        
        if self.sec_data:
            df_sec = pd.DataFrame(self.sec_data)
            df_sec['date_scraped'] = pd.to_datetime(df_sec['scraped_date'])
            dataframes['sec'] = df_sec

        all_data = []
        all_data.extend(self.sec_data)
        if all_data:
            df_combined = pd.DataFrame(all_data)
            df_combined['date_scraped'] = pd.to_datetime(df_combined['scraped_date'])
            
            df_combined['title_length'] = df_combined['title'].str.len()
            df_combined['text_length'] = df_combined['raw_text'].str.len()
            df_combined['insider_in_title'] = df_combined['title'].str.lower().str.contains('insider', na=False)
            df_combined['insider_in_text'] = df_combined['raw_text'].str.lower().str.contains('insider', na=False)
            
            df_combined['initial_classification'] = df_combined.apply(
                lambda x: 'Possible_Insider' if x['has_insider_keyword'] else 'Non_Insider', 
                axis=1
            )
            
            dataframes['combined'] = df_combined
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, df in dataframes.items():
            csv_file = f"{self.output_dir}/processed/{name}_dataframe_{timestamp}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
            excel_file = f"{self.output_dir}/processed/{name}_dataframe_{timestamp}.xlsx"
            df.to_excel(excel_file, index=False)
            
            self.logger.info(f"Saved {name} DataFrame with {len(df)} records")
        
        return dataframes


    def run_full_crawl(self) -> Dict[str, pd.DataFrame]:
        self.logger.info("Starting full litigation data crawl...")
        start_time = datetime.now()
        
        try:
            # crawl and update data
            self.crawl_sec_data()
          ##  self.crawl_fbi_white_collar()
            dataframes = self.create_dataframes()
         ##   self.generate_summary_report(dataframes)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info(f"Full crawl completed in {duration}")
            self.logger.info(f"Total records collected: {sum(len(df) for df in dataframes.values() if df is not dataframes.get('combined'))}")
            
            return dataframes
            
        except Exception as e:
            self.logger.error(f"Crawl failed with error: {e}")
            raise


if __name__ == "__main__":
    crawler = LitigationCrawler(
        output_dir="litigation_data",
        delay=1.5  # Be respectful to servers
    )
    
    try:
        dataframes = crawler.run_full_crawl()
        
        # Access individual DataFrames
        if 'combined' in dataframes:
            df = dataframes['combined']
            print(f"\nCombined DataFrame shape: {df.shape}")
            print("\nColumns:", df.columns.tolist())
            print("\nFirst few records:")
            print(df[['source', 'title', 'has_insider_keyword']].head())
            
            # Show insider trading related records
            insider_records = df[df['has_insider_keyword'] == True]
            print(f"\nFound {len(insider_records)} potential insider trading records")
            
    except Exception as e:
        print(f"Crawling failed: {e}")
        import traceback
        traceback.print_exc()