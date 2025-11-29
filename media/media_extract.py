import requests
import time
import logging
from datetime import datetime, timedelta
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GDELTSampler:
    def __init__(self, start_year=2018, end_year=2025, files_per_week=3, delay=1.5):
        self.base_url = "http://data.gdeltproject.org/gdeltv2/"
        self.start_date = datetime(start_year, 1, 1)
        self.end_date = datetime(end_year, 12, 31)
        self.files_per_week = files_per_week
        self.delay = delay
        
    def generate_sample_times(self):
        """Generate sample datetimes: 3 random times per week"""
        samples = []
        current_date = self.start_date
        
        while current_date <= self.end_date:
            # Get the start and end of the current week
            week_start = current_date
            week_end = current_date + timedelta(days=7)
            
            # Generate 3 random times during this week
            for _ in range(self.files_per_week):
                # Random day in the week (0-6 days)
                random_days = random.randint(0, 6)
                # Random hour (0-23)
                random_hour = random.randint(0, 23)
                # Random 15-minute interval (0, 15, 30, 45)
                random_minute = random.choice([0, 15, 30, 45])
                
                sample_time = week_start + timedelta(
                    days=random_days,
                    hours=random_hour,
                    minutes=random_minute
                )
                
                # Make sure we don't go past end_date
                if sample_time <= self.end_date:
                    samples.append(sample_time)
            
            # Move to next week
            current_date += timedelta(days=7)
        
        logger.info(f"Generated {len(samples)} sample times from {self.start_date.year} to {self.end_date.year}")
        return sorted(samples)
    
    def datetime_to_url(self, dt):
        """Convert datetime to GDELT URL"""
        # Format: YYYYMMDDHHMMSS
        timestamp = dt.strftime('%Y%m%d%H%M%S')
        filename = f"{timestamp}.gkg.csv.zip"
        url = f"{self.base_url}{filename}"
        return url, filename
    
    def download_file(self, url, filename):
        """Download a single GKG file with retry logic"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    url, 
                    stream=True, 
                    timeout=120,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                
                # If file doesn't exist (404), skip it
                if response.status_code == 404:
                    logger.warning(f"File not found (404): {filename} - skipping")
                    return False
                
                response.raise_for_status()
                
                # Save the compressed file
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                logger.info(f"✓ Saved: {filename}")
                time.sleep(self.delay)
                return True
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    logger.warning(f"File not found: {filename} - skipping")
                    return False
                logger.warning(f"HTTP error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    logger.error(f"Failed after all retries: {filename}")
                    return False
                    
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError) as e:
                logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    logger.error(f"Failed after all retries: {filename}")
                    return False
                    
            except Exception as e:
                logger.error(f"Unexpected error downloading {filename}: {str(e)}")
                return False
    
    def run(self):
        """Run the complete sampling and download process"""
        try:
            # Generate sample times
            logger.info("Generating sample times...")
            sample_times = self.generate_sample_times()
            
            logger.info(f"\nWill attempt to download {len(sample_times)} files")
            logger.info(f"Estimated time: {len(sample_times) * self.delay / 60:.1f} minutes")
            logger.info(f"Note: Some files may not exist (404) and will be skipped\n")
            
            successful = 0
            failed = 0
            not_found = 0
            
            for i, sample_time in enumerate(sample_times, 1):
                url, filename = self.datetime_to_url(sample_time)
                
                logger.info(f"[{i}/{len(sample_times)}] {sample_time.strftime('%Y-%m-%d %H:%M')} - Progress: {i/len(sample_times)*100:.1f}%")
                
                result = self.download_file(url, filename)
                if result:
                    successful += 1
                elif result is False:  # 404 or skipped
                    not_found += 1
                else:
                    failed += 1
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Download complete!")
            logger.info(f"Successful: {successful}")
            logger.info(f"Not found (404): {not_found}")
            logger.info(f"Failed: {failed}")
            logger.info(f"Files saved to current directory")
            
        except Exception as e:
            logger.error(f"Error in sampling process: {str(e)}")
            raise


if __name__ == "__main__":
    # Sample 3 files per week from 2015-2025
    # GDELT 2.0 started February 2015, so files before that won't exist
    sampler = GDELTSampler(
        start_year=2015,
        end_year=2025,
        files_per_week=3,
        delay=1.5  # Be respectful to servers
    )
    
    sampler.run()