import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
import time
from urllib.parse import urljoin

BASE_URL = "https://repository.chuka.ac.ke"
CSV_FILE = "d:/Jupyter notebook/chuka_exam_papers_metadata.csv"

def get_soup(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def extract_course_code(title):
    # Regex for common course codes like 'COSC 123', 'MATH 101', 'ZOOL 443'
    # Look for 3-4 letters followed by 3-4 digits, possibly with space
    match = re.search(r'([A-Z]{3,4})\s*(\d{3,4})', title, re.IGNORECASE)
    if match:
        return f"{match.group(1).upper()} {match.group(2)}"
    return ""

def search_via_api(base_url, query, max_items=25000):
    """Searches for items using DSpace 7 REST API."""
    # DSpace 7 API endpoint for discovery
    api_url = f"{base_url}/server/api/discover/search/objects"
    
    # We use a larger page size to minimize requests (Rate limit seems to be ~500)
    # 100 is typically a safe upper limit for DSpace pages
    params = {
        "dsoType": "ITEM",
        "size": 100, 
        "sort": "dc.date.issued,DESC"
    }
    if query:
        params["query"] = query
    
    page = 0
    total_fetched = 0
    
    # Initialize CSV if not exists (so we can append safely)
    if not os.path.exists(CSV_FILE):
        # Create empty CSV with headers
        pd.DataFrame(columns=["course_code", "year", "title", "repository_link", "scraped_date"]).to_csv(CSV_FILE, index=False)
    
    while total_fetched < max_items:
        params['page'] = page
        print(f"Fetching API page {page} (Total fetched so far: {total_fetched})...")
        try:
            # CLAUDE FEEDBACK NOTE: verify=False is required here because the 
            # Chuka University server has an invalid/self-signed SSL certificate.
            # Without this, the script will crash with SSLError/CertificateError.
            # We suppress the warning in production code usually, but here we just accept it.
            resp = requests.get(api_url, params=params, verify=False) 
            resp.raise_for_status()
            
            # Check Rate Limits
            remaining = resp.headers.get('X-RateLimit-Remaining')
            if remaining and int(remaining) < 20:
                print(f"WARNING: Rate limit low ({remaining}). Sleeping for 60 seconds...")
                time.sleep(60)
            
            data = resp.json()
            
            # Navigate JSON structure: _embedded -> searchResult -> _embedded -> objects
            search_result = data.get('_embedded', {}).get('searchResult', {})
            objects = search_result.get('_embedded', {}).get('objects', [])
            
            if not objects:
                print("No more objects found.")
                break
            
            page_items = []
            for obj in objects:
                item = obj.get('_embedded', {}).get('indexableObject', {})
                metadata = item.get('metadata', {})
                
                # Extract title
                title_list = metadata.get('dc.title', [])
                title = title_list[0].get('value') if title_list else "Untitled"
                
                # Extract date
                date_list = metadata.get('dc.date.issued', [])
                date = date_list[0].get('value') if date_list else ""
                
                # Extract UUID/Handle
                uuid = item.get('uuid')
                handle = item.get('handle')
                
                # Construct link (frontend link)
                # DSpace 7 frontend is usually /items/{uuid} or /handle/{handle}
                link = f"{base_url}/items/{uuid}" if uuid else ""
                
                course_code = extract_course_code(title)
                
                # Append ALL items, regardless of course code
                page_items.append({
                    "course_code": course_code, # May be empty
                    "year": date,
                    "title": title,
                    "repository_link": link,
                    "scraped_date": pd.Timestamp.now().isoformat()
                })
            
            # SAVE INCREMENTALLY
            # This ensures that if the script crashes or is stopped, we don't lose progress.
            if page_items:
                df_page = pd.DataFrame(page_items)
                # Append to CSV. Header=False because file already exists (created at start or existing)
                df_page.to_csv(CSV_FILE, mode='a', header=False, index=False)
                # Deduplicate on the fly is expensive, so we'll just append and user can dedup later
                # actually, let's keep it simple: just append. 
                print(f"  -> Saved {len(page_items)} items to {CSV_FILE}")

            total_fetched += len(objects)
            page += 1
            # Polite API spacing
            time.sleep(1)
            
        except Exception as e:
            print(f"API Error at page {page}: {e}")
            # If rate limited (429) or error, pause and retry logic could go here
            # For now, we wait a bit and retry the same loop (or break if strict)
            time.sleep(5)
            # Break to avoid infinite loops on hard errors
            break
            
    return total_fetched

def main():
    print("Starting API-based scrape for ALL repository items...")
    print("NOTE: Data is saved incrementally. You can stop the script at any time.")
    
    # 2. Search API
    # Empty query "" generally fetches all items in DSpace discovery
    print(f"Fetching all items...")
    
    # We verify if we are appending or creating new
    if os.path.exists(CSV_FILE):
        print(f"Appending to existing file: {CSV_FILE}")
    else:
        print(f"Creating new file: {CSV_FILE}")

    total = search_via_api(BASE_URL, query="", max_items=25000)
    
    print(f"Finished. Total items fetched this session: {total}")

if __name__ == "__main__":
    main()
