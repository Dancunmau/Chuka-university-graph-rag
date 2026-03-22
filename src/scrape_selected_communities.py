"""
Scrape Specific Communities (Excluding Faculties and Examinations)
"""
import requests
import pandas as pd
import json
import time
import os

BASE_URL = "https://repository.chuka.ac.ke"
OUTPUT_FILE = r'd:/Jupyter notebook/Graph rag/data/communities.csv'

def load_communities():
    """Load the communities list"""
    with open('d:/Jupyter notebook/communities_list.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def scrape_community(community_uuid, community_name, max_items=10000):
    """Scrape all items from a specific community"""
    print(f"\n{'='*60}")
    print(f"Scraping: {community_name}")
    print('='*60)
    
    api_url = f"{BASE_URL}/server/api/discover/search/objects"
    
    params = {
        "scope": community_uuid,
        "dsoType": "ITEM",
        "size": 100,
        "sort": "dc.date.issued,DESC"
    }
    
    items = []
    page = 0
    total_fetched = 0
    
    while total_fetched < max_items:
        params['page'] = page
        print(f"  Page {page} (fetched: {total_fetched})...", end='\r')
        
        try:
            resp = requests.get(api_url, params=params, verify=False, timeout=15)
            resp.raise_for_status()
            
            remaining = resp.headers.get('X-RateLimit-Remaining')
            if remaining and int(remaining) < 20:
                print(f"\n  Rate limit low ({remaining}). Sleeping 60s...")
                time.sleep(60)
            
            data = resp.json()
            search_result = data.get('_embedded', {}).get('searchResult', {})
            objects = search_result.get('_embedded', {}).get('objects', [])
            
            if not objects:
                break
            
            for obj in objects:
                item = obj.get('_embedded', {}).get('indexableObject', {})
                metadata = item.get('metadata', {})
                
                title_list = metadata.get('dc.title', [])
                title = title_list[0].get('value') if title_list else "Untitled"
                
                date_list = metadata.get('dc.date.issued', [])
                date = date_list[0].get('value') if date_list else ""
                
                author_list = metadata.get('dc.contributor.author', [])
                author = author_list[0].get('value') if author_list else ""
                
                type_list = metadata.get('dc.type', [])
                item_type = type_list[0].get('value') if type_list else ""
                
                uuid = item.get('uuid')
                handle = item.get('handle')
                link = f"{BASE_URL}/items/{uuid}" if uuid else ""
                
                items.append({
                    "community": community_name,
                    "community_uuid": community_uuid,
                    "title": title,
                    "author": author,
                    "type": item_type,
                    "year": date,
                    "handle": handle,
                    "repository_link": link,
                    "scraped_date": pd.Timestamp.now().isoformat()
                })
            
            total_fetched += len(objects)
            page += 1
            time.sleep(1)
            
        except Exception as e:
            print(f"\n  Error at page {page}: {e}")
            time.sleep(5)
            break
    
    print(f"\n  ✓ Fetched {total_fetched} items")
    return items

def main():
    print("="*60)
    print("SCRAPING SELECTED COMMUNITIES")
    print("(Excluding Faculties and Examinations)")
    print("="*60)
    
    # Load communities
    communities = load_communities()
    print(f"\nTotal communities: {len(communities)}")
    
    # Filter out faculty and examination communities
    exclude_keywords = [
        "faculty",
        "examination",
        "exam paper",
        "past paper",
        "school of"  # Also exclude "School of X",similar to faculties
    ]
    
    filtered_communities = []
    excluded = []
    
    for comm in communities:
        name_lower = comm['name'].lower()
        if any(keyword in name_lower for keyword in exclude_keywords):
            excluded.append(comm['name'])
        else:
            filtered_communities.append(comm)
    
    print(f"\nExcluded {len(excluded)} communities:")
    for name in sorted(excluded):
        print(f"  - {name}")
    
    print(f"\n{'='*60}")
    print(f"Communities to scrape: {len(filtered_communities)}")
    print('='*60)
    for i, comm in enumerate(filtered_communities, 1):
        print(f"{i}. {comm['name']}")
    
    print(f"\n{'='*60}")
    print("Starting scrape...")
    print("(Data will be saved incrementally)")
    print('='*60)
    
    # load existing data
    if os.path.exists(OUTPUT_FILE):
        print(f"\nAppending to existing file: {OUTPUT_FILE}")
        df_existing = pd.read_csv(OUTPUT_FILE)
        all_items = df_existing.to_dict('records')
    else:
        print(f"\nCreating new file: {OUTPUT_FILE}")
        all_items = []
    
    # Scrape each community
    for i, comm in enumerate(filtered_communities, 1):
        print(f"\n[{i}/{len(filtered_communities)}]")
        
        items = scrape_community(comm['uuid'], comm['name'])
        all_items.extend(items)
        
        # Save incrementally
        df = pd.DataFrame(all_items)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"  Progress saved. Total items: {len(all_items)}")
        
        time.sleep(2)
    
    print("\n" + "="*60)
    print("SCRAPING COMPLETE!")
    print("="*60)
    print(f"\nTotal items scraped: {len(all_items)}")
    print(f"Output file: {OUTPUT_FILE}")
    
    # Summary by community
    df_final = pd.DataFrame(all_items)
    print("\nItems per community:")
    print(df_final['community'].value_counts())

if __name__ == "__main__":
    main()
