# web_scraper.py
"""
Script to scrape images from the web for user-specified prompts.
Saves images and metadata to the data/raw directory.
"""
import os
import requests
import json
import hashlib
from typing import List, Dict
from datetime import datetime
from urllib.parse import quote
import time

def scrape_bing_images(query: str, num_images: int = 50) -> List[Dict]:
    """
    Scrape image URLs from Bing Image Search.
    Args:
        query (str): Search term.
        num_images (int): Number of images to scrape.
    Returns:
        List of dictionaries containing image URLs and metadata.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    images_data = []
    base_url = f"https://www.bing.com/images/search?q={quote(query)}&first="
    
    print(f"[INFO] Scraping Bing for '{query}'...")
    
    # Bing shows ~35 images per page
    pages_needed = (num_images // 35) + 1
    
    for page in range(pages_needed):
        try:
            url = base_url + str(page * 35)
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Extract image URLs from the page
                # This is a simplified extraction - you may need to parse HTML properly
                import re
                img_urls = re.findall(r'murl&quot;:&quot;(.*?)&quot;', response.text)
                
                for img_url in img_urls[:num_images - len(images_data)]:
                    images_data.append({
                        'url': img_url,
                        'query': query,
                        'source': 'bing',
                        'scraped_at': datetime.now().isoformat()
                    })
                    
                if len(images_data) >= num_images:
                    break
                    
            time.sleep(1)  # Be respectful with requests
            
        except Exception as e:
            print(f"[WARNING] Error scraping page {page}: {e}")
            continue
    
    print(f"[INFO] Found {len(images_data)} image URLs")
    return images_data

def download_image(img_data: Dict, save_dir: str, index: int) -> bool:
    """
    Download a single image and save metadata.
    Args:
        img_data (dict): Image metadata including URL.
        save_dir (str): Directory to save the image.
        index (int): Image index for naming.
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        response = requests.get(img_data['url'], timeout=10, stream=True)
        
        if response.status_code == 200:
            # Get file extension from content-type
            content_type = response.headers.get('content-type', '')
            ext = 'jpg'
            if 'png' in content_type:
                ext = 'png'
            elif 'jpeg' in content_type or 'jpg' in content_type:
                ext = 'jpg'
            elif 'webp' in content_type:
                ext = 'webp'
            
            # Create filename
            query_safe = img_data['query'].replace(' ', '_').replace('/', '_')
            filename = f"{query_safe}_{index:04d}.{ext}"
            filepath = os.path.join(save_dir, filename)
            
            # Save image
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            
            # Save metadata
            img_data['local_path'] = filepath
            img_data['filename'] = filename
            img_data['file_size'] = os.path.getsize(filepath)
            
            return True
    except Exception as e:
        print(f"[WARNING] Failed to download image {index}: {e}")
        return False
    
    return False

def download_images(query: str, num_images: int = 50, save_dir: str = "../data/raw"):
    """
    Download images from Bing Image Search for a given query.
    Args:
        query (str): Search term.
        num_images (int): Number of images to download.
        save_dir (str): Directory to save images.
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Scrape image URLs
    images_data = scrape_bing_images(query, num_images)
    
    if not images_data:
        print("[ERROR] No images found!")
        return
    
    # Download images
    print(f"[INFO] Downloading {len(images_data)} images...")
    successful_downloads = []
    
    for idx, img_data in enumerate(images_data, 1):
        print(f"[INFO] Downloading {idx}/{len(images_data)}...", end='\r')
        if download_image(img_data, save_dir, idx):
            successful_downloads.append(img_data)
        time.sleep(0.5)  # Rate limiting
    
    print(f"\n[INFO] Successfully downloaded {len(successful_downloads)}/{len(images_data)} images")
    
    # Save metadata to JSON
    metadata_path = os.path.join(save_dir, f"{query.replace(' ', '_')}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(successful_downloads, f, indent=2)
    
    print(f"[INFO] Metadata saved to {metadata_path}")

if __name__ == "__main__":
    prompt = input("Enter object prompt to search for: ")
    num = input("Number of images to download (default 50): ")
    num_images = int(num) if num.strip() else 50
    
    download_images(prompt, num_images)
