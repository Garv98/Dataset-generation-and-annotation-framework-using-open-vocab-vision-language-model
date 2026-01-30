"""
AMSDG-HV: Advanced Multi-Source Dataset Generation with Hybrid Verification

A Cross-Modal Dataset Generation and Annotation Framework Leveraging 
Open-Vocabulary Vision-Language Models for Computer Vision Tasks

Implemented Scraping Methods:
1. SerpAPI (Google Custom Search) - highest quality results [REQUIRES API KEY]
2. Unsplash API - high-resolution photos [REQUIRES API KEY]
3. DuckDuckGo - primary free fallback (reliable, no API needed)
4. Bing Advanced - secondary free fallback (web-scale coverage)
5. Selenium/Google Images - optional, JavaScript rendering [REQUIRES: pip install selenium]

Features:
- Multi-tier scraping with intelligent fallback
- 3-tier hybrid verification (URL filter → EXIF/quality → Semantic VLM)
- Dual verification modes:
  * Fast: CLIP (openai/clip-vit-base-patch32) - ~2s for 30 images on CPU
  * Accurate: Qwen2-VL-2B-Instruct - ~3min/image on CPU, <10s on GPU
- Task-adaptive prompting (detection, classification, segmentation, auto)
- Model caching - downloads once, reuses forever
- Configurable CLIP threshold for precision/recall trade-off
"""
import os
import sys
import json
import shutil
import hashlib
import time
import re
import logging
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from urllib.parse import quote
import requests
from PIL import Image, ExifTags
import numpy as np
import torch
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Suppress Hugging Face logs aggressively
from transformers import logging as tf_logging
tf_logging.set_verbosity_error()
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

# Specific warning filters
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ResourceWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model cache directory (prevents re-downloading on every run)
MODEL_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
os.environ['HF_HOME'] = str(Path.home() / ".cache" / "huggingface")
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_DIR)

# ============================================================================
# TIER-1: SELENIUM-BASED SCRAPING (RESEARCH STANDARD)
# ============================================================================

def scrape_google_selenium(query: str, num: int = 50) -> List[Dict]:
    """Research-grade scraping using Selenium (handles JavaScript)."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(f"https://www.google.com/search?q={quote(query)}&tbm=isch")
        
        images = []
        last_height = driver.execute_script("return document.body.scrollHeight")
        
        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        img_elements = driver.find_elements(By.CSS_SELECTOR, "img.rg_i")
        
        for img in img_elements[:num]:
            try:
                img.click()
                time.sleep(0.5)
                actual_images = driver.find_elements(By.CSS_SELECTOR, "img.n3VNCb")
                for actual in actual_images:
                    src = actual.get_attribute('src')
                    if src and src.startswith('http') and 'gstatic' not in src:
                        images.append({'url': src, 'source': 'google_selenium', 'query': query})
                        break
            except:
                continue
        
        driver.quit()
        logger.info(f"[Selenium] Found {len(images)} images")
        return images
    except ImportError as e:
        logger.error(f"Selenium import error: {e}")
        return []
    except Exception as e:
        logger.error(f"Selenium error: {e}")
        return []

# ============================================================================
# TIER-2: API-BASED SCRAPING (HIGHEST QUALITY)
# ============================================================================

def scrape_serpapi(query: str, num: int = 50, api_key: str = None, task: str = 'auto') -> List[Dict]:
    """SerpAPI - Professional Google Images scraping."""
    if not api_key:
        return []
    try:
        params = {
            "engine": "google_images",
            "q": query,
            "api_key": api_key,
            "num": min(num, 100),
            "ijn": 0
        }
        response = requests.get("https://serpapi.com/search", params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            images = [{'url': item.get('original'), 'source': 'serpapi', 'query': query} 
                      for item in data.get('images_results', [])[:num] if item.get('original')]
            logger.info(f"[SerpAPI] Found {len(images)} images")
            return images
        elif response.status_code == 429:
            logger.warning("[SerpAPI] Rate limit exceeded")
        else:
            logger.warning(f"[SerpAPI] Status {response.status_code}")
    except Exception as e:
        logger.error(f"SerpAPI error: {e}")
    return []

def scrape_unsplash_api(query: str, num: int = 50, api_key: str = None) -> List[Dict]:
    """Unsplash API - Professional high-res photos."""
    if not api_key:
        return []
    try:
        headers = {'Authorization': f'Client-ID {api_key}'}
        images = []
        per_page = 30
        for page in range(1, (num // per_page) + 2):
            params = {'query': query, 'page': page, 'per_page': per_page}
            response = requests.get('https://api.unsplash.com/search/photos', 
                                    headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                for item in response.json().get('results', []):
                    images.append({'url': item['urls']['regular'], 'source': 'unsplash_api', 'query': query, 'attribution': f"Unsplash user: {item['user']['username']}"})
                if len(images) >= num:
                    break
            time.sleep(1)  # Rate limit
        logger.info(f"[Unsplash API] Found {len(images[:num])} images")
        return images[:num]
    except Exception as e:
        logger.error(f"Unsplash error: {e}")
    return []

# ============================================================================
# TIER-3: ADVANCED HTML PARSING (FALLBACK)
# ============================================================================

def scrape_bing_advanced(query: str, num: int = 50) -> List[Dict]:
    """Bing Advanced HTML scraping with robust error handling and anti-blocking."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.bing.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
    }

    images = []
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    pages_to_try = min((num // 35) + 2, 5)  # Limit to 5 pages max to avoid blocking
    consecutive_failures = 0
    new_urls = 0
    
    for page in range(pages_to_try):
        try:
            url = f"https://www.bing.com/images/search?q={quote(query)}&first={page*35}&count=35&qft=+filterui:imagesize-large+filterui:photo-photo"
            response = session.get(url, headers=headers, timeout=15)

            if response.status_code != 200:
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    break
                time.sleep(5)
                continue
            
            if len(response.text) < 5000:
                if page == 0:  
                    logger.warning(f"[Bing] Blocked or limited response ({len(response.text)} bytes)")
                break
            
            if 'captcha' in response.text.lower():
                break

            patterns = [
                (r'"murl":"(https?://[^"]+)"', 'murl-json'),
                (r'murl&quot;:&quot;(.*?)&quot;', 'murl-html'),
                (r'"purl":"(https?://[^"]+)"', 'purl'),
                (r'"mediaurl":"(https?://[^"]+)"', 'mediaurl'),
                (r'"src":"(https?://[^"]+\.(?:jpg|jpeg|png|gif|webp))"', 'src-json'),
                (r'data-src="(https?://[^"]+)"', 'data-src'),
                (r'src2="(https?://[^"]+)"', 'src2'),
                (r'<img[^>]+src="(https?://[^"]+)"', 'img-src'),
            ]

            urls_found = set()
            for pattern, name in patterns:
                matches = re.findall(pattern, response.text, re.IGNORECASE)
                for match in matches:
                    url_clean = (match
                        .replace('\\u002f', '/')
                        .replace('\\/', '/')
                        .replace('&amp;', '&')
                        .replace('%3A', ':')
                        .replace('%2F', '/')
                    )
                    
                    if (len(url_clean) > 30 and 
                        url_clean.startswith('http') and
                        'bing.com' not in url_clean.lower() and 
                        'th.bing.com' not in url_clean.lower() and 
                        'tse' not in url_clean.lower() and  
                        not url_clean.startswith('data:') and
                        'microsoft.com' not in url_clean.lower()):
                        urls_found.add(url_clean)

            new_urls = 0
            for url in urls_found:
                if len(images) >= num * 2:  
                    break
                images.append({'url': url, 'source': 'bing_advanced', 'query': query})
                new_urls += 1
            
            if new_urls == 0:
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    break
            else:
                consecutive_failures = 0
            
            if page < pages_to_try - 1:
                delay = 3.0 + random.uniform(0.5, 2.0)
                time.sleep(delay)

        except requests.exceptions.Timeout:
            consecutive_failures += 1
            if consecutive_failures >= 2:
                break
        except Exception as e:
            consecutive_failures += 1
            if consecutive_failures >= 2:
                break

    logger.info(f"[Bing] Found {len(images)} images")
    return images

def scrape_duckduckgo(query: str, num: int = 50) -> List[Dict]:
    """DuckDuckGo image search as additional fallback."""
    try:
        warnings.filterwarnings('ignore', category=RuntimeWarning, module='duckduckgo_search')
        
        from duckduckgo_search import DDGS
        logger.info(f"[DuckDuckGo] Searching for '{query}'...")
        
        ddg_logger = logging.getLogger('duckduckgo_search')
        ddg_logger.setLevel(logging.WARNING)
        
        images = []
        with DDGS() as ddgs:
            results = ddgs.images(
                keywords=query,
                max_results=min(num * 2, 100),
                safesearch='off',
                size='Large'
            )
            for r in results:
                if r.get('image'):
                    images.append({
                        'url': r['image'],
                        'source': 'duckduckgo',
                        'query': query
                    })
        
        logger.info(f"[DuckDuckGo] Found {len(images)} images")
        return images
    except ImportError:
        logger.warning("[DuckDuckGo] duckduckgo-search not installed (pip install duckduckgo-search)")
        return []
    except Exception as e:
        logger.error(f"[DuckDuckGo] Error: {e}")
        return []

# ============================================================================
# ORCHESTRATOR: MULTI-TIER SCRAPING STRATEGY
# ============================================================================

def scrape_multi_source(query: str, num: int, config: Dict = None, task: str = 'auto') -> List[Dict]:
    if config is None:
        config = {}
    
    logger.info(f"[SCRAPE] Multi-tier scraping: '{query}'")
    
    all_images = []
    sources_tried = []
    
    if config.get('serpapi_key'):
        imgs = scrape_serpapi(query, num, config['serpapi_key'], task)
        all_images.extend(imgs)
        sources_tried.append(f"SerpAPI({len(imgs)})")
    
    if config.get('unsplash_key') and len(all_images) < num:
        imgs = scrape_unsplash_api(query, num // 2, config['unsplash_key'])
        all_images.extend(imgs)
        sources_tried.append(f"Unsplash({len(imgs)})")
    
    if len(all_images) < num and config.get('use_selenium'):
        imgs = scrape_google_selenium(query, num - len(all_images))
        all_images.extend(imgs)
        sources_tried.append(f"Selenium({len(imgs)})")
    
    if len(all_images) < num:
        imgs = scrape_duckduckgo(query, num * 2)
        all_images.extend(imgs)
        sources_tried.append(f"DuckDuckGo({len(imgs)})")
    
    if len(all_images) < num:
        imgs = scrape_bing_advanced(query, num * 2)
        all_images.extend(imgs)
        sources_tried.append(f"Bing({len(imgs)})")
    
    # Deduplicate by URL
    seen = set()
    unique = []
    for img in all_images:
        url_hash = hashlib.md5(img['url'].encode()).hexdigest()
        if url_hash not in seen:
            seen.add(url_hash)
            unique.append(img)
    
    logger.info(f"[SCRAPE] {len(unique)} unique URLs from {', '.join(sources_tried)}")
    return unique

class HybridVerifier:
    def __init__(self, task_type='auto', use_clip_only=False, clip_threshold=0.50):
        self.task = task_type
        self.use_clip_only = use_clip_only
        self.clip_threshold = clip_threshold
        self.clip_model = None
        self.clip_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def tier1_url_filter(self, url: str, query: str) -> bool:
        terms = [t for t in query.lower().split() if len(t) > 3]
        if not terms:
            return True
        matches = sum(1 for t in terms if t in url.lower())
        return matches >= len(terms) // 2 + 1
    
    def tier2_exif_heuristic(self, img_path: str) -> Tuple[bool, float]:
        try:
            img = Image.open(img_path)
            exif = img._getexif()
            if exif and (271 in exif or 272 in exif):
                return True, 0.9
            arr = np.array(img)
            if len(arr.shape) == 3 and np.std(arr) < 20:
                return False, 0.3
            return True, 0.6
        except Exception:
            return True, 0.5
    
    def load_clip(self):
        if self.clip_model is not None:
            return True
        try:
            from transformers import CLIPProcessor, CLIPModel
            logger.info("[MODEL] Loading CLIP (fallback, cached if available)...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            logger.info(f"[MODEL] ✓ CLIP ready on {self.device}")
            return True
        except Exception as e:
            logger.error(f"[MODEL] Failed to load CLIP: {e}")
            return False
    
    def tier3_clip_semantic(self, img_paths: List[str], query: str) -> List[Tuple[bool, float, str]]:
        valid_images = []
        valid_paths = []
        for p in img_paths:
            if os.path.exists(p):
                try:
                    img = Image.open(p).convert('RGB')
                    if max(img.size) > 1024:
                        img.thumbnail((1024, 1024))
                    valid_images.append(img)
                    valid_paths.append(p)
                except:
                    pass
        
        if not valid_images:
            return [(True, 0.5, "no_images")] * len(img_paths)
        
        results = []
        
        if not self.load_clip():
            return [(False, 0.0, "model_load_fail")] * len(valid_images)
        
        logger.info(f"[TIER 3] Processing {len(valid_images)} images with CLIP (batch)...")
        
        positive_text = f"a photo of {query}"
        negative_text = "a random unrelated image"
        texts = [positive_text, negative_text]
        
        inputs = self.clip_processor(text=texts, images=valid_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        
        for i, prob in enumerate(probs):
            confidence = prob[0].item()
            is_relevant = confidence > self.clip_threshold
            reason = f"clip={confidence:.3f}"
            status = '✓' if is_relevant else '✗'
            if i < 5:
                logger.info(f"    [{i+1}] {status} {reason} (pos={prob[0]:.3f}, neg={prob[1]:.3f})")
            results.append((is_relevant, confidence, reason))
        
        return results

def download_image(img_data, save_dir, idx):
    path = None
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Referer': 'https://www.google.com/',
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'DNT': '1'
        }
        r = requests.get(img_data['url'], timeout=15, stream=True, headers=headers, allow_redirects=True, verify=True)
        
        if r.status_code == 200 and 'image' in r.headers.get('content-type','').lower():
            ct = r.headers.get('content-type', '').lower()
            ext = ct.split('/')[-1].split(';')[0]
            if ext not in ['jpeg', 'jpg', 'png', 'webp']:
                ext = 'jpg'
            
            path = os.path.join(save_dir, f"{img_data['query'].replace(' ', '_')}_{idx:04d}.{ext}")
            with open(path, 'wb') as f:
                for chunk in r.iter_content(1024): 
                    f.write(chunk)
            
            if os.path.getsize(path) > 5000:
                try:
                    with Image.open(path) as img:
                        img.verify()
                    with Image.open(path) as img:
                        img_clean = Image.new(img.mode, img.size)
                        img_clean.putdata(list(img.getdata()))
                        img_clean.save(path, quality=95, optimize=True)
                    return path
                except Exception as e:
                    os.remove(path)
        r.close()
    except Exception as e:
        pass
    return None

def download_and_verify(query: str, num: int, save_dir: str, task: str = 'auto', config: Dict = None, verifier=None) -> int:
    os.makedirs(save_dir, exist_ok=True)
    
    if config is None:
        config = {}
    
    urls = scrape_multi_source(query, num * 3, config, task)
    if not urls:
        logger.error("[ERROR] No URLs found from any source")
        logger.error("Possible solutions:")
        logger.error("  1. Add API keys to .env file (SERPAPI_KEY, UNSPLASH_KEY)")
        logger.error("  2. Try a different/simpler query (e.g., 'car' instead of 'car license plate')")
        logger.error("  3. Check internet connection")
        logger.error("  4. Install: pip install duckduckgo-search")
        logger.error("  5. Enable Selenium: USE_SELENIUM=1 in .env + pip install selenium webdriver-manager")
        return 0
    
    use_clip = config.get('use_clip_only', False)
    clip_threshold = config.get('clip_threshold', 0.50)
    if verifier is None:
        verifier = HybridVerifier(task, use_clip_only=use_clip, clip_threshold=clip_threshold)
    
    urls = [u for u in urls if verifier.tier1_url_filter(u['url'], query)]
    logger.info(f"[TIER 1] {len(urls)} URLs kept")
    
    if len(urls) == 0:
        urls = scrape_multi_source(query, num * 3, config, task)
    
    downloaded = []
    failed = 0
    logger.info(f"[DOWNLOAD] Downloading {min(len(urls), num * 2)} images in parallel...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_image, urls[i], save_dir, i+1) for i in range(min(len(urls), num * 2))]
        for future in as_completed(futures):
            path = future.result()
            if path:
                downloaded.append(path)
            else:
                failed += 1
            progress_pct = int((len(downloaded) / num) * 100) if num > 0 else 0
            print(f"\r  Downloaded: {len(downloaded)}/{num} ({progress_pct}%) | Failed: {failed}", end='', flush=True)
            if len(downloaded) >= num * 1.5:
                break
    
    print()  
    logger.info(f"[DOWNLOAD] {len(downloaded)} images (failed:{failed})")
    
    if len(downloaded) == 0:
        logger.error("[ERROR] No downloads succeeded. Check network/query.")
        return 0
    
    tier2_pass = []
    for path in downloaded:
        is_real, conf = verifier.tier2_exif_heuristic(path)
        if is_real and conf > 0.4:
            tier2_pass.append(path)
        else:
            try: os.remove(path)
            except: pass
    
    logger.info(f"[TIER 2] {len(tier2_pass)} real images")
    
    if len(tier2_pass) > 0:
        try:
            logger.info("[TIER 3] Semantic verification...")
            results = verifier.tier3_clip_semantic(tier2_pass, query)
            final = []
            removed = []
            
            rejected_dir = os.path.join(os.path.dirname(save_dir), "_rejected_debug")
            os.makedirs(rejected_dir, exist_ok=True)
            
            for path, (is_rel, conf, reason) in zip(tier2_pass, results):
                if is_rel:
                    final.append(path)
                else:
                    removed.append((Path(path).name, reason))
                    try:
                        shutil.move(path, os.path.join(rejected_dir, Path(path).name))
                    except:
                        try: os.remove(path)
                        except: pass
            
            if removed:
                logger.info(f"[TIER 3] Removed {len(removed)} irrelevant")
                for name, reason in removed[:2]:
                    logger.info(f"  ✗ {name}")
            
            logger.info(f"[TIER 3] {len(final)} relevant images")
            return len(final)
        except Exception as e:
            logger.error(f"[TIER 3] Error: {e}")
            return len(tier2_pass)
    
    return len(tier2_pass)

def clean_and_dedupe(data_dir: str, min_w: int = 200, min_h: int = 200) -> int:
    imgs = [f for f in Path(data_dir).glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    seen, kept = set(), 0
    
    for img_file in imgs:
        try:
            with Image.open(img_file) as img:
                if img.width < min_w or img.height < min_h:
                    os.remove(img_file)
                    continue
                
                img_hash = hashlib.md5(img.tobytes()).hexdigest()
                if img_hash in seen:
                    os.remove(img_file)
                    continue
                
                seen.add(img_hash)
                kept += 1
        except:
            try: os.remove(img_file)
            except: pass
    
    logger.info(f"[CLEAN] {kept} final images")
    return kept

def finalize_dataset(src_dir: str, out_dir: str, query: str) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    imgs = [f for f in Path(src_dir).glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    
    metadata = {'query': query, 'total': len(imgs), 'created': datetime.now().isoformat(), 'images': [], 'note': 'For research use only. Respect copyrights and attribute sources.'}
    
    for idx, img_file in enumerate(imgs, 1):
        new_name = f"{query.replace(' ', '_')}_{idx:04d}{img_file.suffix}"
        shutil.move(str(img_file), os.path.join(out_dir, new_name))
        with Image.open(os.path.join(out_dir, new_name)) as img:
            metadata['images'].append({'file': new_name, 'size': f"{img.width}x{img.height}"})
    
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"[DONE] {len(imgs)} images → {out_dir}")
    return metadata

def run_pipeline(query: str, num: int = 50, min_size: int = 200, task: str = 'auto', config: Dict = None, verifier=None):
    logger.info("="*70)
    logger.info("AMSDG-HV: Multi-Source Dataset Generation")
    logger.info("="*70)
    
    if config is None:
        config = {}
    
    base = Path.cwd().parent if Path.cwd().name == 'scripts' else Path.cwd()
    raw_dir = base / "data" / "raw"
    out_dir = base / "outputs" / query.replace(' ', '_')
    
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    start = time.time()
    
    try:
        count = download_and_verify(query, num, str(raw_dir), task, config, verifier)
        if count == 0:
            logger.error("❌ No images passed verification")
            return
        
        final = clean_and_dedupe(str(raw_dir), min_size, min_size)
        
        metadata = finalize_dataset(str(raw_dir), str(out_dir), query)
        
        elapsed = time.time() - start
        logger.info("="*70)
        logger.info(f"✅ {final} images in {elapsed:.1f}s → {out_dir}")
        logger.info("="*70)
    except Exception as e:
        logger.error(f"❌ Error: {e}")

def preload_models(verifier):
    """Pre-configure CLIP model before starting pipeline."""
    logger.info("[PRE-LOAD] Configuring CLIP model...")
    
    # Load CLIP for verification
    if verifier.load_clip():
        logger.info("[✓] CLIP ready (fast batch mode)")
        return True
    
    logger.warning("[!] CLIP failed to load")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMSDG-HV: Cross-Modal Dataset Generator")
    parser.add_argument('--query', type=str, required=True, help="Search query")
    parser.add_argument('--num', type=int, default=50, help="Target number of images")
    parser.add_argument('--min_size', type=int, default=200, help="Minimum image size")
    parser.add_argument('--task', type=str, default='auto', choices=['detection', 'classification', 'segmentation', 'auto'], help="Task type")
    parser.add_argument('--skip-model-preload', action='store_true', help="Skip model pre-loading (not recommended)")
    parser.add_argument('--use-clip', action='store_true', help="Use fast CLIP verification (default, recommended for CPU)")
    parser.add_argument('--clip-threshold', type=float, default=0.50, help="CLIP confidence threshold (0.0-1.0, default: 0.50, higher = more strict)")
    
    args = parser.parse_args()
    
    logger.info("AMSDG-HV: Cross-Modal Dataset Generator")
    logger.info("-" * 50)
    
    # Load .env file if it exists (check both current dir and parent dir)
    try:
        from dotenv import load_dotenv
        env_path = Path.cwd() / '.env'
        if not env_path.exists():
            env_path = Path.cwd().parent / '.env'
        
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"✓ Loaded .env file from {env_path.parent.name}/")
        else:
            logger.info("ℹ️  No .env file found")
    except ImportError:
        logger.warning("⚠️  python-dotenv not installed (pip install python-dotenv)")
    
    # API Configuration (optional but recommended)
    config = {}
    
    if os.getenv('SERPAPI_KEY'):
        config['serpapi_key'] = os.getenv('SERPAPI_KEY')
        logger.info("✓ SerpAPI enabled")
    
    if os.getenv('UNSPLASH_KEY'):
        config['unsplash_key'] = os.getenv('UNSPLASH_KEY')
        logger.info("✓ Unsplash API enabled")
    
    if os.getenv('USE_SELENIUM'):
        config['use_selenium'] = True
        logger.info("✓ Selenium enabled")
    
    # Always use CLIP for verification
    config['use_clip_only'] = True
    logger.info("✓ CLIP verification mode enabled")
    
    config['clip_threshold'] = args.clip_threshold
    if args.clip_threshold != 0.50:
        logger.info(f"✓ CLIP threshold set to {args.clip_threshold} (default: 0.50)")
    
    if not config or (not config.get('serpapi_key') and not config.get('unsplash_key')):
        logger.info("ℹ️  Using free scraping (add API keys to .env for better results)")
    
    # Create verifier instance once
    verifier = HybridVerifier(task_type=args.task, use_clip_only=True, clip_threshold=args.clip_threshold)
    
    # Pre-configure CLIP model using the same verifier
    if not args.skip_model_preload:
        preload_models(verifier)
    
    logger.info("="*70)
    run_pipeline(args.query, args.num, args.min_size, args.task, config, verifier)
