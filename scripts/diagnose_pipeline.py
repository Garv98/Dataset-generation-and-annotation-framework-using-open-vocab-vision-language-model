"""
Pipeline Diagnostic Tool
Checks each stage to identify where images are getting lost or wrong content is being scraped.
"""

import requests
from PIL import Image
from io import BytesIO
import re
from urllib.parse import quote

def test_scraping(query: str = "car license plate front view", num_test: int = 5):
    """Test if scraping returns valid URLs."""
    print("\n" + "="*70)
    print("🔍 STAGE 1: SCRAPING TEST")
    print("="*70)
    print(f"Query: '{query}'")
    print(f"Testing first {num_test} results from Bing...\n")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        url = f"https://www.bing.com/images/search?q={quote(query)}&first=0&qft=+filterui:imagesize-large"
        response = requests.get(url, headers=headers, timeout=15)
        
        print(f"Response status: {response.status_code}")
        print(f"Response size: {len(response.text)} bytes")
        
        # Check for blocking
        if 'captcha' in response.text.lower():
            print("❌ BLOCKED: Bing detected automation (CAPTCHA)")
            return [], False
        
        if len(response.text) < 5000:
            print("❌ BLOCKED: Response too short (likely blocked)")
            return [], False
        
        # Extract URLs
        urls = re.findall(r'murl&quot;:&quot;(.*?)&quot;', response.text)
        
        print(f"✅ Found {len(urls)} image URLs")
        
        if len(urls) == 0:
            print("❌ PROBLEM: No URLs extracted")
            print("   Possible causes:")
            print("   - Bing changed HTML structure")
            print("   - Query blocked by search engine")
            print("   - Network/firewall blocking")
            return [], False
        
        return urls[:num_test], True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return [], False

def test_downloads(urls: list):
    """Test if URLs can be downloaded as valid images."""
    print("\n" + "="*70)
    print("📥 STAGE 2: DOWNLOAD TEST")
    print("="*70)
    print(f"Testing {len(urls)} URLs...\n")
    
    valid_count = 0
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://www.google.com/'
    }
    
    results = []
    
    for i, url in enumerate(urls, 1):
        print(f"[{i}] Testing: {url[:70]}...")
        
        try:
            response = requests.get(url, timeout=10, headers=headers)
            
            if response.status_code != 200:
                print(f"    ❌ Failed: HTTP {response.status_code}")
                results.append(('failed', f"HTTP {response.status_code}"))
                continue
            
            # Check content type
            ct = response.headers.get('content-type', '').lower()
            if 'image' not in ct and 'octet-stream' not in ct:
                print(f"    ❌ Not an image: {ct}")
                results.append(('failed', f"Wrong content-type: {ct}"))
                continue
            
            # Try to open as image
            try:
                img = Image.open(BytesIO(response.content))
                print(f"    ✅ Valid image: {img.size[0]}x{img.size[1]} {img.format}")
                valid_count += 1
                results.append(('success', f"{img.size[0]}x{img.size[1]} {img.format}"))
            except Exception as e:
                print(f"    ❌ Corrupted image: {str(e)[:40]}")
                results.append(('failed', "Corrupted"))
                
        except Exception as e:
            print(f"    ❌ Error: {str(e)[:50]}")
            results.append(('failed', str(e)[:50]))
    
    print(f"\n✅ Valid downloadable images: {valid_count}/{len(urls)}")
    
    return valid_count > 0, results

def test_content_relevance(query: str):
    """Manual check - user needs to verify if downloaded images match query."""
    print("\n" + "="*70)
    print("🔍 STAGE 3: CONTENT RELEVANCE (Manual Check Needed)")
    print("="*70)
    print(f"Query: '{query}'")
    print("\n⚠️  After running the full pipeline, check the outputs/ folder:")
    print(f"    outputs/{query.replace(' ', '_')}/")
    print("\nManually verify:")
    print("  1. Do images contain the objects mentioned in query?")
    print("  2. Are they relevant to your use case?")
    print("  3. If images are WRONG → Problem is SCRAPING (not CLIP)")
    print("  4. If images are CORRECT → CLIP is working properly")

def provide_recommendations(query: str, scraping_ok: bool, download_ok: bool):
    """Provide specific recommendations based on diagnostic results."""
    print("\n" + "="*70)
    print("💡 DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    # Check for license plate queries
    is_license_plate = any(term in query.lower() for term in ['license plate', 'number plate', 'registration'])
    
    if not scraping_ok:
        print("\n❌ PROBLEM: Scraping failed (no URLs found)")
        print("\n✅ SOLUTIONS:")
        print("   1. Check internet connection")
        print("   2. Try different query terms")
        print("   3. Wait 30 minutes (rate limit)")
        print("   4. Use VPN if blocked by region")
        print("   5. Consider Google Custom Search API (100 free queries/day)")
    
    elif not download_ok:
        print("\n❌ PROBLEM: URLs found but downloads failing")
        print("\n✅ SOLUTIONS:")
        print("   1. URLs may be expired/invalid")
        print("   2. Check firewall/antivirus blocking downloads")
        print("   3. Try different source (Flickr, Unsplash)")
        print("   4. Increase timeout in pipeline.py")
    
    else:
        print("\n✅ Scraping and downloads working!")
        print("\nNext: Run full pipeline and check content relevance:")
        print(f"   python pipeline.py")
        print(f"   Query: {query}")
        print(f"   Check: outputs/{query.replace(' ', '_')}/")
    
    # Special recommendations for license plates
    if is_license_plate:
        print("\n" + "="*70)
        print("🚗 SPECIAL RECOMMENDATION: LICENSE PLATE DETECTION")
        print("="*70)
        print("\nWeb scraping has LIMITED license plate images because:")
        print("  • Privacy: Search engines blur license plates")
        print("  • Copyright: Professional photos hide plates")
        print("  • Rarity: Real-world plate images not well-indexed")
        print("\n✅ RECOMMENDED SOLUTION: Use existing annotated datasets")
        print("\n1. Kaggle Car License Plate Detection (4,000+ images)")
        print("   URL: https://www.kaggle.com/datasets/andrewmvd/car-plate-detection")
        print("   Format: XML annotations (Pascal VOC)")
        print("\n2. Roboflow License Plates (10,000+ images)")
        print("   URL: https://public.roboflow.com/object-detection/license-plates")
        print("   Format: YOLO, COCO, Pascal VOC")
        print("\n3. OpenALPR Benchmarks (6,000+ images)")
        print("   URL: https://github.com/openalpr/benchmarks")
        print("   Format: Text annotations")
        print("\n4. CCPD Dataset (250,000+ Chinese plates)")
        print("   URL: https://github.com/detectRecog/CCPD")
        print("   Format: Image filenames encode labels")
        print("\nThese datasets are:")
        print("  ✓ Already annotated (no manual labeling needed)")
        print("  ✓ High quality (visible, clear plates)")
        print("  ✓ Diverse (different angles, lighting, countries)")
        print("  ✓ FREE for research/educational use")
        print("\nFor your research paper, you can cite using existing datasets")
        print("and focus your contribution on the MODEL architecture/training,")
        print("not dataset collection (which is a known hard problem).")

def main():
    """Run complete diagnostic."""
    print("\n" + "="*70)
    print("🔬 AMSDG-HV PIPELINE DIAGNOSTIC TOOL")
    print("="*70)
    print("\nThis tool checks each pipeline stage to identify problems.\n")
    
    # Get query from user
    query = input("Enter query to test (or press Enter for 'car license plate front view'): ").strip()
    if not query:
        query = "car license plate front view"
    
    print(f"\nTesting query: '{query}'")
    
    # Stage 1: Test scraping
    urls, scraping_ok = test_scraping(query, num_test=5)
    
    download_ok = False
    if scraping_ok and urls:
        # Stage 2: Test downloads
        download_ok, results = test_downloads(urls)
    
    # Stage 3: Manual content check
    if download_ok:
        test_content_relevance(query)
    
    # Provide recommendations
    provide_recommendations(query, scraping_ok, download_ok)
    
    print("\n" + "="*70)
    print("Diagnostic complete!")
    print("="*70)

if __name__ == "__main__":
    main()
