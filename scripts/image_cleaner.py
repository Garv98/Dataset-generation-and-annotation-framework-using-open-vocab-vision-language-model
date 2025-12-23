# image_cleaner.py
"""
Script to clean, filter, and deduplicate scraped images.
Removes low-quality, corrupted, or duplicate images.
"""
import os
import json
import hashlib
from PIL import Image
from typing import List, Dict, Tuple
import imagehash

def get_image_hash(image_path: str) -> str:
    """
    Generate perceptual hash for duplicate detection.
    Args:
        image_path (str): Path to the image.
    Returns:
        str: Perceptual hash of the image.
    """
    try:
        img = Image.open(image_path)
        return str(imagehash.average_hash(img))
    except Exception as e:
        print(f"[WARNING] Could not hash {image_path}: {e}")
        return None

def check_image_quality(image_path: str, min_width: int = 200, min_height: int = 200) -> Tuple[bool, Dict]:
    """
    Check if image meets quality criteria.
    Args:
        image_path (str): Path to the image.
        min_width (int): Minimum acceptable width.
        min_height (int): Minimum acceptable height.
    Returns:
        Tuple of (is_valid, metadata).
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            
            # Check minimum dimensions
            if width < min_width or height < min_height:
                return False, {'reason': 'too_small', 'size': (width, height)}
            
            # Check if image is corrupted
            img.verify()
            
            return True, {
                'width': width,
                'height': height,
                'mode': mode,
                'format': img.format
            }
    except Exception as e:
        return False, {'reason': 'corrupted', 'error': str(e)}

def clean_images(data_dir: str = "../data/raw", 
                 min_width: int = 200, 
                 min_height: int = 200,
                 remove_duplicates: bool = True):
    """
    Clean images in the data directory.
    Args:
        data_dir (str): Directory containing images.
        min_width (int): Minimum image width.
        min_height (int): Minimum image height.
        remove_duplicates (bool): Whether to remove duplicate images.
    """
    if not os.path.exists(data_dir):
        print(f"[ERROR] Directory {data_dir} does not exist!")
        return
    
    print(f"[INFO] Cleaning images in {data_dir}...")
    
    image_files = [f for f in os.listdir(data_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    
    print(f"[INFO] Found {len(image_files)} images")
    
    # Track statistics
    removed_low_quality = 0
    removed_duplicates = 0
    kept_images = []
    seen_hashes = set()
    
    for img_file in image_files:
        img_path = os.path.join(data_dir, img_file)
        
        # Check quality
        is_valid, metadata = check_image_quality(img_path, min_width, min_height)
        
        if not is_valid:
            print(f"[INFO] Removing {img_file}: {metadata.get('reason', 'unknown')}")
            os.remove(img_path)
            removed_low_quality += 1
            continue
        
        # Check for duplicates
        if remove_duplicates:
            img_hash = get_image_hash(img_path)
            if img_hash and img_hash in seen_hashes:
                print(f"[INFO] Removing duplicate: {img_file}")
                os.remove(img_path)
                removed_duplicates += 1
                continue
            if img_hash:
                seen_hashes.add(img_hash)
        
        kept_images.append({
            'filename': img_file,
            'path': img_path,
            **metadata
        })
    
    print(f"\n[INFO] Cleaning complete!")
    print(f"  - Kept: {len(kept_images)} images")
    print(f"  - Removed (low quality): {removed_low_quality}")
    print(f"  - Removed (duplicates): {removed_duplicates}")
    
    # Save cleaned metadata
    metadata_path = os.path.join(data_dir, "cleaned_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(kept_images, f, indent=2)
    
    print(f"[INFO] Cleaned metadata saved to {metadata_path}")

if __name__ == "__main__":
    data_directory = input("Enter data directory path (default: ../data/raw): ").strip()
    if not data_directory:
        data_directory = "../data/raw"
    
    min_w = input("Minimum width (default: 200): ").strip()
    min_h = input("Minimum height (default: 200): ").strip()
    
    min_width = int(min_w) if min_w else 200
    min_height = int(min_h) if min_h else 200
    
    clean_images(data_directory, min_width, min_height)
