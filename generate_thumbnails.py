#!/usr/bin/env python3
"""
Generate thumbnails for all photos in the Photos directory.
Creates 400x400px WebP thumbnails in Photos_thumbs directory.
"""

import os
import json
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
SOURCE_DIR = Path("D:\parojan and 50th anniversary/Main Cam")
THUMB_DIR = Path("D:\parojan and 50th anniversary/Photos_thumbs")
THUMB_SIZE = (400, 400)
QUALITY = 80  # WebP quality (1-100, 80 provides good balance)
MAX_WORKERS = 8  # Parallel processing threads

def create_thumbnail(image_path, thumb_path):
    """Create a thumbnail for a single image."""
    try:
        # Skip if thumbnail already exists and is newer than source
        if thumb_path.exists():
            if thumb_path.stat().st_mtime > image_path.stat().st_mtime:
                return f"Skipped (exists): {image_path.name}"
        
        # Open and process image
        with Image.open(image_path) as img:
            # Convert RGBA to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create thumbnail maintaining aspect ratio
            img.thumbnail(THUMB_SIZE, Image.Resampling.LANCZOS)
            
            # Save as WebP
            img.save(thumb_path, 'WEBP', quality=QUALITY, method=6)
        
        return f"Created: {image_path.name}"
    
    except Exception as e:
        return f"Error processing {image_path.name}: {str(e)}"

def get_image_files(source_dir):
    """Get all image files from source directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp', '.tiff', '.tif', '.bmp'}
    image_files = []
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(Path(root) / file)
    
    return image_files

def main():
    print(f"🖼️  Thumbnail Generator")
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {THUMB_DIR}")
    print(f"Size: {THUMB_SIZE[0]}x{THUMB_SIZE[1]}px")
    print("-" * 50)
    
    # Create thumbnail directory
    THUMB_DIR.mkdir(exist_ok=True)
    
    # Get all image files
    image_files = get_image_files(SOURCE_DIR)
    total_files = len(image_files)
    
    if total_files == 0:
        print("❌ No images found in source directory")
        return
    
    print(f"Found {total_files} images")
    print(f"Processing with {MAX_WORKERS} threads...\n")
    
    # Process images in parallel
    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for img_path in image_files:
            # Maintain directory structure
            relative_path = img_path.relative_to(SOURCE_DIR)
            thumb_path = THUMB_DIR / relative_path.with_suffix('.webp')
            
            # Create subdirectories if needed
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Submit task
            future = executor.submit(create_thumbnail, img_path, thumb_path)
            tasks.append(future)
        
        # Show progress
        completed = 0
        for future in as_completed(tasks):
            completed += 1
            result = future.result()
            print(f"[{completed}/{total_files}] {result}")
    
    print("\n✅ Thumbnail generation complete!")
    print(f"Thumbnails saved to: {THUMB_DIR}/")
    
    # Update photos.json to include thumbnail paths
    print("\nUpdating photos.json with thumbnail paths...")
    try:
        with open('photos.json', 'r') as f:
            photos_data = json.load(f)
        
        for photo in photos_data:
            source_path = Path(photo['SourceFile'])
            relative_path = source_path.relative_to(SOURCE_DIR)
            thumb_path = THUMB_DIR / relative_path.with_suffix('.webp')
            photo['Thumbnail'] = str(thumb_path)
        
        with open('photos.json', 'w') as f:
            json.dump(photos_data, f, indent=2)
        
        print("✅ photos.json updated with thumbnail paths")
    
    except Exception as e:
        print(f"⚠️  Could not update photos.json: {str(e)}")
        print("You'll need to manually update your code to use thumbnail paths")

if __name__ == "__main__":
    main()
