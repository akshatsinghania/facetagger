#!/usr/bin/env python3
"""
firebase_distribute.py

1. Reads photos.json
2. Distributes images equally across 10 Firebase projects
3. Adds 'project' and 'newSrc' fields to each photo in photos.json
4. Creates 10 project folders with firebase.json, .firebaserc, and public/
5. Copies images into respective project public/ folders
6. Deploys each project one by one via Firebase CLI

Usage:
    python firebase_distribute.py

Edit CONFIG section before running.
"""

import os
import json
import shutil
import subprocess
from pathlib import Path
from math import ceil

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

# Root folder where your photos actually live on disk
PHOTOS_ROOT = "D:\parojan and 50th anniversary\Main Cam"

# photos.json path (SourceFile paths are relative to PHOTOS_ROOT)
PHOTOS_JSON = "D:\parojan and 50th anniversar\public\photos.json"

# Where to create the 10 project deploy folders
DEPLOY_DIR = "D:\parojan and 50th anniversary\deploy"

# Output updated photos.json
OUTPUT_PHOTOS_JSON = "D:\parojan and 50th anniversar\public\photos_updated.json"

# 10 Firebase projects: (project_id, hosted_url)
FIREBASE_PROJECTS = [
    ("croslandcrafts",  "https://crosland-crafts.web.app"),
    ("sightsense-official",  "https://sightsense-official.web.app"),
    ("tedx-jpis",  "https://tedx-jpis.web.app"),
    ("art-showcase-akshat",  "https://art-showcase-akshat.web.app"),
    ("cool-animation-by-akshat",  "https://cool-animation-by-akshat.web.app"),
    ("papyri-decipher",  "https://papyri-decipher.web.app"),
    ("amit-portfolio-ee7cf",  "https://amit-portfolio-ee7cf.web.app"),
    ("croandcraftsofficial",  "https://croandcraftsofficial.web.app"),
    ("meditations-apps",  "https://meditations-apps.web.app"),
    ("home-page--assignment", "https://home-page--assignment.web.app"),
]

# firebase.json content (same for all projects)
FIREBASE_JSON_CONTENT = {
    "hosting": {
        "public": "public",
        "ignore": [
            "firebase.json",
            "**/.*",
            "**/node_modules/**"
        ],
        "headers": [
            {
                "source": "**/*.@(jpg|jpeg|png|webp|heic|tiff|JPG|JPEG|PNG)",
                "headers": [
                    {"key": "Cache-Control", "value": "public, max-age=31536000, immutable"},
                    {"key": "Access-Control-Allow-Origin", "value": "*"}
                ]
            }
        ]
    }
}

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def load_photos(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_photos(photos, path):
    with open(path, 'w') as f:
        json.dump(photos, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved updated photos.json → {path}")

def distribute(items, n):
    """Split list into n roughly equal chunks."""
    chunk_size = ceil(len(items) / n)
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def create_project_folder(project_id, hosted_url, deploy_dir):
    """Create project folder with firebase.json, .firebaserc, and public/."""
    project_dir = Path(deploy_dir) / project_id
    public_dir  = project_dir / "public"
    public_dir.mkdir(parents=True, exist_ok=True)

    # Write firebase.json
    firebase_json_path = project_dir / "firebase.json"
    with open(firebase_json_path, 'w') as f:
        json.dump(FIREBASE_JSON_CONTENT, f, indent=2)

    # Write .firebaserc
    firebaserc_path = project_dir / ".firebaserc"
    firebaserc = {"projects": {"default": project_id}}
    with open(firebaserc_path, 'w') as f:
        json.dump(firebaserc, f, indent=2)

    print(f"  ✓ Created {project_dir}")
    return public_dir

def safe_dest_filename(src_file, dest_dir):
    """
    Use just the filename. If duplicate exists, prefix with parent folder name.
    """
    filename = Path(src_file).name
    dest = Path(dest_dir) / filename
    if dest.exists():
        parent = Path(src_file).parent.name
        filename = f"{parent}_{filename}"
        dest = Path(dest_dir) / filename
    return dest, filename

def deploy_project(project_dir):
    """Run firebase deploy for a given project directory."""
    print(f"\n  Deploying {project_dir}...")
    result = subprocess.run(
        ["firebase", "deploy", "--only", "hosting"],
        cwd=project_dir,
        capture_output=False
    )
    if result.returncode == 0:
        print(f"  ✓ Deployed {project_dir}")
    else:
        print(f"  ✗ Deploy failed for {project_dir} (exit code {result.returncode})")
    return result.returncode == 0

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 50)
    print("Firebase Image Distributor")
    print("=" * 50)

    # 1. Load photos.json
    print(f"\nLoading {PHOTOS_JSON}...")
    photos = load_photos(PHOTOS_JSON)
    print(f"  {len(photos)} photos found")

    # 2. Distribute photos across projects
    n = len(FIREBASE_PROJECTS)
    batches = distribute(photos, n)
    print(f"\nDistribution plan ({len(photos)} photos across {n} projects):")
    for i, (project_id, _) in enumerate(FIREBASE_PROJECTS):
        count = len(batches[i]) if i < len(batches) else 0
        print(f"  {project_id}: {count} photos")

    # 3. Create project folders
    print(f"\nCreating project folders in {DEPLOY_DIR}...")
    public_dirs = []
    for project_id, hosted_url in FIREBASE_PROJECTS:
        public_dir = create_project_folder(project_id, hosted_url, DEPLOY_DIR)
        public_dirs.append(public_dir)

    # 4. Copy images + update photos.json fields
    print("\nCopying images and updating photos.json...")
    for i, (project_id, hosted_url) in enumerate(FIREBASE_PROJECTS):
        if i >= len(batches):
            break
        batch = batches[i]
        public_dir = public_dirs[i]
        copied = 0
        skipped = 0

        for photo in batch:
            src_relative = photo.get("SourceFile", "")
            src_abs = Path(PHOTOS_ROOT) / src_relative

            if not src_abs.exists():
                print(f"    WARNING: File not found: {src_abs}")
                skipped += 1
                continue

            # Determine destination filename (handle duplicates)
            dest_path, dest_filename = safe_dest_filename(src_relative, public_dir)

            # Copy the file
            shutil.copy2(src_abs, dest_path)
            copied += 1

            # Build new Firebase URL
            new_src = f"{hosted_url}/{dest_filename}"

            # Add fields to photo entry (keep all existing fields)
            photo["project"]  = project_id
            photo["newSrc"]   = new_src

        print(f"  {project_id}: {copied} copied, {skipped} skipped")

    # 5. Save updated photos.json
    print()
    save_photos(photos, OUTPUT_PHOTOS_JSON)

    # 6. Deploy all projects
    print("\n" + "=" * 50)
    print("Deploying all projects...")
    print("=" * 50)
    success = 0
    failed = []
    for project_id, _ in FIREBASE_PROJECTS:
        project_dir = Path(DEPLOY_DIR) / project_id
        ok = deploy_project(str(project_dir))
        if ok:
            success += 1
        else:
            failed.append(project_id)

    # 7. Summary
    print("\n" + "=" * 50)
    print("DONE")
    print(f"  ✓ {success} projects deployed successfully")
    if failed:
        print(f"  ✗ Failed: {', '.join(failed)}")
        print("    Re-run deploy manually:")
        for project_id in failed:
            print(f"    cd {DEPLOY_DIR}/{project_id} && firebase deploy --only hosting")
    print("=" * 50)

    # Remind about index.html
    print("\nNOTE: Update index.html to use 'newSrc' field instead of 'SourceFile'")
    print("      for images, while keeping 'SourceFile' for reference.")


if __name__ == "__main__":
    main()