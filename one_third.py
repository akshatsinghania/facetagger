"""
Firebase Image Load Distributor
================================
Splits images from one Firebase hosting project across multiple new projects
to reduce bandwidth load. Original project keeps ALL images (bandwidth relief
comes from 66% of URLs now pointing to new projects).

Usage:
    python distribute_images.py

Edit the CONFIG section below before running.
"""

import json
import os
import shutil
import math

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these values before running
# ─────────────────────────────────────────────────────────────────────────────

# Full path to the original Firebase project folder (contains public/ subfolder)
ORIGINAL_PROJECT_PATH = r"D:\parojan and 50th anniversary\deploy\hi124535"

# Full path to the input photos JSON
INPUT_JSON_PATH = r"D:\parojan and 50th anniversar\public\photos.json"

# Full path to write the updated photos JSON
OUTPUT_JSON_PATH = r"D:\parojan and 50th anniversar\public\photos_updated_divided.json"

# Full path to the firebase.json file to copy into new projects
# (usually lives at ORIGINAL_PROJECT_PATH/firebase.json)
FIREBASE_JSON_SOURCE = os.path.join(ORIGINAL_PROJECT_PATH, "firebase.json")

# Parent folder where new project folders will be created
# (sibling of project1 by default)
DEPLOY_PARENT = os.path.dirname(ORIGINAL_PROJECT_PATH)

# Base URL of the project you want to split (no trailing slash)
# Only photos whose newSrc starts with this will be redistributed.
# Everything else in the JSON passes through untouched.
ORIGINAL_BASE_URL = "https://hi124535.web.app"  # ← change me

# New projects: list of dicts with 'project_id' and 'base_url'
#   project_id  → used as folder name AND in .firebaserc
#   base_url    → the Firebase Hosting base URL (no trailing slash)
NEW_PROJECTS = [
    {
        "project_id": "arcade-leaderboard",          # ← change me
        "base_url":   "https://arcade-leaderboard.web.app",  # ← change me
    },
    {
        "project_id": "conwayss-game-of-life",          # ← change me
        "base_url":   "https://conwayss-game-of-life.web.app",  # ← change me
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# END CONFIG
# ─────────────────────────────────────────────────────────────────────────────


def firebaserc_content(project_id: str) -> dict:
    return {
        "projects": {
            "default": project_id
        }
    }


def get_filename_from_url(url: str) -> str:
    """Extract just the filename from a URL."""
    return url.split("/")[-1]


def build_new_url(base_url: str, filename: str) -> str:
    return f"{base_url}/{filename}"



def setup_new_project_folder(project_id: str, deploy_parent: str) -> str:
    """Create folder structure for a new Firebase project."""
    project_path = os.path.join(deploy_parent, project_id)
    public_path  = os.path.join(project_path, "public")
    os.makedirs(public_path, exist_ok=True)

    # .firebaserc
    firebaserc_path = os.path.join(project_path, ".firebaserc")
    with open(firebaserc_path, "w", encoding="utf-8") as f:
        json.dump(firebaserc_content(project_id), f, indent=2)

    # firebase.json (copy from source)
    dest_firebase_json = os.path.join(project_path, "firebase.json")
    if os.path.exists(FIREBASE_JSON_SOURCE):
        shutil.copy2(FIREBASE_JSON_SOURCE, dest_firebase_json)
        print(f"  ✓ Copied firebase.json → {dest_firebase_json}")
    else:
        print(f"  ⚠ firebase.json not found at {FIREBASE_JSON_SOURCE}, skipping copy.")

    print(f"  ✓ Created project folder: {project_path}")
    return public_path


def main():
    print("=" * 60)
    print("Firebase Image Load Distributor")
    print("=" * 60)

    # ── Load JSON ──────────────────────────────────────────────
    print(f"\n[1/4] Loading photos JSON from:\n      {INPUT_JSON_PATH}")
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        photos = json.load(f)
    print(f"      Found {len(photos)} photo entries.")

    # ── Filter only photos belonging to the project we're splitting ───
    # Only entries whose newSrc starts with ORIGINAL_BASE_URL are redistributed.
    # All other projects' entries pass through the JSON completely untouched.
    photos_for_project = [
        p for p in photos
        if p.get("newSrc", "").startswith(ORIGINAL_BASE_URL)
    ]
    total = len(photos_for_project)
    print(f"      {total} entries belong to {ORIGINAL_BASE_URL} (will be split).")
    print(f"      {len(photos) - total} entries from other projects (untouched).")

    original_public = os.path.join(ORIGINAL_PROJECT_PATH, "public")

    # ── Split into thirds ──────────────────────────────────────
    # First third  → stays in original (no file move, no URL change)
    # Second third → goes to NEW_PROJECTS[0]
    # Third third  → goes to NEW_PROJECTS[1]
    chunk = math.ceil(total / 3)
    keep_range   = photos_for_project[:chunk]
    bucket_ranges = [
        photos_for_project[chunk : chunk * 2],
        photos_for_project[chunk * 2 :],
    ]

    print(f"\n      Split: {len(keep_range)} stay | "
          f"{len(bucket_ranges[0])} → project 2 | "
          f"{len(bucket_ranges[1])} → project 3")

    # ── Set up new project folders ─────────────────────────────
    print(f"\n[2/4] Setting up new project folders in:\n      {DEPLOY_PARENT}")
    new_public_paths = []
    for proj in NEW_PROJECTS:
        print(f"\n  → {proj['project_id']}")
        pub = setup_new_project_folder(proj["project_id"], DEPLOY_PARENT)
        new_public_paths.append(pub)

    # ── Copy images & build URL mapping ───────────────────────
    print(f"\n[3/4] Copying images to new project folders...")

    # Map: original idx in photos list → new URL
    url_updates = {}  # photo object id → new URL

    for proj_i, (bucket, proj_info, new_pub) in enumerate(
        zip(bucket_ranges, NEW_PROJECTS, new_public_paths)
    ):
        print(f"\n  → Copying {len(bucket)} images to {proj_info['project_id']} ...")
        copied = 0
        skipped = 0
        for photo in bucket:
            src_url  = photo["newSrc"]
            filename = get_filename_from_url(src_url)
            src_file = os.path.join(original_public, filename)
            dst_file = os.path.join(new_pub, filename)

            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
                copied += 1
            else:
                skipped += 1
                if skipped <= 5:
                    print(f"    ⚠ File not found (URL will still update): {src_file}")

            # Record the new URL regardless (file may be deployed separately)
            new_url = build_new_url(proj_info["base_url"], filename)
            url_updates[id(photo)] = new_url

        print(f"    ✓ {copied} copied, {skipped} not found locally.")

    # ── Update JSON ────────────────────────────────────────────
    print(f"\n[4/4] Writing updated JSON to:\n      {OUTPUT_JSON_PATH}")

    updated_photos = []
    changed = 0
    for photo in photos:
        new_photo = dict(photo)
        if id(photo) in url_updates:
            new_photo["newSrc"] = url_updates[id(photo)]
            changed += 1
        updated_photos.append(new_photo)

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(updated_photos, f, ensure_ascii=False, indent=2)

    print(f"  ✓ {changed} URLs updated, {len(photos) - changed} unchanged.")
    print(f"  ✓ Total entries written: {len(updated_photos)}")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE! Summary:")
    print(f"  Original project : {len(keep_range)} images (no change)")
    for i, (proj, bucket) in enumerate(zip(NEW_PROJECTS, bucket_ranges)):
        print(f"  {proj['project_id']} : {len(bucket)} images copied + URLs updated")
    print(f"\n  Updated JSON → {OUTPUT_JSON_PATH}")
    print("\nNext steps:")
    print("  1. Edit NEW_PROJECTS in this script with your real project IDs & URLs")
    print("  2. Deploy each new project folder via: firebase deploy --project <id>")
    print("  3. Replace your live photos.json with photos_updated.json")
    print("=" * 60)


if __name__ == "__main__":
    main()