"""
Face Tagger with Clustering
============================
Solves the "same face treated as different people" problem by:
1. Extracting 512D embeddings from ALL faces across ALL photos
2. Clustering similar embeddings together using Chinese Whispers algorithm
3. Showing you one representative face per cluster to name
4. Writing names to XMP metadata of original photos

PHASES:
  Phase 1 -- Scan all photos, extract face embeddings, cluster them, generate CSV + cluster preview
  Phase 2 -- After you fill in the CSV, write names to photo metadata

INSTALL:
  pip install insightface onnxruntime opencv-python pillow numpy scipy scikit-learn
  # Chinese Whispers via dlib (optional but recommended) OR we use our own implementation
  pip install dlib  # optional, for chinese_whispers_clustering

  # exiftool (for metadata writing)
  # macOS:   brew install exiftool
  # Ubuntu:  sudo apt install libimage-exiftool-perl
  # Windows: https://exiftool.org/install.html

USAGE:
  python face_tagger_v2.py --phase 1 --photos_dir "D:\parojan and 50th anniversary\Main Cam"

# remove all name tags | this is only to be done incase the output is deleted and then done again`
exiftool -XMP:PersonInImage= -XMP:Subject= -IPTC:Keywords= -overwrite_original -r /Users/unnatisinghania/Documents/facetag/Photos


  # Fill in faces_output/clusters/cluster_names.csv
  python face_tagger_v2.py --phase 2 --photos_dir "D:\parojan and 50th anniversary\Main Cam"





# last phase
exiftool -json -XMP:PersonInImage -r "D:\parojan and 50th anniversary\Main Cam" > photos.json





TUNING
   python face_tagger_v2.py --recluster --similarity 0.65

  --similarity  Cosine similarity threshold (0.0-1.0). Default: 0.55
                Higher = stricter (more clusters, fewer false merges)
                Lower  = looser  (fewer clusters, may merge different people)
                Try 0.45-0.65 depending on your photo set.
"""

import os
import json
import csv
import argparse
import shutil
from pathlib import Path
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────
FACES_OUTPUT_DIR   = "faces_output"
EMBEDDINGS_FILE    = os.path.join(FACES_OUTPUT_DIR, "embeddings.json")
PROGRESS_FILE      = os.path.join(FACES_OUTPUT_DIR, "progress.json")
CLUSTERS_DIR       = os.path.join(FACES_OUTPUT_DIR, "clusters")
CSV_PATH           = os.path.join(FACES_OUTPUT_DIR, "cluster_names.csv")

SUPPORTED_EXTS     = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
SIMILARITY_DEFAULT = 0.4   # cosine similarity threshold

# ── Utilities ───────────────────────────────────────────────────────────────

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"processed": []}

def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)

def load_embeddings():
    """Returns list of dicts: {photo_path, face_index, bbox, embedding}"""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE) as f:
            data = json.load(f)
        # Convert embedding lists back to numpy arrays
        for d in data:
            d["embedding"] = np.array(d["embedding"], dtype=np.float32)
        return data
    return []

def save_embeddings(faces):
    """Saves face list to JSON (embeddings as plain lists)."""
    serializable = []
    for d in faces:
        serializable.append({
            "photo_path":  d["photo_path"],
            "face_index":  d["face_index"],
            "bbox":        d["bbox"],
            "embedding":   d["embedding"].tolist(),
            "crop_path":   d.get("crop_path", ""),
        })
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(serializable, f)

def get_all_photos(photos_dir):
    photos = []
    for root, _, files in os.walk(photos_dir):
        for fname in files:
            if Path(fname).suffix.lower() in SUPPORTED_EXTS:
                photos.append(os.path.join(root, fname))
    return sorted(photos)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# ── Chinese Whispers clustering (pure Python, no dlib dependency) ───────────

def chinese_whispers_clustering(embeddings, threshold=0.55, iterations=20):
    """
    Groups face embeddings into clusters.
    Returns: list of cluster labels (same length as embeddings).
             Faces with no match get their own unique cluster.
    
    threshold: minimum cosine similarity to be considered same person.
               0.55 is a good default; lower = looser grouping.
    """
    n = len(embeddings)
    if n == 0:
        return []

    print(f"  Computing pairwise similarities for {n} faces...")

    # Build adjacency list: edges where similarity >= threshold
    # We compute this in batches to avoid memory issues with large sets
    emb_matrix = np.stack(embeddings)  # (n, 512)
    # Normalize
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_norm = emb_matrix / (norms + 1e-10)

    # Compute full cosine similarity matrix
    # For large n this could be memory-intensive; chunk if n > 5000
    CHUNK = 500
    labels = list(range(n))   # Each face starts as its own cluster

    for iteration in range(iterations):
        changed = False
        order = np.random.permutation(n)

        for i in order:
            # Find all neighbors above threshold
            # Compute similarities between face i and all others in chunks
            neighbors = {}  # label -> count

            for start in range(0, n, CHUNK):
                end = min(start + CHUNK, n)
                sims = emb_norm[i] @ emb_norm[start:end].T   # (end-start,)
                for j_local, sim in enumerate(sims):
                    j = start + j_local
                    if j != i and sim >= threshold:
                        lbl = labels[j]
                        neighbors[lbl] = neighbors.get(lbl, 0) + sim  # weight by similarity

            if neighbors:
                best_label = max(neighbors, key=neighbors.get)
                if labels[i] != best_label:
                    labels[i] = best_label
                    changed = True

        if not changed:
            break

    # Remap labels to 0, 1, 2, ...
    unique_labels = {}
    counter = 0
    final = []
    for lbl in labels:
        if lbl not in unique_labels:
            unique_labels[lbl] = counter
            counter += 1
        final.append(unique_labels[lbl])

    return final

# ── Phase 1: Extract embeddings and cluster ─────────────────────────────────

def run_phase1(photos_dir, similarity_threshold):
    import insightface
    import cv2

    os.makedirs(CLUSTERS_DIR, exist_ok=True)

    # Load InsightFace
    print("Loading InsightFace buffalo_l model (downloads ~200MB on first run)...")
    app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    progress  = load_progress()
    done_set  = set(progress["processed"])
    all_faces = load_embeddings()  # resume existing
    done_photos = {d["photo_path"] for d in all_faces} | done_set

    all_photos = get_all_photos(photos_dir)
    total      = len(all_photos)
    new_photos = [p for p in all_photos if p not in done_photos]

    print(f"\nTotal photos:     {total}")
    print(f"Already scanned:  {total - len(new_photos)}")
    print(f"To scan now:      {len(new_photos)}\n")

    face_global_idx = len(all_faces)

    for i, photo_path in enumerate(new_photos):
        print(f"[{i+1}/{len(new_photos)}] {photo_path}")

        try:
            img = cv2.imread(photo_path)
            if img is None:
                print(f"  ⚠ Unreadable, skipping.")
                progress["processed"].append(photo_path)
                save_progress(progress)
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces   = app.get(img_rgb)

            if not faces:
                print(f"  No faces.")
            else:
                print(f"  {len(faces)} face(s).")
                for fi, face in enumerate(faces):
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    h, w = img.shape[:2]
                    pad  = 30
                    x1c, y1c = max(0, x1-pad), max(0, y1-pad)
                    x2c, y2c = min(w, x2+pad), min(h, y2+pad)

                    crop      = img[y1c:y2c, x1c:x2c]
                    crop_name = f"face_{face_global_idx:05d}.jpg"
                    crop_path = os.path.join(FACES_OUTPUT_DIR, "all_crops", crop_name)
                    os.makedirs(os.path.dirname(crop_path), exist_ok=True)
                    cv2.imwrite(crop_path, crop)

                    all_faces.append({
                        "photo_path": photo_path,
                        "face_index": fi,
                        "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                        "embedding":  face.normed_embedding.astype(np.float32),
                        "crop_path":  crop_path,
                    })
                    face_global_idx += 1

            progress["processed"].append(photo_path)
            save_progress(progress)
            save_embeddings(all_faces)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            progress["processed"].append(photo_path)
            save_progress(progress)

    save_progress(progress)
    save_embeddings(all_faces)

    if not all_faces:
        print("\n⚠ No faces found in any photos.")
        return

    # ── Cluster ──────────────────────────────────────────────────────────
    print(f"\n📊 Clustering {len(all_faces)} faces (threshold={similarity_threshold})...")
    embeddings = [d["embedding"] for d in all_faces]
    labels     = chinese_whispers_clustering(embeddings, threshold=similarity_threshold)

    num_clusters = len(set(labels))
    print(f"   Found {num_clusters} unique people/clusters.\n")

    # Attach labels to face records
    for i, d in enumerate(all_faces):
        d["cluster"] = labels[i]

    # ── Save representative crop per cluster ─────────────────────────────
    # For each cluster, save the highest-quality crop as representative
    cluster_to_faces = {}
    for d in all_faces:
        c = d["cluster"]
        cluster_to_faces.setdefault(c, []).append(d)

    # Clean clusters dir
    if os.path.exists(CLUSTERS_DIR):
        shutil.rmtree(CLUSTERS_DIR)
    os.makedirs(CLUSTERS_DIR)

    cluster_csv_rows = []

    for cluster_id in sorted(cluster_to_faces.keys()):
        faces_in_cluster = cluster_to_faces[cluster_id]
        n_faces          = len(faces_in_cluster)

        # Use middle crop as representative (usually a cleaner photo)
        rep = faces_in_cluster[len(faces_in_cluster) // 2]

        import cv2
        rep_img = cv2.imread(rep["crop_path"])
        rep_dest = os.path.join(CLUSTERS_DIR, f"cluster_{cluster_id:04d}_{n_faces}faces.jpg")
        if rep_img is not None:
            cv2.imwrite(rep_dest, rep_img)

        # Collect all photos this cluster appears in
        photos = list({d["photo_path"] for d in faces_in_cluster})

        cluster_csv_rows.append({
            "cluster_id":   cluster_id,
            "name":         "",   # USER FILLS THIS IN
            "faces_count":  n_faces,
            "photos_count": len(photos),
            "preview_image": rep_dest,
            "example_photos": " | ".join(photos[:3]),   # first 3 photos as reference
        })

    # ── Write CSV ─────────────────────────────────────────────────────────
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cluster_id","name","faces_count","photos_count","preview_image","example_photos"])
        writer.writeheader()
        writer.writerows(cluster_csv_rows)

    # Save full mapping (cluster label per face) for phase 2
    mapping_file = os.path.join(FACES_OUTPUT_DIR, "cluster_mapping.json")
    cluster_map = []
    for d in all_faces:
        cluster_map.append({
            "photo_path": d["photo_path"],
            "face_index": d["face_index"],
            "bbox":       d["bbox"],
            "cluster":    d["cluster"],
        })
    with open(mapping_file, "w") as f:
        json.dump(cluster_map, f, indent=2)

    print(f"✅ Phase 1 complete!\n")
    print(f"   Total faces detected : {len(all_faces)}")
    print(f"   Unique clusters found: {num_clusters}")
    print(f"   Preview images saved : {CLUSTERS_DIR}/")
    print(f"   CSV to fill in       : {CSV_PATH}\n")
    print(f"━━━ NEXT STEPS ━━━")
    print(f"1. Open '{CLUSTERS_DIR}/' — each image is ONE person's cluster.")
    print(f"   Filename format: cluster_0001_42faces.jpg (cluster #1 with 42 photos)")
    print(f"2. Open '{CSV_PATH}' and fill in the 'name' column for people you recognize.")
    print(f"   Leave blank for people you don't know — they won't get tagged.")
    print(f"3. Run Phase 2 to write names to your photo files.")
    print(f"\n💡 TIP: If you see the same person split into 2 clusters, lower --similarity.")
    print(f"        If different people are merged into 1 cluster, raise --similarity.")
    print(f"        Then re-run Phase 1 — it resumes without re-scanning photos.\n")


# ── Phase 2: Write names to metadata ────────────────────────────────────────

def run_phase2(photos_dir):
    import subprocess

    if not os.path.exists(CSV_PATH):
        print(f"✗ CSV not found: {CSV_PATH}\nRun Phase 1 first.")
        return

    mapping_file = os.path.join(FACES_OUTPUT_DIR, "cluster_mapping.json")
    if not os.path.exists(mapping_file):
        print(f"✗ Cluster mapping not found: {mapping_file}\nRun Phase 1 first.")
        return

    # Read cluster_id -> name from CSV
    cluster_to_name = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip()
            if name and name.lower() not in ("unknown", ""):
                cluster_to_name[int(row["cluster_id"])] = name

    if not cluster_to_name:
        print("⚠ No names found. Fill in the 'name' column in the CSV first.")
        return

    print(f"Found {len(cluster_to_name)} named clusters: {list(cluster_to_name.values())}\n")

    # Build photo_path -> [names] mapping
    with open(mapping_file) as f:
        cluster_map = json.load(f)

    photo_to_names = {}
    for entry in cluster_map:
        cluster = entry["cluster"]
        if cluster not in cluster_to_name:
            continue
        name = cluster_to_name[cluster]
        photo = entry["photo_path"]
        photo_to_names.setdefault(photo, set()).add(name)

    print(f"Tagging {len(photo_to_names)} photos...\n")

    success, failed = 0, 0

    for photo_path, names in photo_to_names.items():
        if not os.path.exists(photo_path):
            print(f"  ⚠ Not found: {photo_path}")
            failed += 1
            continue

        names = list(names)
        print(f"  {os.path.basename(photo_path)}  →  {', '.join(names)}")

        cmd = ["exiftool", "-overwrite_original"]
        for name in names:
            cmd += [
                f"-XMP:PersonInImage+={name}",
                f"-XMP:Subject+={name}",
                f"-IPTC:Keywords+={name}",
            ]
        cmd.append(photo_path)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                success += 1
            else:
                print(f"    ✗ exiftool: {result.stderr.strip()}")
                failed += 1
        except FileNotFoundError:
            print("\n✗ exiftool not installed.")
            print("  macOS:   brew install exiftool")
            print("  Ubuntu:  sudo apt install libimage-exiftool-perl")
            print("  Windows: https://exiftool.org/install.html")
            return

    print(f"\n✅ Phase 2 complete!")
    print(f"   Tagged:  {success} photos")
    print(f"   Failed:  {failed} photos")
    print(f"\nYour photos now have XMP:PersonInImage metadata ready for PhotoPrism.")


# ── Re-cluster only (no re-scanning) ────────────────────────────────────────

def run_recluster(similarity_threshold):
    """Re-runs clustering on already-extracted embeddings. Fast — no photo scanning."""
    all_faces = load_embeddings()
    if not all_faces:
        print("No embeddings found. Run Phase 1 first.")
        return

    print(f"Re-clustering {len(all_faces)} faces with threshold={similarity_threshold}...")
    embeddings = [d["embedding"] for d in all_faces]
    labels     = chinese_whispers_clustering(embeddings, threshold=similarity_threshold)
    num_clusters = len(set(labels))
    print(f"Found {num_clusters} clusters.")

    # Re-use phase1's cluster visualization
    for i, d in enumerate(all_faces):
        d["cluster"] = labels[i]

    if os.path.exists(CLUSTERS_DIR):
        shutil.rmtree(CLUSTERS_DIR)
    os.makedirs(CLUSTERS_DIR)

    import cv2
    cluster_to_faces = {}
    for d in all_faces:
        cluster_to_faces.setdefault(d["cluster"], []).append(d)

    cluster_csv_rows = []
    for cluster_id in sorted(cluster_to_faces.keys()):
        faces_in_cluster = cluster_to_faces[cluster_id]
        rep = faces_in_cluster[len(faces_in_cluster) // 2]
        rep_img = cv2.imread(rep["crop_path"])
        rep_dest = os.path.join(CLUSTERS_DIR, f"cluster_{cluster_id:04d}_{len(faces_in_cluster)}faces.jpg")
        if rep_img is not None:
            cv2.imwrite(rep_dest, rep_img)
        photos = list({d["photo_path"] for d in faces_in_cluster})
        cluster_csv_rows.append({
            "cluster_id":   cluster_id,
            "name":         "",
            "faces_count":  len(faces_in_cluster),
            "photos_count": len(photos),
            "preview_image": rep_dest,
            "example_photos": " | ".join(photos[:3]),
        })

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cluster_id","name","faces_count","photos_count","preview_image","example_photos"])
        writer.writeheader()
        writer.writerows(cluster_csv_rows)

    mapping_file = os.path.join(FACES_OUTPUT_DIR, "cluster_mapping.json")
    with open(mapping_file, "w") as f:
        json.dump([{"photo_path": d["photo_path"], "face_index": d["face_index"],
                    "bbox": d["bbox"], "cluster": d["cluster"]} for d in all_faces], f, indent=2)

    print(f"\n✅ Re-clustering done! {num_clusters} clusters → fill in {CSV_PATH}")


# ── Entry ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Face clustering & tagging pipeline")
    parser.add_argument("--phase", type=int, choices=[1, 2],
                        help="1=detect+cluster, 2=write metadata")
    parser.add_argument("--recluster", action="store_true",
                        help="Re-run clustering only (no photo scanning). Use after adjusting --similarity.")
    parser.add_argument("--photos_dir", type=str,
                        help="Root photos directory (searched recursively)")
    parser.add_argument("--similarity", type=float, default=SIMILARITY_DEFAULT,
                        help=f"Cosine similarity threshold (default {SIMILARITY_DEFAULT}). "
                              "Higher=stricter/more clusters, Lower=looser/fewer clusters.")
    args = parser.parse_args()

    os.makedirs(FACES_OUTPUT_DIR, exist_ok=True)

    if args.recluster:
        run_recluster(args.similarity)
        return

    if not args.phase:
        parser.print_help()
        return

    if not args.photos_dir:
        print("--photos_dir is required for phase 1 and 2")
        return

    photos_dir = os.path.abspath(args.photos_dir)
    if not os.path.isdir(photos_dir):
        print(f"✗ Not a directory: {photos_dir}")
        return

    print(f"Photos dir : {photos_dir}")
    print(f"Output dir : {os.path.abspath(FACES_OUTPUT_DIR)}\n")

    if args.phase == 1:
        run_phase1(photos_dir, args.similarity)
    elif args.phase == 2:
        run_phase2(photos_dir)


if __name__ == "__main__":
    main()