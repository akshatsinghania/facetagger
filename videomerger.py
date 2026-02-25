import os
import re
import subprocess
import tempfile

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_DIR   = "/path/to/your/directory"          # <-- change this
OUTPUT_FILE = "/path/to/output/combined.mp4"     # <-- change this
# ──────────────────────────────────────────────────────────────────────────────

# Order: (priority_number, name_keyword_lowercase)
ORDER = [
    (1,  "unnati"),
    (3,  "manju"),
    (4,  "prakash"),
    (5,  "bimal"),
    (6,  "akshat"),
    (7,  "sibu"),
    (8,  "aruna"),
    (9,  "boby"),
    (10, "uday"),
]

def find_video_for_name(directory, name_keyword):
    """Return the first mp4/mov file whose name (lowercase) contains the keyword."""
    for fname in os.listdir(directory):
        if fname.lower().endswith((".mp4", ".mov")):
            if name_keyword in fname.lower():
                return os.path.join(directory, fname)
    return None

def merge_videos(input_dir, output_file, order):
    ordered_files = []
    for priority, keyword in order:
        path = find_video_for_name(input_dir, keyword)
        if path:
            print(f"[{priority}] {keyword} → {os.path.basename(path)}")
            ordered_files.append(path)
        else:
            print(f"[{priority}] {keyword} → ⚠️  NOT FOUND, skipping")

    if not ordered_files:
        print("No videos found. Aborting.")
        return

    # Write a concat list file for ffmpeg
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_path = f.name
        for vpath in ordered_files:
            # ffmpeg concat demuxer needs escaped single-quotes
            safe = vpath.replace("'", r"'\''")
            f.write(f"file '{safe}'\n")

    print(f"\nConcat list written to: {list_path}")
    print("Merging with ffmpeg (stream copy – no re-encoding)…\n")

    cmd = [
        "ffmpeg",
        "-y",                        # overwrite output if exists
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",                # stream copy = zero quality loss
        output_file,
    ]

    result = subprocess.run(cmd)

    os.unlink(list_path)  # clean up temp file

    if result.returncode == 0:
        print(f"\n✅  Done! Combined video saved to:\n   {output_file}")
    else:
        print(f"\n❌  ffmpeg exited with code {result.returncode}")

if __name__ == "__main__":
    merge_videos(INPUT_DIR, OUTPUT_FILE, ORDER)
