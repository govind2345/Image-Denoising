"""
prepare_dataset.py
------------------
Converts the raw MVTec AD dataset structure into the format expected by model.py.

Raw MVTec layout (per category, under DATASET_ROOT):
    <category>/
        train/good/          <- defect-free images used as GT
        test/<defect_type>/  <- defective images used as Degraded input
        ground_truth/<defect_type>/  <- binary masks (*_mask.png)

Target layout (per category, under OUTPUT_ROOT):
    <category>/
        Train/
            Degraded_image/   <- synthetic noisy version of defective test images
            GT_clean_image/   <- corresponding clean (good) image (nearest match)
            Defect_mask/      <- binary defect mask (*_mask.png)
        Val/
            Degraded_image/
            GT_clean_image/
            Defect_mask/

Strategy:
  - Defective test images  -> Degraded_image (with added Gaussian noise)
  - A randomly selected good training image -> GT_clean_image (same filename)
  - Existing ground_truth mask -> Defect_mask (already named *_mask.png)
  - 80% of defective images go to Train, 20% to Val
  - test/good images are skipped (no mask available)

NOTE: Output is written to a SEPARATE directory (OUTPUT_ROOT) to avoid
      Windows case-insensitive collisions between source `train/` and
      output `Train/`.
"""

import os
import shutil
import stat
import random
import numpy as np
from PIL import Image


def _remove_readonly(func, path, _):
    """Error handler for shutil.rmtree to clear read-only flag on Windows."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _safe_copy(src, dst):
    """Copy file, clearing read-only flag on destination if it already exists."""
    if os.path.exists(dst):
        os.chmod(dst, stat.S_IWRITE)
    shutil.copy2(src, dst)
    # Ensure destination is writable for future overwrites
    os.chmod(dst, stat.S_IWRITE | stat.S_IREAD)

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET_ROOT = "Denoising_Dataset_train_val"   # raw MVTec data (read-only source)
OUTPUT_ROOT  = "Denoising_Dataset_prepared"     # prepared output (separate dir, avoids Windows case collision)
TRAIN_RATIO  = 0.8          # fraction of defective samples used for training
NOISE_STD    = 25           # Gaussian noise std dev added to degraded images (0-255 scale)
SEED         = 42
random.seed(SEED)
np.random.seed(SEED)
# ───────────────────────────────────────────────────────────────────────────────


def add_gaussian_noise(img_array: np.ndarray, std: float = 25) -> np.ndarray:
    """Add Gaussian noise to an image array (uint8, 0-255)."""
    noise = np.random.normal(0, std, img_array.shape).astype(np.float32)
    noisy = img_array.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def process_category(src_category_path: str, category: str) -> None:
    train_good_dir   = os.path.join(src_category_path, "train", "good")
    test_dir         = os.path.join(src_category_path, "test")
    gt_dir           = os.path.join(src_category_path, "ground_truth")

    out_category_path = os.path.join(OUTPUT_ROOT, category)

    # ── Collect good images for GT ────────────────────────────────────────────
    good_images = sorted([
        f for f in os.listdir(train_good_dir)
        if f.lower().endswith(".png")
    ]) if os.path.isdir(train_good_dir) else []

    if not good_images:
        print(f"  [SKIP] {category}: no good training images found")
        return

    # ── Collect defective test images with their masks ────────────────────────
    samples = []   # list of (test_img_path, mask_path, img_filename, mask_filename)

    for defect_type in sorted(os.listdir(test_dir)):
        if defect_type == "good":
            continue  # no masks for good test images

        defect_test_dir = os.path.join(test_dir, defect_type)
        defect_mask_dir = os.path.join(gt_dir, defect_type)

        if not os.path.isdir(defect_test_dir) or not os.path.isdir(defect_mask_dir):
            print(f"  [WARN] {category}/{defect_type}: missing test or mask dir, skipping")
            continue

        for img_file in sorted(os.listdir(defect_test_dir)):
            if not img_file.lower().endswith(".png"):
                continue

            stem      = os.path.splitext(img_file)[0]          # e.g. "000"
            mask_name = f"{stem}_mask.png"
            mask_path = os.path.join(defect_mask_dir, mask_name)

            if not os.path.exists(mask_path):
                print(f"  [WARN] mask not found: {mask_path}, skipping")
                continue

            samples.append((
                os.path.join(defect_test_dir, img_file),
                mask_path,
                img_file,          # original filename (e.g. 000.png)
                mask_name          # mask filename    (e.g. 000_mask.png)
            ))

    if not samples:
        print(f"  [SKIP] {category}: no valid defective samples found")
        return

    # ── Clean previous output ─────────────────────────────────────────────────
    for split_dir in ("Train", "Val"):
        out_path = os.path.join(out_category_path, split_dir)
        if os.path.isdir(out_path):
            shutil.rmtree(out_path, onerror=_remove_readonly)

    # ── Train / Val split ─────────────────────────────────────────────────────
    random.shuffle(samples)
    n_train = max(1, int(len(samples) * TRAIN_RATIO))
    splits  = {"Train": samples[:n_train], "Val": samples[n_train:]}

    if not splits["Val"]:
        # Guarantee at least one val sample
        splits["Val"]   = [splits["Train"][-1]]
        splits["Train"] = splits["Train"][:-1]

    print(f"  {category}: {len(splits['Train'])} train / {len(splits['Val'])} val samples")

    # ── Write files ───────────────────────────────────────────────────────────
    for split_name, split_samples in splits.items():
        degraded_out = os.path.join(out_category_path, split_name, "Degraded_image")
        clean_out    = os.path.join(out_category_path, split_name, "GT_clean_image")
        mask_out     = os.path.join(out_category_path, split_name, "Defect_mask")

        os.makedirs(degraded_out, exist_ok=True)
        os.makedirs(clean_out,    exist_ok=True)
        os.makedirs(mask_out,     exist_ok=True)

        for (test_img_path, mask_path, img_filename, mask_filename) in split_samples:
            # 1. Degraded image — original defective image + Gaussian noise
            img_arr  = np.array(Image.open(test_img_path).convert("RGB"))
            noisy    = add_gaussian_noise(img_arr, std=NOISE_STD)
            Image.fromarray(noisy).save(os.path.join(degraded_out, img_filename))

            # 2. GT clean image — randomly sampled good image (same filename for traceability)
            good_file = random.choice(good_images)
            _safe_copy(
                os.path.join(train_good_dir, good_file),
                os.path.join(clean_out, img_filename)    # rename to match degraded filename
            )

            # 3. Defect mask — copy as-is (already named *_mask.png)
            _safe_copy(mask_path, os.path.join(mask_out, mask_filename))


def main() -> None:
    categories = sorted([
        d for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d))
    ])

    print(f"Found {len(categories)} categories: {categories}\n")

    for category in categories:
        src_category_path = os.path.join(DATASET_ROOT, category)
        print(f"Processing: {category}")
        try:
            process_category(src_category_path, category)
        except Exception as e:
            print(f"  [ERROR] {category}: {e}")

    print(f"\nDone! Dataset preparation complete.")
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"You can now run:  python model.py")


if __name__ == "__main__":
    main()
