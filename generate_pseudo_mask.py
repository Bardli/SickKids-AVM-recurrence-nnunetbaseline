#!/usr/bin/env python3
"""
Create foreground masks for nnUNet images (multiprocessing, Otsu-based).

- Reads from: /bdm-das/ADSP_v1/nnUNet_data/nnUNet_raw/Dataset035_PDCAD_T1/imagesTr
- Writes to : /bdm-das/ADSP_v1/nnUNet_data/nnUNet_raw/Dataset035_PDCAD_T1/labelsTr
- Output filenames drop the '_0000' suffix (e.g., case_001_0000.nii.gz -> case_001.nii.gz)

Requires: SimpleITK, tqdm
    pip install SimpleITK tqdm
"""

import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count, get_context
from functools import partial

import SimpleITK as sitk
from tqdm import tqdm

# ================== CONFIG ==================
# IMAGES_DIR = Path("/bdm-das/ADSP_v1/nnUNet_data/nnUNet_raw/Dataset035_PDCAD_T1/imagesTr")
# LABELS_DIR = Path("/bdm-das/ADSP_v1/nnUNet_data/nnUNet_raw/Dataset035_PDCAD_T1/labelsTr")
# IMAGES_DIR = Path("/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset003_AVM_T1+C/imagesTr")
# LABELS_DIR = Path("/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset003_AVM_T1+C/labelsTr")
# IMAGES_DIR = Path("/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset004_AVM_TOFMRA/imagesTr")
# LABELS_DIR = Path("/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset004_AVM_TOFMRA/labelsTr")
# IMAGES_DIR = Path("/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset033_AVM_T1+C/imagesTr")
# LABELS_DIR = Path("/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset033_AVM_T1+C/labelsTr")
# IMAGES_DIR = Path("/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset044_AVM_TOFMRA/imagesTr")
# LABELS_DIR = Path("/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset044_AVM_TOFMRA/labelsTr")
# IMAGES_DIR = Path("/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset055_AVM_T1+C_TOFMRA/imagesTr")
# LABELS_DIR = Path("/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset055_AVM_T1+C_TOFMRA/labelsTr")
IMAGES_DIR = Path("/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset066_AVM_T1+C_postSurg/imagesTr")
LABELS_DIR = Path("/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset066_AVM_T1+C_postSurg/labelsTr")

GLOB_PATTERN = "*_0000.nii.gz"
OVERWRITE = True
NUM_WORKERS = max(1, min(20, cpu_count()))
CHUNKSIZE = 2

# Foreground extraction options
USE_OTSU = True                  # True: Otsu threshold -> foreground=1, else use >0 (nonzero) heuristic
CLOSE_RADIUS = 1                 # morphological closing radius (voxels); 0 disables
FILL_HOLES = True                # fill holes in binary mask
KEEP_N_LARGEST = 1               # keep N largest connected components; 0 disables
# ============================================


def _to_label_name(img_path: Path) -> Path:
    name = img_path.name
    if name.endswith("_0000.nii.gz"):
        name = name[:-len("_0000.nii.gz")] + ".nii.gz"
    else:
        name = name.replace("_0000", "")
        if not name.endswith(".nii.gz"):
            name += ".nii.gz"
    return LABELS_DIR / name


def _init_worker():
    # Avoid over-subscribing CPU threads inside each process
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _make_fg_mask(img: sitk.Image) -> sitk.Image:
    """
    Create a uint8 mask (0 background, 1 foreground) using Otsu (or >0 heuristic),
    with optional morphology + largest-component filtering.
    """
    if USE_OTSU:
        mask = sitk.OtsuThreshold(img, 0, 1)  # 1 = foreground above threshold
    else:
        mask = sitk.Cast(sitk.Greater(img, 0), sitk.sitkUInt8)

    # Morphological cleanup (optional)
    if CLOSE_RADIUS and CLOSE_RADIUS > 0:
        mask = sitk.BinaryMorphologicalClosing(mask, [CLOSE_RADIUS]*3)

    if FILL_HOLES:
        mask = sitk.BinaryFillhole(mask)

    # Keep N largest components (optional)
    if KEEP_N_LARGEST and KEEP_N_LARGEST > 0:
        cc = sitk.ConnectedComponent(mask)
        relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
        # Build mask of top N labels
        # relabeled labels start at 1 for largest; 0 is background
        comp_mask = sitk.Equal(relabeled, 1)
        for k in range(2, KEEP_N_LARGEST + 1):
            comp_mask = sitk.Or(comp_mask, sitk.Equal(relabeled, k))
        mask = sitk.Cast(comp_mask, sitk.sitkUInt8)

    # Preserve geometry
    mask.CopyInformation(img)
    return mask


def _process_one(img_p: Path, overwrite: bool) -> tuple[Path, str]:
    """
    Worker task: read image, build foreground mask, write label.
    Returns (output_path, status) where status in {"created","skipped","overwritten","error:..."}.
    """
    try:
        out_p = _to_label_name(img_p)
        if out_p.exists() and not overwrite:
            return (out_p, "skipped")
        status = "overwritten" if out_p.exists() else "created"

        img = sitk.ReadImage(str(img_p))
        mask = _make_fg_mask(img)
        sitk.WriteImage(mask, str(out_p), useCompression=True)
        return (out_p, status)
    except Exception as e:
        return (img_p, f"error: {e}")


def main():
    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Images directory not found: {IMAGES_DIR}")
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(IMAGES_DIR.glob(GLOB_PATTERN))
    if not image_paths:
        print(f"No images matching '{GLOB_PATTERN}' found in {IMAGES_DIR}")
        sys.exit(0)

    print(f"Found {len(image_paths)} image(s). Writing foreground masks to: {LABELS_DIR}")
    print(f"Workers: {NUM_WORKERS} | Overwrite: {OVERWRITE} | Otsu: {USE_OTSU} | "
          f"Close r={CLOSE_RADIUS} | FillHoles={FILL_HOLES} | KeepLargest={KEEP_N_LARGEST}")

    task_fn = partial(_process_one, overwrite=OVERWRITE)

    created = skipped = overwritten = errors = 0
    with get_context("spawn").Pool(processes=NUM_WORKERS, initializer=_init_worker) as pool:
        for out_p, status in tqdm(pool.imap_unordered(task_fn, image_paths, chunksize=CHUNKSIZE),
                                  total=len(image_paths), ncols=100):
            if status == "created":
                created += 1
            elif status == "skipped":
                skipped += 1
            elif status == "overwritten":
                overwritten += 1
            elif status.startswith("error"):
                errors += 1
                print(f"\n[ERROR] {out_p}: {status}", flush=True)

    print(f"\nDone. created={created}, overwritten={overwritten}, skipped={skipped}, errors={errors}")


if __name__ == "__main__":
    main()
