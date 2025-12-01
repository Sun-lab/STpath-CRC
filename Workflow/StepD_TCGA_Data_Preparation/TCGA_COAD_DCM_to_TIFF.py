"""
STPath-COAD: TCGA DICOM to TIFF Conversion
==========================================

This script converts TCGA DICOM format histopathology images to TIFF format
for downstream analysis. Filters and processes only relevant pathology slides.

Author: Saishi Cui
Date: December 2025

Purpose: Convert TCGA DICOM histopathology files to TIFF format, filter out
non-pathological slides, and prepare data for STPath-COAD analysis pipeline.
"""

import os
import pydicom
import numpy as np
from PIL import Image
import math

# input and output path
base_dir = "TCGA_COAD_HE"
output_dir = "TCGA_COAD_HE_converted"
os.makedirs(output_dir, exist_ok=True)

for root, dirs, files in os.walk(base_dir):
    for fname in files:
        if not fname.lower().endswith(".dcm"):
            continue

        fpath = os.path.join(root, fname)
        try:
            # only read metadata to judge if need to process
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            modality = ds.get("Modality", "").upper()
            desc = ds.get("SeriesDescription", "").upper()
            frames = int(ds.get("NumberOfFrames", 0))

            # delete non-pathological slices
            if modality != "SM":
                os.remove(fpath)
                print(f"❌ Delete non-SM: {fname}")
                continue

            # delete Frozen samples
            if "FROZEN" in desc:
                os.remove(fpath)
                print(f"❌ Delete Frozen: {fname}")
                continue

            # delete abnormal tile number
            if frames < 2000 or frames > 60000:
                os.remove(fpath)
                print(f"❌ Delete abnormal tile number ({frames}): {fname}")
                continue

            # get slide ID (only keep pure value)
            raw_id = ds.get((0x0040, 0x0512), "unknownslide")
            slide_id = raw_id.value if hasattr(raw_id, "value") else str(raw_id)
            slide_id = slide_id.replace(" ", "_")
            out_name = f"{slide_id}.tiff"
            out_path = os.path.join(output_dir, out_name)

            # if already exists, skip
            if os.path.exists(out_path):
                print(f"⏭️ Already exists, skipping: {out_name}")
                continue

            # load full pixel and stitch
            ds = pydicom.dcmread(fpath)
            tile_h, tile_w = ds.Rows, ds.Columns
            total_w = int(ds.TotalPixelMatrixColumns)
            total_h = int(ds.TotalPixelMatrixRows)
            cols = math.ceil(total_w / tile_w)
            rows = math.ceil(total_h / tile_h)

            print(f"🧩 stitching: {fname} → {cols}×{rows} tiles")

            tiles = ds.pixel_array
            stitched = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

            for idx, tile in enumerate(tiles):
                r = idx // cols
                c = idx % cols
                stitched[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w, :] = tile

            stitched = stitched[:total_h, :total_w, :]

            # save as normal TIFF (not bigtiff)
            Image.fromarray(stitched).save(out_path)
            print(f"✅ save done: {out_name}")

        except Exception as e:
            print(f"⚠️ Process failed {fname}: {e}")
