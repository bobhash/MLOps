import os
import glob
import argparse
from typing import Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_imgs", type=int, required=True, help="Number of images to take"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size to save. Note: you can set other value later inside modelopt",
    )
    parser.add_argument(
        "--preprocessor_path", required=True, help="The path to the exact img processor"
    )
    parser.add_argument(
        "--calib_images_path", required=True, help="Path to directory with images"
    )
    parser.add_argument(
        "--savepath", required=True, help="The path to save resulting npz"
    )
    args = parser.parse_args()
    return args


def build_calibration_npz(
    image_dir: str,
    output_npz: str,
    processor,
    max_n: int,
    batch_size: int,
    patterns: Tuple[str, ...] = ("*.jpg", "*.jpeg", "*.png", "*.webp"),
) -> Tuple[str, Tuple[int, ...]]:
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(image_dir, pat)))
    paths = sorted(set(paths))
    assert len(paths) >= max_n
    paths = paths[:max_n]

    arrays = []
    for i in tqdm(range(0, len(paths), batch_size)):
        batch_paths = paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(img)
        if not imgs:
            continue
        out = processor(images=imgs, return_tensors="np")
        arrays.append(out["pixel_values"])

    pixel_values = np.concatenate(arrays, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(output_npz) or ".", exist_ok=True)
    np.savez(output_npz, IMAGES=pixel_values)
    return output_npz, pixel_values.shape


def main():
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.preprocessor_path)
    build_calibration_npz(
        image_dir=args.calib_images_path,
        output_npz=args.savepath,
        processor=processor,
        max_n=args.num_imgs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
