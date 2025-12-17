import os
import sys
import inspect
import logging
import argparse
from typing import Any, List, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fp32_onnx", required=True, type=str, help="The path to the original onnx"
    )
    parser.add_argument(
        "--converted_onnx",
        required=True,
        type=str,
        help="The path to the converted one",
    )
    parser.add_argument("--images_dir", required=True, type=str)
    parser.add_argument(
        "--num_images_to_test",
        required=True,
        type=int,
    )
    parser.add_argument("--preprocessor_path", required=True, type=str)
    parser.add_argument(
        "--ort_execution_provider",
        required=True,
        type=str,
        choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
    )
    parser.add_argument(
        "--batch_size",
        required=True,
        type=int,
    )
    parser.add_argument("--ort_num_threads", required=False, type=int, default=16)
    args = parser.parse_args()
    return args


def cosine_stats(original_outputs: np.ndarray, converted_outputs: np.ndarray):
    assert original_outputs.shape == converted_outputs.shape
    orig_norm = original_outputs / (
        np.linalg.norm(original_outputs, axis=1, keepdims=True) + 1e-12
    )
    conv_norm = converted_outputs / (
        np.linalg.norm(converted_outputs, axis=1, keepdims=True) + 1e-12
    )
    cosines = np.sum(orig_norm * conv_norm, axis=1)
    stats = {
        "mean": float(np.mean(cosines)),
        "median": float(np.median(cosines)),
        "p0_05": float(np.percentile(cosines, 0.05)),
        "p0_025": float(np.percentile(cosines, 0.025)),
        "p0_01": float(np.percentile(cosines, 0.01)),
        "p0_001": float(np.percentile(cosines, 0.001)),
    }
    return stats


def get_shell_logger(name: Optional[str] = None) -> logging.Logger:
    if name is None:
        frame = inspect.currentframe()
        outer = inspect.getouterframes(frame)[1]
        name = f"{outer.frame.f_globals['__name__']}.{outer.function}"

    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO"), None))
    return logger


def _list_images_in_dir(images_path: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    fnames = sorted(os.listdir(images_path))
    paths = [
        os.path.join(images_path, f)
        for f in fnames
        if os.path.splitext(f)[1].lower() in exts
    ]
    return paths


def run_and_collect_embeddings(
    logging_entity_name: str,
    onnx_path: str,
    images_path: str,
    preprocessor: Any,
    batch_size: int,
    num_images_to_test: int,
    ort_execution_provider: str,
    ort_num_threads: int,
):
    logger = get_shell_logger(logging_entity_name)
    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = ort_num_threads
    sess_options.intra_op_num_threads = ort_num_threads
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = True
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    sess = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=[ort_execution_provider],
    )

    input_name = sess.get_inputs()[0].name
    logger.info("ORT session is ready. Input name: %s", input_name)

    all_image_paths = _list_images_in_dir(images_path)
    if len(all_image_paths) < num_images_to_test:
        raise ValueError(
            f"Requested {num_images_to_test} images, but found only "
            f"{len(all_image_paths)} in {images_path}"
        )

    image_paths = all_image_paths[:num_images_to_test]
    logger.info(
        "Processing %d images from '%s' with batch_size=%d",
        len(image_paths),
        images_path,
        batch_size,
    )

    outputs = []
    pbar = tqdm(total=len(image_paths), desc="Calculating Image Embeddings")
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(img)
        inputs_numpy = preprocessor(images=imgs, return_tensors="np")[
            "pixel_values"
        ].astype(np.float32)
        ort_out = sess.run(None, {input_name: inputs_numpy})
        batch_embs = ort_out[0]  # Since ort_out is [batch_embs].
        outputs.append(batch_embs)
        pbar.update(len(batch_paths))
    pbar.close()
    embeddings = np.concatenate(outputs, axis=0)
    logger.info("Final embeddings shape: %s", embeddings.shape)
    return embeddings


def main():
    args = parse_args()

    processor = AutoProcessor.from_pretrained(args.preprocessor_path)

    common_kwargs = {
        "images_path": args.images_dir,
        "preprocessor": processor,
        "batch_size": args.batch_size,
        "num_images_to_test": args.num_images_to_test,
        "ort_execution_provider": args.ort_execution_provider,
        "ort_num_threads": args.ort_num_threads,
    }

    original_outputs = run_and_collect_embeddings(
        logging_entity_name="fp32_onnx",
        onnx_path=args.fp32_onnx,
        **common_kwargs,
    )
    converted_outputs = run_and_collect_embeddings(
        logging_entity_name="converted_onnx",
        onnx_path=args.converted_onnx,
        **common_kwargs,
    )

    stats = cosine_stats(original_outputs, converted_outputs)
    for k, v in stats.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
