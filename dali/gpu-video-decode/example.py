from typing import List

import cv2
import pandas as pd
from tqdm import tqdm
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from joblib import Parallel, delayed
from nvidia.dali.pipeline import Pipeline


class VideoPipe(Pipeline):
    def __init__(
        self,
        batch_size: int,
        num_threads: int,
        device_id: int,
        filenames: List[str],
        sequence_length: int,
        step: int,
    ):
        super().__init__(batch_size, num_threads, device_id, seed=12)
        self.filenames = filenames
        self.sequence_length = sequence_length
        self.step = step

    def define_graph(self):
        frames = fn.readers.video(
            device="gpu",
            filenames=self.filenames,
            sequence_length=self.sequence_length,
            normalized=False,
            random_shuffle=False,
            image_type=types.RGB,
            dtype=types.UINT8,
            step=self.step,
            file_list_include_preceding_frame=False,
        )
        return frames


def process_video_dali(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    sequence_length = int(total_frames / fps)

    pipe = VideoPipe(
        batch_size=1,
        num_threads=4,
        device_id=3,
        filenames=[video_path],
        sequence_length=sequence_length,
        step=int(fps),
    )
    pipe.build()
    pipe_out = pipe.run()
    out = pipe_out[0].as_cpu().as_array()
    print(out.shape)


def process_video_python(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % fps == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    print(f"Sampled {len(frames)} frames")


def main():
    video_paths = pd.read_parquet("video_paths.parquet").values
    Parallel(n_jobs=8, backend="threading")(
        delayed(process_video_dali)(path[0]) for path in tqdm(video_paths)
    )
    Parallel(n_jobs=8, backend="threading")(
        delayed(process_video_python)(path[0]) for path in tqdm(video_paths)
    )


if __name__ == "__main__":
    main()
