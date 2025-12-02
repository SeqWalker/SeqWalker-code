

from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter:
    def __init__(self, log_dir: str, *args: Any, **kwargs: Any):

        self.writer = None
        if log_dir is not None and len(log_dir) > 0:
            self.writer = SummaryWriter(log_dir, *args, **kwargs)

    def __getattr__(self, item):
        if self.writer:
            return self.writer.__getattribute__(item)
        else:
            return lambda *args, **kwargs: None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.close()

    def add_video_from_np_images(
        self, video_name: str, step_idx: int, images: np.ndarray, fps: int = 10
    ) -> None:

        if not self.writer:
            return

        frame_tensors = [
            torch.from_numpy(np_arr).unsqueeze(0) for np_arr in images
        ]
        video_tensor = torch.cat(tuple(frame_tensors))

        self.writer.add_video(
            video_name, video_tensor, fps=fps, global_step=step_idx
        )
