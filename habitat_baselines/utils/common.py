
import glob
import os
import re
import shutil
import tarfile
from collections import defaultdict
from io import BytesIO
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
from gym.spaces import Box
from PIL import Image
from torch import Size, Tensor
from torch import nn as nn

from habitat import logger
from habitat.core.dataset import Episode
from habitat.core.utils import try_cv2_import
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import images_to_video
from habitat_baselines.common.tensor_dict import DictTree, TensorDict
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

cv2 = try_cv2_import()


class CustomFixedCategorical(torch.distributions.Categorical):
    def sample(
        self, sample_shape: Size = torch.Size()
    ) -> Tensor:
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions: Tensor) -> Tensor:
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor) -> CustomFixedCategorical:
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)


def linear_decay(epoch: int, total_num_updates: int) -> float:

    return 1 - (epoch / float(total_num_updates))


@torch.no_grad()
@profiling_wrapper.RangeContext("batch_obs")
def batch_obs(
    observations: List[DictTree],
    device: Optional[torch.device] = None,
) -> TensorDict:

    batch: DefaultDict[str, List] = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(torch.as_tensor(obs[sensor]))

    batch_t: TensorDict = TensorDict()

    for sensor in batch:
        batch_t[sensor] = torch.stack(batch[sensor], dim=0)

    return batch_t.map(lambda v: v.to(device))


def get_checkpoint_id(ckpt_path: str) -> Optional[int]:

    ckpt_path = os.path.basename(ckpt_path)
    nums: List[int] = [int(s) for s in ckpt_path.split(".") if s.isdigit()]
    if len(nums) > 0:
        return nums[-1]
    return None


def poll_checkpoint_folder(
    checkpoint_folder: str, previous_ckpt_ind: int
) -> Optional[str]:

    assert os.path.isdir(checkpoint_folder), (
        f"invalid checkpoint folder " f"path {checkpoint_folder}"
    )
    models_paths = list(
        filter(os.path.isfile, glob.glob(checkpoint_folder + "/*"))
    )
    models_paths.sort(key=os.path.getmtime)
    ind = previous_ckpt_ind + 1
    if ind < len(models_paths):
        return models_paths[ind]
    return None


def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    episode_id: Union[int, str],
    checkpoint_idx: int,
    metrics: Dict[str, float],
    tb_writer: TensorboardWriter,
    fps: int = 10,
) -> None:

    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}={v:.2f}")

    video_name = f"episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(
        metric_strs
    )
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name)
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(
            f"episode{episode_id}", checkpoint_idx, images, fps=fps
        )


def tensor_to_depth_images(tensor: Union[torch.Tensor, List]) -> np.ndarray:

    images = []

    for img_tensor in tensor:
        image = img_tensor.permute(1, 2, 0).cpu().numpy() * 255
        images.append(image)

    return images


def tensor_to_bgr_images(
    tensor: Union[torch.Tensor, Iterable[torch.Tensor]]
) -> List[np.ndarray]:

    images = []

    for img_tensor in tensor:
        img = img_tensor.permute(1, 2, 0).cpu().numpy() * 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)

    return images


def image_resize_shortest_edge(
    img: Tensor, size: int, channels_last: bool = False
) -> torch.Tensor:

    img = torch.as_tensor(img)
    no_batch_dim = len(img.shape) == 3
    if len(img.shape) < 3 or len(img.shape) > 5:
        raise NotImplementedError()
    if no_batch_dim:
        img = img.unsqueeze(0)
    h, w = get_image_height_width(img, channels_last=channels_last)
    if channels_last:
        if len(img.shape) == 4:

            img = img.permute(0, 3, 1, 2)
        else:

            img = img.permute(0, 1, 4, 2, 3)

    scale = size / min(h, w)
    h = int(h * scale)
    w = int(w * scale)
    img = torch.nn.functional.interpolate(
        img.float(), size=(h, w), mode="area"
    ).to(dtype=img.dtype)
    if channels_last:
        if len(img.shape) == 4:

            img = img.permute(0, 2, 3, 1)
        else:

            img = img.permute(0, 1, 3, 4, 2)
    if no_batch_dim:
        img = img.squeeze(dim=0)
    return img


def center_crop(
    img: Tensor, size: Union[int, Tuple[int, int]], channels_last: bool = False
) -> Tensor:

    h, w = get_image_height_width(img, channels_last=channels_last)

    if isinstance(size, int):
        size_tuple: Tuple[int, int] = (int(size), int(size))
    else:
        size_tuple = size
    assert len(size_tuple) == 2, "size should be (h,w) you wish to resize to"
    cropy, cropx = size_tuple

    startx = w // 2 - (cropx // 2)
    starty = h // 2 - (cropy // 2)
    if channels_last:
        return img[..., starty : starty + cropy, startx : startx + cropx, :]
    else:
        return img[..., starty : starty + cropy, startx : startx + cropx]


def get_image_height_width(
    img: Union[Box, np.ndarray, torch.Tensor], channels_last: bool = False
) -> Tuple[int, int]:
    if img.shape is None or len(img.shape) < 3 or len(img.shape) > 5:
        raise NotImplementedError()
    if channels_last:

        h, w = img.shape[-3:-1]
    else:

        h, w = img.shape[-2:]
    return h, w


def overwrite_gym_box_shape(box: Box, shape) -> Box:
    if box.shape == shape:
        return box
    shape = list(shape) + list(box.shape[len(shape) :])
    low = box.low if np.isscalar(box.low) else np.min(box.low)
    high = box.high if np.isscalar(box.high) else np.max(box.high)
    return Box(low=low, high=high, shape=shape, dtype=box.dtype)


def get_scene_episode_dict(episodes: List[Episode]) -> Dict:
    scene_ids = []
    scene_episode_dict = {}

    for episode in episodes:
        if episode.scene_id not in scene_ids:
            scene_ids.append(episode.scene_id)
            scene_episode_dict[episode.scene_id] = [episode]
        else:
            scene_episode_dict[episode.scene_id].append(episode)

    return scene_episode_dict


def base_plus_ext(path: str) -> Union[Tuple[str, str], Tuple[None, None]]:

    match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", path)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def valid_sample(sample: Optional[Any]) -> bool:

    return (
        sample is not None
        and isinstance(sample, dict)
        and len(list(sample.keys())) > 0
        and not sample.get("__bad__", False)
    )


def img_bytes_2_np_array(
    x: Tuple[int, torch.Tensor, bytes]
) -> Tuple[int, torch.Tensor, bytes, np.ndarray]:

    images = []
    img_bytes: bytes
    for img_bytes in x[3:]:
        bytes_obj = BytesIO()
        bytes_obj.write(img_bytes)
        image = np.array(Image.open(bytes_obj))
        img = image.transpose(2, 0, 1)
        img = img / 255.0
        images.append(img)
    return (*x[0:3], np.array(images, dtype=np.float32))


def create_tar_archive(archive_path: str, dataset_path: str) -> None:

    logger.info("[ Creating tar archive. This will take a few minutes. ]")

    with tarfile.open(archive_path, "w:gz") as tar:
        for file in sorted(os.listdir(dataset_path)):
            tar.add(os.path.join(dataset_path, file))


def delete_folder(path: str) -> None:
    shutil.rmtree(path)
