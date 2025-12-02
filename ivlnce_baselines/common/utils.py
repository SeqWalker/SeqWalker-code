from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple
import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from habitat_baselines.common.tensor_dict import DictTree
from torch import Size, Tensor

def extract_instruction_tokens(
    observations: List[Dict],
    instruction_sensor_uuid: str,
    tokens_uuid: str = "tokens",
) -> Dict[str, Any]:
    if (
        instruction_sensor_uuid not in observations[0]
        or instruction_sensor_uuid == "pointgoal_with_gps_compass"
    ):
        return observations
    for i in range(len(observations)):
        if (
            isinstance(observations[i][instruction_sensor_uuid], dict)
            and tokens_uuid in observations[i][instruction_sensor_uuid]
        ):
            observations[i][instruction_sensor_uuid] = observations[i][
                instruction_sensor_uuid
            ]["tokens"]
        else:
            break
    return observations

def extract_instruction_texts(
    observations: List[Dict],
    instruction_sensor_uuid: str,
    tokens_uuid: str = "text",
) -> Dict[str, Any]:
    if (
        instruction_sensor_uuid not in observations[0]
        or instruction_sensor_uuid == "pointgoal_with_gps_compass"
    ):
        return observations
    for i in range(len(observations)):
        if (
            isinstance(observations[i][instruction_sensor_uuid], dict)
            and tokens_uuid in observations[i][instruction_sensor_uuid]
        ):
            observations[i][instruction_sensor_uuid] = observations[i][
                instruction_sensor_uuid
            ]["text"]
        else:
            break
    return observations


def single_frame_box_shape(box: spaces.Box) -> spaces.Box:
    if len(box.shape) < 4:
        return box

    return spaces.Box(
        low=box.low.min(),
        high=box.high.max(),
        shape=box.shape[1:],
        dtype=box.high.dtype,
    )


def check_if_type_is_np_uint_32(arr):
    if not isinstance(arr, np.ndarray):
        return False
    return arr.dtype == np.uint32

def batch_obs(
    observations: List[DictTree],
    device: Optional[torch.device] = None,
    ignore_keys: Optional[Set[str]] = None,
) -> Dict:
    if ignore_keys is None:
        ignore_keys = {"env_name"}

    batch: DefaultDict[str, List] = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            if check_if_type_is_np_uint_32(obs[sensor]):
                obs[sensor] = np.int32(obs[sensor])
            if sensor not in ignore_keys:
                obs[sensor] = torch.as_tensor(obs[sensor])
            batch[sensor].append(obs[sensor])

    batch_t: Dict = dict()

    for sensor in batch:
        if sensor not in ignore_keys:
            batch_t[sensor] = torch.stack(batch[sensor], dim=0).to(device)
        else:
            batch_t[sensor] = batch[sensor]

    return batch_t


def batch_to(
    batch: Tuple,
    device: torch.device = None,
    non_blocking: bool = True,
) -> Tuple:
    (
        observations_batch,
        prev_actions_batch,
        episode_not_done_masks,
        tour_not_done_mask,
        corrected_actions_batch,
        weights_batch,
    ) = batch

    observations_batch = {
        k: v.to(
            device=device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        )
        for k, v in observations_batch.items()
    }

    return (
        observations_batch,
        prev_actions_batch.to(device=device, non_blocking=non_blocking),
        episode_not_done_masks.to(device=device, non_blocking=non_blocking),
        tour_not_done_mask.to(device=device, non_blocking=non_blocking),
        corrected_actions_batch.to(device=device, non_blocking=non_blocking),
        weights_batch.to(device=device, non_blocking=non_blocking),
    )

def add_batched_data_to_observations(
    observations: List[Dict],
    batched_data: torch.LongTensor,
    batched_data_key: str,
):
    if batched_data is not None:
        for i in range(len(observations)):
            observations[i][batched_data_key] = batched_data[i]
    return observations


class CustomFixedCategorical(torch.distributions.Categorical):
    def sample(
        self, sample_shape: Size = torch.Size()
    ) -> Tensor:
        return super().sample(sample_shape).unsqueeze(-1)

    def log_prob(self, actions: Tensor) -> Tensor:
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
        self.num_outputs = num_outputs

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor) -> CustomFixedCategorical:
        if x.shape[-1] != self.num_outputs:
            x = self.linear(x)
        return CustomFixedCategorical(logits=x)
