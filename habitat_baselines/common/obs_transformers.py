
import abc
import copy
import numbers
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from gym import spaces
from torch import nn

from habitat.config import Config
from habitat.core.logging import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import (
    center_crop,
    get_image_height_width,
    image_resize_shortest_edge,
    overwrite_gym_box_shape,
)


class ObservationTransformer(nn.Module, metaclass=abc.ABCMeta):


    def transform_observation_space(
        self, observation_space: spaces.Dict, **kwargs
    ):
        return observation_space

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Config):
        pass

    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return observations


@baseline_registry.register_obs_transformer()
class ResizeShortestEdge(ObservationTransformer):


    def __init__(
        self,
        size: int,
        channels_last: bool = True,
        trans_keys: Tuple[str] = ("rgb", "depth", "semantic"),
    ):

        super(ResizeShortestEdge, self).__init__()
        self._size: int = size
        self.channels_last: bool = channels_last
        self.trans_keys: Tuple[str] = trans_keys

    def transform_observation_space(
        self,
        observation_space: spaces.Dict,
    ):
        size = self._size
        observation_space = copy.deepcopy(observation_space)
        if size:
            for key in observation_space.spaces:
                if key in self.trans_keys:

                    h, w = get_image_height_width(
                        observation_space.spaces[key], channels_last=True
                    )
                    if size == min(h, w):
                        continue
                    scale = size / min(h, w)
                    new_h = int(h * scale)
                    new_w = int(w * scale)
                    new_size = (new_h, new_w)
                    logger.info(
                        "Resizing observation of %s: from %s to %s"
                        % (key, (h, w), new_size)
                    )
                    observation_space.spaces[key] = overwrite_gym_box_shape(
                        observation_space.spaces[key], new_size
                    )
        return observation_space

    def _transform_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return image_resize_shortest_edge(
            obs, self._size, channels_last=self.channels_last
        )

    @torch.no_grad()
    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self._size is not None:
            observations.update(
                {
                    sensor: self._transform_obs(observations[sensor])
                    for sensor in self.trans_keys
                    if sensor in observations
                }
            )
        return observations

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.RL.POLICY.OBS_TRANSFORMS.RESIZE_SHORTEST_EDGE.SIZE)


@baseline_registry.register_obs_transformer()
class CenterCropper(ObservationTransformer):


    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        channels_last: bool = True,
        trans_keys: Tuple[str] = ("rgb", "depth", "semantic"),
    ):

        super().__init__()
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        assert len(size) == 2, "forced input size must be len of 2 (h, w)"
        self._size = size
        self.channels_last = channels_last
        self.trans_keys = trans_keys

    def transform_observation_space(
        self,
        observation_space: spaces.Dict,
    ):
        size = self._size
        observation_space = copy.deepcopy(observation_space)
        if size:
            for key in observation_space.spaces:
                if (
                    key in self.trans_keys
                    and observation_space.spaces[key].shape[-3:-1] != size
                ):
                    h, w = get_image_height_width(
                        observation_space.spaces[key], channels_last=True
                    )
                    logger.info(
                        "Center cropping observation size of %s from %s to %s"
                        % (key, (h, w), size)
                    )

                    observation_space.spaces[key] = overwrite_gym_box_shape(
                        observation_space.spaces[key], size
                    )
        return observation_space

    def _transform_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return center_crop(
            obs,
            self._size,
            channels_last=self.channels_last,
        )

    @torch.no_grad()
    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self._size is not None:
            observations.update(
                {
                    sensor: self._transform_obs(observations[sensor])
                    for sensor in self.trans_keys
                    if sensor in observations
                }
            )
        return observations

    @classmethod
    def from_config(cls, config: Config):
        cc_config = config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER
        return cls(
            (
                cc_config.HEIGHT,
                cc_config.WIDTH,
            )
        )


class _DepthFrom(Enum):
    Z_VAL = 0
    OPTI_CENTER = 1


class CameraProjection(metaclass=abc.ABCMeta):


    def __init__(
        self,
        img_h: int,
        img_w: int,
        R: Optional[torch.Tensor] = None,
        depth_from: _DepthFrom = _DepthFrom.OPTI_CENTER,
    ):

        self.img_h = img_h
        self.img_w = img_w
        self.depth_from = depth_from


        if R is not None:
            self.R = R.float()
        else:
            self.R = None

    @abc.abstractmethod
    def projection(
        self, world_pts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:


    @abc.abstractmethod
    def unprojection(
        self, with_rotation: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:


    @property
    def rotation(self):

        if self.R is None:
            return torch.eye(3, dtype=torch.float32)
        else:
            return self.R

    @property
    def shape(self):

        return (self.img_h, self.img_w)

    def size(self):

        return self.shape

    def camcoord2worldcoord(self, pts: torch.Tensor):

        if self.R is None:
            return pts
        else:

            _h, _w, _ = pts.shape

            rotated_pts = torch.matmul(pts.view((-1, 3)), self.R.T)
            return rotated_pts.view(_h, _w, 3)

    def worldcoord2camcoord(self, pts: torch.Tensor):

        if self.R is None:
            return pts
        else:

            _h, _w, _ = pts.shape

            rotated_pts = torch.matmul(pts.view((-1, 3)), self.R)
            return rotated_pts.view(_h, _w, 3)


class PerspectiveProjection(CameraProjection):


    def __init__(
        self,
        img_h: int,
        img_w: int,
        f: Optional[float] = None,
        R: Optional[torch.Tensor] = None,
    ):

        super(PerspectiveProjection, self).__init__(
            img_h, img_w, R, _DepthFrom.Z_VAL
        )
        if f is None:
            self.f = max(img_h, img_w) / 2
        else:
            self.f = f

    def projection(
        self, world_pts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        world_pts = self.worldcoord2camcoord(world_pts)


        img_pts = self.f * world_pts / torch.abs(world_pts[..., 2:3])
        cx = self.img_w / 2
        cy = self.img_h / 2
        u = img_pts[..., 0] + cx
        v = img_pts[..., 1] + cy


        mapx = 2 * u / self.img_w - 1.0
        mapy = 2 * v / self.img_h - 1.0
        proj_pts = torch.stack([mapx, mapy], dim=-1)


        valid_mask = torch.abs(proj_pts).max(-1)[0] <= 1
        valid_mask *= img_pts[..., 2] > 0
        return proj_pts, valid_mask

    def unprojection(
        self, with_rotation: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v, u = torch.meshgrid(
            torch.arange(self.img_h), torch.arange(self.img_w)
        )
        x = (u + 0.5) - self.img_w / 2
        y = (v + 0.5) - self.img_h / 2
        z = torch.full_like(x, self.f, dtype=torch.float)
        unproj_pts = torch.stack([x, y, z], dim=-1)

        unproj_pts /= torch.norm(unproj_pts, dim=-1, keepdim=True)

        valid_mask = torch.full(unproj_pts.shape[:2], True, dtype=torch.bool)


        if with_rotation:
            unproj_pts = self.camcoord2worldcoord(unproj_pts)

        return unproj_pts, valid_mask


class EquirectProjection(CameraProjection):

    def __init__(
        self, img_h: int, img_w: int, R: Optional[torch.Tensor] = None
    ):

        super(EquirectProjection, self).__init__(img_h, img_w, R)

    def projection(
        self, world_pts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        world_pts = self.worldcoord2camcoord(world_pts)

        x, y, z = world_pts[..., 0], world_pts[..., 1], world_pts[..., 2]

        theta = torch.atan2(x, z)
        c = torch.sqrt(x * x + z * z)
        phi = torch.atan2(y, c)


        mapx = theta / np.pi
        mapy = phi / (np.pi / 2)
        proj_pts = torch.stack([mapx, mapy], dim=-1)


        valid_mask = torch.full(proj_pts.shape[:2], True, dtype=torch.bool)
        return proj_pts, valid_mask

    def unprojection(
        self, with_rotation: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_map, phi_map = self.get_theta_phi_map(self.img_h, self.img_w)
        unproj_pts = self.angle2sphere(theta_map, phi_map)

        valid_mask = torch.full(unproj_pts.shape[:2], True, dtype=torch.bool)

        if with_rotation:
            unproj_pts = self.camcoord2worldcoord(unproj_pts)
        return unproj_pts, valid_mask

    def get_theta_phi_map(
        self, img_h: int, img_w: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        phi, theta = torch.meshgrid(torch.arange(img_h), torch.arange(img_w))
        theta_map = (theta + 0.5) * 2 * np.pi / img_w - np.pi
        phi_map = (phi + 0.5) * np.pi / img_h - np.pi / 2
        return theta_map, phi_map

    def angle2sphere(
        self, theta_map: torch.Tensor, phi_map: torch.Tensor
    ) -> torch.Tensor:

        sin_theta = torch.sin(theta_map)
        cos_theta = torch.cos(theta_map)
        sin_phi = torch.sin(phi_map)
        cos_phi = torch.cos(phi_map)
        return torch.stack(
            [cos_phi * sin_theta, sin_phi, cos_phi * cos_theta], dim=-1
        )


class FisheyeProjection(CameraProjection):


    def __init__(
        self,
        img_h: int,
        img_w: int,
        fish_fov: float,
        cx: float,
        cy: float,
        fx: float,
        fy: float,
        xi: float,
        alpha: float,
        R: Optional[torch.Tensor] = None,
    ):

        super(FisheyeProjection, self).__init__(img_h, img_w, R)

        self.fish_fov = fish_fov
        fov_rad = self.fish_fov / 180 * np.pi
        self.fov_cos = np.cos(fov_rad / 2)
        self.fish_param = [cx, cy, fx, fy, xi, alpha]

    def projection(
        self, world_pts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        world_pts = self.worldcoord2camcoord(world_pts)


        cx, cy, fx, fy, xi, alpha = self.fish_param

        x, y, z = world_pts[..., 0], world_pts[..., 1], world_pts[..., 2]


        world_pts_fov_cos = z
        fov_mask = world_pts_fov_cos >= self.fov_cos


        x2 = x * x
        y2 = y * y
        z2 = z * z
        d1 = torch.sqrt(x2 + y2 + z2)
        zxi = xi * d1 + z
        d2 = torch.sqrt(x2 + y2 + zxi * zxi)

        div = alpha * d2 + (1 - alpha) * zxi
        u = fx * x / div + cx
        v = fy * y / div + cy


        mapx = 2 * u / self.img_w - 1.0
        mapy = 2 * v / self.img_h - 1.0
        proj_pts = torch.stack([mapx, mapy], dim=-1)


        if alpha <= 0.5:
            w1 = alpha / (1 - alpha)
        else:
            w1 = (1 - alpha) / alpha
        w2 = w1 + xi / np.sqrt(2 * w1 * xi + xi * xi + 1)
        valid_mask = z > -w2 * d1
        valid_mask *= fov_mask

        return proj_pts, valid_mask

    def unprojection(
        self, with_rotation: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        cx, cy, fx, fy, xi, alpha = self.fish_param


        v, u = torch.meshgrid(
            [torch.arange(self.img_h), torch.arange(self.img_w)]
        )
        mx = (u - cx) / fx
        my = (v - cy) / fy
        r2 = mx * mx + my * my
        mz = (1 - alpha * alpha * r2) / (
            alpha * torch.sqrt(1 - (2 * alpha - 1) * r2) + 1 - alpha
        )
        mz2 = mz * mz

        k1 = mz * xi + torch.sqrt(mz2 + (1 - xi * xi) * r2)
        k2 = mz2 + r2
        k = k1 / k2


        unproj_pts = k.unsqueeze(-1) * torch.stack([mx, my, mz], dim=-1)
        unproj_pts[..., 2] -= xi


        unproj_fov_cos = unproj_pts[..., 2]
        fov_mask = unproj_fov_cos >= self.fov_cos
        if alpha > 0.5:
            fov_mask *= r2 <= (1 / (2 * alpha - 1))

        if with_rotation:
            unproj_pts = self.camcoord2worldcoord(unproj_pts)

        return unproj_pts, fov_mask


class ProjectionConverter(nn.Module):


    def __init__(
        self,
        input_projections: Union[List[CameraProjection], CameraProjection],
        output_projections: Union[List[CameraProjection], CameraProjection],
    ):

        super(ProjectionConverter, self).__init__()

        if not isinstance(input_projections, list):
            input_projections = [input_projections]
        if not isinstance(output_projections, list):
            output_projections = [output_projections]

        self.input_models = input_projections
        self.output_models = output_projections
        self.input_len = len(self.input_models)
        self.output_len = len(self.output_models)


        input_size = self.input_models[0].size()
        for it in self.input_models:
            assert (
                input_size == it.size()
            ), "All input models must have the same image size"

        output_size = self.output_models[0].size()
        for it in self.output_models:
            assert (
                output_size == it.size()
            ), "All output models must have the same image size"


        self.input_zfactor = self.calculate_zfactor(self.input_models)

        self.output_zfactor = self.calculate_zfactor(
            self.output_models, inverse=True
        )


        self.grids = self.generate_grid()

        self._grids_cache = None

    def _generate_grid_one_output(
        self, output_model: CameraProjection
    ) -> torch.Tensor:

        world_pts, not_assigned_mask = output_model.unprojection()

        grids = []
        for input_model in self.input_models:
            grid, input_mask = input_model.projection(world_pts)

            input_mask *= not_assigned_mask

            grid[~input_mask] = 2

            not_assigned_mask *= ~input_mask
            grids.append(grid)
        grids = torch.stack(grids, dim=0)
        return grids

    def generate_grid(self) -> torch.Tensor:
        multi_output_grids = []
        for output_model in self.output_models:
            grids = self._generate_grid_one_output(output_model)
            multi_output_grids.append(grids.unsqueeze(1))
        multi_output_grids = torch.cat(multi_output_grids, dim=1)
        return multi_output_grids

    def _convert(self, batch: torch.Tensor) -> torch.Tensor:

        batch_size, ch, _H, _W = batch.shape
        out_h, out_w = self.output_models[0].size()
        if batch_size == 0 or batch_size % self.input_len != 0:
            raise ValueError(f"Batch size should be {self.input_len}x")
        output = torch.nn.functional.grid_sample(
            batch,
            self._grids_cache,
            align_corners=True,
            padding_mode="zeros",
        )
        output = output.view(
            batch_size // self.input_len,
            self.input_len,
            ch,
            out_h,
            out_w,
        ).sum(dim=1)
        return output

    def to_converted_tensor(self, batch: torch.Tensor) -> torch.Tensor:

        batch_size, ch, in_h, in_w = batch.size()

        out_h, out_w = self.output_models[0].size()


        if batch_size == 0 or batch_size % self.input_len != 0:
            raise ValueError(f"Batch size should be {self.input_len}x")

        num_input_set = batch_size // self.input_len


        self.grids = self.grids.to(batch.device)


        multi_out_batch = (
            batch.view(num_input_set, self.input_len, ch, in_h, in_w)
            .repeat(1, self.output_len, 1, 1, 1)
            .view(self.output_len * batch_size, ch, in_h, in_w)
        )


        if (
            self._grids_cache is None
            or self._grids_cache.size()[0] != multi_out_batch.size()[0]
        ):

            self._grids_cache = self.grids.repeat(
                num_input_set, 1, 1, 1, 1
            ).view(batch_size * self.output_len, out_h, out_w, 2)
        self._grids_cache = self._grids_cache.to(batch.device)

        return self._convert(multi_out_batch)

    def calculate_zfactor(
        self, projections: List[CameraProjection], inverse: bool = False
    ) -> Optional[torch.Tensor]:

        z_factors = []
        for cam in projections:
            if cam.depth_from == _DepthFrom.Z_VAL:
                pts_on_sphere, _ = cam.unprojection(with_rotation=False)
                zval_to_optcenter = 1 / pts_on_sphere[..., 2]
                z_factors.append(zval_to_optcenter.unsqueeze(0))
            else:
                all_one = torch.full(
                    (1, cam.img_h, cam.img_w), 1.0, dtype=torch.float
                )
                z_factors.append(all_one)
        z_factors = torch.stack(z_factors)

        if (z_factors == 1.0).all():

            return None
        else:
            if not inverse:

                return z_factors
            else:

                return 1 / z_factors

    def forward(
        self, batch: torch.Tensor, is_depth: bool = False
    ) -> torch.Tensor:


        if is_depth and self.input_zfactor is not None:
            input_b = batch.size()[0] // self.input_len
            self.input_zfactor = self.input_zfactor.to(batch.device)
            batch = batch * self.input_zfactor.repeat(input_b, 1, 1, 1)


        out = self.to_converted_tensor(batch)


        if is_depth and self.output_zfactor is not None:
            output_b = out.size()[0] // self.output_len
            self.output_zfactor = self.output_zfactor.to(batch.device)
            out = out * self.output_zfactor.repeat(output_b, 1, 1, 1)

        return out


def get_cubemap_projections(
    img_h: int = 256, img_w: int = 256
) -> List[CameraProjection]:

    rotations = [
        torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
        torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
        torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
        torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    ]

    projections = []
    for rot in rotations:
        cam = PerspectiveProjection(img_h, img_w, R=rot)
        projections.append(cam)
    return projections


class Cube2Equirect(ProjectionConverter):


    def __init__(self, equ_h: int, equ_w: int):



        input_projections = get_cubemap_projections()


        output_projection = EquirectProjection(equ_h, equ_w)
        super(Cube2Equirect, self).__init__(
            input_projections, output_projection
        )


class ProjectionTransformer(ObservationTransformer):


    def __init__(
        self,
        converter: ProjectionConverter,
        sensor_uuids: List[str],
        image_shape: Tuple[int, int],
        channels_last: bool = False,
        target_uuids: Optional[List[str]] = None,
        depth_key: str = "depth",
    ):

        super(ProjectionTransformer, self).__init__()
        num_sensors = len(sensor_uuids)
        assert (
            num_sensors % converter.input_len == 0 and num_sensors != 0
        ), f"{len(sensor_uuids)}: length of sensors is not a multiple of {converter.input_len}"

        assert (
            len(image_shape) == 2
        ), f"image_shape must be a tuple of (height, width), given: {image_shape}"
        self.sensor_uuids: List[str] = sensor_uuids
        self.img_shape: Tuple[int, int] = image_shape
        self.channels_last: bool = channels_last
        self.converter = converter
        if target_uuids is None:
            self.target_uuids: List[str] = self.sensor_uuids[::6]
        else:
            self.target_uuids: List[str] = target_uuids
        self.depth_key = depth_key

    def transform_observation_space(
        self,
        observation_space: spaces.Dict,
    ):

        for i, key in enumerate(self.target_uuids):
            assert (
                key in observation_space.spaces
            ), f"{key} not found in observation space: {observation_space.spaces}"
            h, w = get_image_height_width(
                observation_space.spaces[key], channels_last=True
            )
            in_len = self.converter.input_len
            logger.info(
                f"Overwrite sensor: {key} from size of ({h}, {w}) to image of"
                f" {self.img_shape} from sensors: {self.sensor_uuids[i*in_len:(i+1)*in_len]}"
            )
            if (h, w) != self.img_shape:
                observation_space.spaces[key] = overwrite_gym_box_shape(
                    observation_space.spaces[key], self.img_shape
                )
        return observation_space

    @torch.no_grad()
    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        for i, target_sensor_uuid in enumerate(self.target_uuids):

            in_len = self.converter.input_len
            in_sensor_uuids = self.sensor_uuids[i * in_len : (i + 1) * in_len]


            is_depth = any(self.depth_key in s for s in in_sensor_uuids)

            assert target_sensor_uuid in in_sensor_uuids
            sensor_obs = [observations[sensor] for sensor in in_sensor_uuids]
            target_obs = observations[target_sensor_uuid]
            sensor_dtype = target_obs.dtype

            imgs = torch.stack(sensor_obs, axis=1)
            imgs = torch.flatten(imgs, end_dim=1)
            if not self.channels_last:
                imgs = imgs.permute((0, 3, 1, 2))
            imgs = imgs.float()

            output = self.converter(imgs, is_depth=is_depth)

            output = output.to(dtype=sensor_dtype)
            if not self.channels_last:
                output = output.permute((0, 2, 3, 1))
            observations[target_sensor_uuid] = output
        return observations


@baseline_registry.register_obs_transformer()
class CubeMap2Equirect(ProjectionTransformer):


    def __init__(
        self,
        sensor_uuids: List[str],
        eq_shape: Tuple[int, int],
        channels_last: bool = False,
        target_uuids: Optional[List[str]] = None,
        depth_key: str = "depth",
    ):

        converter = Cube2Equirect(eq_shape[0], eq_shape[1])
        super(CubeMap2Equirect, self).__init__(
            converter,
            sensor_uuids,
            eq_shape,
            channels_last,
            target_uuids,
            depth_key,
        )

    @classmethod
    def from_config(cls, config):
        cube2eq_config = config.RL.POLICY.OBS_TRANSFORMS.CUBE2EQ
        if hasattr(cube2eq_config, "TARGET_UUIDS"):

            target_uuids = cube2eq_config.TARGET_UUIDS
        else:
            target_uuids = None
        return cls(
            cube2eq_config.SENSOR_UUIDS,
            eq_shape=(
                cube2eq_config.HEIGHT,
                cube2eq_config.WIDTH,
            ),
            target_uuids=target_uuids,
        )


class Cube2Fisheye(ProjectionConverter):

    def __init__(
        self,
        fish_h: int,
        fish_w: int,
        fish_fov: float,
        cx: float,
        cy: float,
        fx: float,
        fy: float,
        xi: float,
        alpha: float,
    ):



        input_projections = get_cubemap_projections()


        output_projection = FisheyeProjection(
            fish_h, fish_w, fish_fov, cx, cy, fx, fy, xi, alpha
        )
        super(Cube2Fisheye, self).__init__(
            input_projections, output_projection
        )


@baseline_registry.register_obs_transformer()
class CubeMap2Fisheye(ProjectionTransformer):


    def __init__(
        self,
        sensor_uuids: List[str],
        fish_shape: Tuple[int, int],
        fish_fov: float,
        fish_params: Tuple[float],
        channels_last: bool = False,
        target_uuids: Optional[List[str]] = None,
        depth_key: str = "depth",
    ):

        assert (
            len(fish_params) == 3
        ), "fish_params must have three parameters (f, xi, alpha)"


        fx = fish_params[0] * min(fish_shape)
        fy = fx
        cx = fish_shape[1] / 2
        cy = fish_shape[0] / 2
        xi = fish_params[1]
        alpha = fish_params[2]
        converter: ProjectionConverter = Cube2Fisheye(
            fish_shape[0], fish_shape[1], fish_fov, cx, cy, fx, fy, xi, alpha
        )

        super(CubeMap2Fisheye, self).__init__(
            converter,
            sensor_uuids,
            fish_shape,
            channels_last,
            target_uuids,
            depth_key,
        )

    @classmethod
    def from_config(cls, config):
        cube2fish_config = config.RL.POLICY.OBS_TRANSFORMS.CUBE2FISH
        if hasattr(cube2fish_config, "TARGET_UUIDS"):

            target_uuids = cube2fish_config.TARGET_UUIDS
        else:
            target_uuids = None
        return cls(
            cube2fish_config.SENSOR_UUIDS,
            fish_shape=(
                cube2fish_config.HEIGHT,
                cube2fish_config.WIDTH,
            ),
            fish_fov=cube2fish_config.FOV,
            fish_params=cube2fish_config.PARAMS,
            target_uuids=target_uuids,
        )


class Equirect2Cube(ProjectionConverter):


    def __init__(self, img_h: int, img_w: int):

        input_projection = EquirectProjection(256, 512)

        output_projections = get_cubemap_projections(img_h, img_w)
        super(Equirect2Cube, self).__init__(
            input_projection, output_projections
        )


@baseline_registry.register_obs_transformer()
class Equirect2CubeMap(ProjectionTransformer):


    def __init__(
        self,
        sensor_uuids: List[str],
        img_shape: Tuple[int, int],
        channels_last: bool = False,
        target_uuids: Optional[List[str]] = None,
        depth_key: str = "depth",
    ):


        converter = Equirect2Cube(img_shape[0], img_shape[1])
        super(Equirect2CubeMap, self).__init__(
            converter,
            sensor_uuids,
            img_shape,
            channels_last,
            target_uuids,
            depth_key,
        )

    @classmethod
    def from_config(cls, config):
        eq2cube_config = config.RL.POLICY.OBS_TRANSFORMS.EQ2CUBE

        if hasattr(eq2cube_config, "TARGET_UUIDS"):

            target_uuids = eq2cube_config.TARGET_UUIDS
        else:
            target_uuids = None
        return cls(
            eq2cube_config.SENSOR_UUIDS,
            img_shape=(
                eq2cube_config.HEIGHT,
                eq2cube_config.WIDTH,
            ),
            target_uuids=target_uuids,
        )


def get_active_obs_transforms(config: Config) -> List[ObservationTransformer]:
    active_obs_transforms = []
    if hasattr(config.RL.POLICY, "OBS_TRANSFORMS"):
        obs_transform_names = (
            config.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS
        )
        for obs_transform_name in obs_transform_names:
            obs_trans_cls = baseline_registry.get_obs_transformer(
                obs_transform_name
            )
            obs_transform = obs_trans_cls.from_config(config)
            active_obs_transforms.append(obs_transform)
    return active_obs_transforms


def apply_obs_transforms_batch(
    batch: Dict[str, torch.Tensor],
    obs_transforms: Iterable[ObservationTransformer],
) -> Dict[str, torch.Tensor]:
    for obs_transform in obs_transforms:
        batch = obs_transform(batch)
    return batch


def apply_obs_transforms_obs_space(
    obs_space: spaces.Dict, obs_transforms: Iterable[ObservationTransformer]
) -> spaces.Dict:
    for obs_transform in obs_transforms:
        obs_space = obs_transform.transform_observation_space(obs_space)
    return obs_space
