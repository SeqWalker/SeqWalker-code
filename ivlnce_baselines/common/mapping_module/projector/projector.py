import torch

from .core import ProjectorUtils


class Projector(ProjectorUtils):


    def __init__(
        self,
        vfov,
        batch_size,
        feature_map_height,
        feature_map_width,
        output_height,
        output_width,
        gridcellsize,
        world_shift_origin,
        z_clip_threshold,
        device="cuda",
    ):


        if isinstance(device, str):
            device = torch.device(device)

        ProjectorUtils.__init__(
            self,
            vfov,
            batch_size,
            feature_map_height,
            feature_map_width,
            output_height,
            output_width,
            gridcellsize,
            world_shift_origin,
            z_clip_threshold,
            device,
        )

        self.vfov = vfov
        self.batch_size = batch_size
        self.fmh = feature_map_height
        self.fmw = feature_map_width
        self.output_height = output_height
        self.output_width = output_width
        self.gridcellsize = gridcellsize
        self.z_clip_threshold = z_clip_threshold
        self.device = device

    def forward(self, depth, T, obs_per_map=1, return_heights=False):


        assert depth.shape[2] == self.fmh
        assert depth.shape[3] == self.fmw

        depth = depth[:, 0, :, :]

        no_depth_mask = torch.logical_and(
            depth >= (0.1 * 10), depth <= (0.9 * 10)
        )


        point_cloud = self.pixel_to_world_mapping(depth, T)

        camera_height = T[:, 1, 3]

        projection_indices_2D, outliers = self.discretize_point_cloud(
            point_cloud, camera_height
        )

        outliers = no_depth_mask + outliers

        if return_heights:
            return projection_indices_2D, outliers, point_cloud[..., 1]
        else:
            return projection_indices_2D, outliers
