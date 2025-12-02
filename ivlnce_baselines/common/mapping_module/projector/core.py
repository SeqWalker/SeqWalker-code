import math

import torch


def _transform3D(pose, elevation, heading, device="cpu"):

    if isinstance(device, str):
        device = torch.device(device)

    theta_x = elevation
    cx = torch.cos(theta_x)
    sx = torch.sin(theta_x)

    theta_y = heading
    cy = torch.cos(theta_y)
    sy = torch.sin(theta_y)

    T = torch.zeros(pose.shape[0], 4, 4, device=device)
    T[:, 0, 0] = cy
    T[:, 0, 1] = sx * sy
    T[:, 0, 2] = cx * sy
    T[:, 0, 3] = pose[:, 0]

    T[:, 1, 0] = 0
    T[:, 1, 1] = cx
    T[:, 1, 2] = -sx
    T[:, 1, 3] = pose[:, 1]

    T[:, 2, 0] = -sy
    T[:, 2, 1] = cy * sx
    T[:, 2, 2] = cy * cx
    T[:, 2, 3] = pose[:, 2]

    T[:, 3, 3] = 1
    return T


class ProjectorUtils:
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
        device,
    ):

        self.vfov = vfov
        self.batch_size = batch_size
        self.fmh = feature_map_height
        self.fmw = feature_map_width
        self.output_height = output_height
        self.output_width = output_width
        self.gridcellsize = gridcellsize
        self.z_clip_threshold = z_clip_threshold
        self.world_shift_origin = world_shift_origin
        self.device = device

        self.x_scale, self.y_scale, self.ones = self.compute_scaling_params(
            batch_size, feature_map_height, feature_map_width
        )

    def compute_intrinsic_matrix(self, width, height, vfov):
        hfov = width / height * vfov
        f_x = width / (2.0 * math.tan(hfov / 2.0))
        f_y = height / (2.0 * math.tan(vfov / 2.0))
        cy = height / 2.0
        cx = width / 2.0
        K = torch.Tensor([[f_x, 0, cx], [0, f_y, cy], [0, 0, 1.0]])
        return K

    def compute_scaling_params(self, batch_size, image_height, image_width):

        K = self.compute_intrinsic_matrix(image_width, image_height, self.vfov)
        K = K.to(device=self.device).unsqueeze(0)
        K = K.expand(batch_size, 3, 3)

        fx = K[:, 0, 0].unsqueeze(1).unsqueeze(1)
        fy = K[:, 1, 1].unsqueeze(1).unsqueeze(1)
        cx = K[:, 0, 2].unsqueeze(1).unsqueeze(1)
        cy = K[:, 1, 2].unsqueeze(1).unsqueeze(1)

        x_rows = torch.arange(start=0, end=image_width, device=self.device)
        x_rows = x_rows.unsqueeze(0)
        x_rows = x_rows.repeat((image_height, 1))
        x_rows = x_rows.unsqueeze(0)
        x_rows = x_rows.repeat((batch_size, 1, 1))
        x_rows = x_rows.float()

        y_cols = torch.arange(start=0, end=image_height, device=self.device)
        y_cols = y_cols.unsqueeze(1)
        y_cols = y_cols.repeat((1, image_width))
        y_cols = y_cols.unsqueeze(0)
        y_cols = y_cols.repeat((batch_size, 1, 1))
        y_cols = y_cols.float()


        x_scale = (x_rows + 0.5 - cx) / fx
        y_scale = (y_cols + 0.5 - cy) / fy
        ones = (
            torch.ones(
                (batch_size, image_height, image_width), device=self.device
            )
            .unsqueeze(3)
            .float()
        )
        return x_scale, y_scale, ones

    def point_cloud(self, depth, depth_scaling=1.0):
        shape = depth.shape
        if (
            shape[0] == self.batch_size
            and shape[1] == self.fmh
            and shape[2] == self.fmw
        ):
            x_scale = self.x_scale
            y_scale = self.y_scale
            ones = self.ones
        else:
            x_scale, y_scale, ones = self.compute_scaling_params(
                shape[0], shape[1], shape[2]
            )
        z = depth / float(depth_scaling)
        x = z * x_scale
        y = z * y_scale

        xyz1 = torch.cat(
            (x.unsqueeze(3), y.unsqueeze(3), z.unsqueeze(3), ones), dim=3
        )
        return xyz1

    def transform_camera_to_world(self, xyz1, T):

        return torch.bmm(T, xyz1)

    def pixel_to_world_mapping(self, depth_img_array, T):

        xyz1 = self.point_cloud(depth_img_array)

        xyz1 = torch.reshape(
            xyz1, (xyz1.shape[0], xyz1.shape[1] * xyz1.shape[2], 4)
        )

        xyz1_t = torch.transpose(xyz1, 1, 2)

        xyz1_w = self.transform_camera_to_world(xyz1_t, T)


        world_xyz = xyz1_w.transpose(1, 2)[:, :, :3]


        world_xyz -= self.world_shift_origin


        pixel_to_world = torch.reshape(
            world_xyz,
            (
                (
                    depth_img_array.shape[0],
                    depth_img_array.shape[1],
                    depth_img_array.shape[2],
                    3,
                )
            ),
        )

        return pixel_to_world

    def discretize_point_cloud(self, point_cloud, camera_height):

        pixels_in_map = (
            (point_cloud[:, :, :, [0, 2]]) / self.gridcellsize
        ).round()

        outside_map_indices = (
            (pixels_in_map[:, :, :, 0] >= self.output_width)
            + (pixels_in_map[:, :, :, 1] >= self.output_height)
            + (pixels_in_map[:, :, :, 0] < 0)
            + (pixels_in_map[:, :, :, 1] < 0)
        )

        camera_y = (
            camera_height.unsqueeze(1)
            .unsqueeze(1)
            .repeat(1, pixels_in_map.shape[1], pixels_in_map.shape[2])
        )

        above_threshold_z_indices = point_cloud[:, :, :, 1] > (
            camera_y + self.z_clip_threshold
        )

        mask_outliers = outside_map_indices + above_threshold_z_indices

        return pixels_in_map.long(), mask_outliers
