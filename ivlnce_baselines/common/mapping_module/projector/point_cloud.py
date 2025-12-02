import numpy as np
import torch

from .core import ProjectorUtils


class PointCloud(ProjectorUtils):


    def __init__(
        self,
        camera_parameters,
        batch_size,
        world_shift_origin,
        device="cuda",
    ):

        if isinstance(device, str):
            device = torch.device(device)

        ProjectorUtils.__init__(
            self,
            camera_parameters.vertical_fov_radians,
            batch_size,
            camera_parameters.features_spatial_dimensions[0],
            camera_parameters.features_spatial_dimensions[1],
            1,
            1,
            1,
            world_shift_origin,
            camera_parameters.height_clip,
            device,
        )

        self.batch_size = batch_size

        self.vfov = camera_parameters.vertical_fov_radians
        self.fmh = camera_parameters.features_spatial_dimensions[0]
        self.fmw = camera_parameters.features_spatial_dimensions[1]
        self.z_clip_threshold = camera_parameters.height_clip

        self.world_shift_origin = world_shift_origin
        self.device = device

    def egocentric_depth_to_point_cloud(self, depth, T, obs_per_map=1):


        assert depth.shape[2] == self.fmh
        assert depth.shape[3] == self.fmw

        depth = depth[:, 0, :, :]


        point_cloud = self.pixel_to_world_mapping(depth, T)

        return point_cloud.permute(0, 3, 1, 2)


if __name__ == "__main__":
    from core import _transform3D
    from habitat import get_config
    from habitat.sims import make_sim
    from scipy.spatial.transform import Rotation as R

    house = "17DRP5sb8fy"
    scene = "../data/mp3d/{}/{}.glb".format(house, house)
    config = get_config()
    config.defrost()
    config.SIMULATOR.SCENE = scene
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR"]
    config.freeze()

    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)

    sim.reset()

    vfov = 67.5
    world_shift = torch.FloatTensor([0, 0, 0])
    projector = PointCloud(
        vfov,
        1,
        480,
        640,
        world_shift,
        0.5,
        device=torch.device("cpu"),
    )

    ags = sim.get_agent_state()
    pos = ags.sensor_states["depth"].position
    rot = ags.sensor_states["depth"].rotation
    rot = np.array([rot.x, rot.y, rot.z, rot.w])
    r = R.from_quat(rot)
    elevation, heading, bank = r.as_rotvec()

    xyzhe = np.array(
        [[pos[0], pos[1], pos[2], heading, elevation + np.pi]]
    )
    xyzhe = torch.FloatTensor(xyzhe)
    T = _transform3D(xyzhe)


    depth = sim.render(mode="depth")
    depth = depth[:, :, 0]
    depth = depth.astype(np.float32)
    depth *= 10.0
    depth_var = torch.FloatTensor(depth).unsqueeze(0).unsqueeze(0)

    pc, mask_outliers = projector.forward(depth_var, T)

    pc = pc[~mask_outliers]

    import matplotlib.pyplot as plt
    import numpy as np

    pc = pc.numpy()

    pc = pc[0:-1:100, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    ax.scatter(x, y, z)

    plt.show()
