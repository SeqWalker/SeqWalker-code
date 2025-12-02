from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class CoTAttention(nn.Module):
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=self.kernel_size, padding=kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU())
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim))
        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1))

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)
        v = self.value_embed(x).view(bs, c, -1)
        y = torch.cat([k1, x], dim=1)
        att = self.attention_embed(y)
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)
        return k1 + k2


class CBRAWithAttention(nn.Module):


    def __init__(self, in_channels, out_channels, attention_dim=512, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=7, padding="same"
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
        )
        self.attention = CoTAttention(dim=out_channels, kernel_size=kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return self.attention(x)


class SemanticMapEncoder(nn.Module):


    def __init__(
            self,
            observation_space,
            num_semantic_classes: int = 13,
            ch: int = 32,
            last_ch_mult: int = 8,
            trainable: bool = True,
            from_pretrained: bool = False,
            checkpoint: Optional[str] = None,
    ):
        super().__init__()
        for k in ["occupancy_map", "semantic_map"]:
            if k not in observation_space.spaces:
                raise ValueError(f"key `{k}` expected in observation space.")

        self._map_dimensions = observation_space.spaces["occupancy_map"].shape
        self._num_semantic_classes = num_semantic_classes
        self.last_ch_mult = last_ch_mult

        self._ch = ch
        self.cnn = nn.Sequential(
            CBRAWithAttention(14, ch),
            CBRAWithAttention(ch, ch * 2),
            CBRAWithAttention(ch * 2, ch * 4),
            CBRAWithAttention(ch * 4, ch * last_ch_mult),
        )

        if from_pretrained:
            ckpt = torch.load(checkpoint, map_location="cpu")["state_dict"]
            prefix = "encoder.cnn."
            state_dict = {
                k[len(prefix):]: v
                for k, v in ckpt.items()
                if k.startswith(prefix)
            }
            self.cnn.load_state_dict(state_dict)

        for param in self.cnn.parameters():
            param.requires_grad_(trainable)

        if not trainable:
            self.eval()

    def forward(self, occupancy_map, semantic_map):
        combined_map = torch.cat([occupancy_map, semantic_map], dim=1)
        return self.cnn(combined_map)

    @property
    def output_shape(self):
        nrows = self._map_dimensions[0]
        ncols = self._map_dimensions[1]
        div = 2 ** 4
        return (self._ch * self.last_ch_mult, nrows // div, ncols // div)

    def generate_map_features(self, observations):
        occupancy = observations["occupancy_map"].unsqueeze(1)
        semantic = observations["semantic_map"].long()
        semantic = F.one_hot(semantic, self._num_semantic_classes)
        semantic = semantic.permute(0, 3, 1, 2)
        return torch.cat((occupancy, semantic), 1).to(dtype=torch.float)

    def forward(self, observations):
        for k in ["occupancy_map", "semantic_map"]:
            if k not in observations:
                raise ValueError(f"Observation `{k}` is missing.")

        return self.cnn(self.generate_map_features(observations))
