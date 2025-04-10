import torch
from torch import nn


from transformers import SamConfig


class ColorAttentionAdapter(nn.Module):
    def __init__(
        self, embedding_dim, mlp_ratio=0.25, act_layer=nn.GELU, change=False
    ) -> None:
        super().__init__()
        hidden_dim = int(embedding_dim * mlp_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.act = act_layer()
        self.fc1 = nn.Conv2d(embedding_dim, hidden_dim, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_dim, embedding_dim, 1, bias=False)
        self.Sigmoid = nn.Sigmoid()
        self.change_channel = change

    def forward(self, x):
        if self.change_channel:
            x = x.permute(0, 3, 1, 2).contiguous()
            avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
            return self.Sigmoid(avg_out + max_out).view(x.shape[0], 1, 1, -1)
        else:
            avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
            return self.Sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)


class Adapter(nn.Module):
    def __init__(
        self, embedding_dim, mlp_ratio=0.25, act_layer=nn.GELU, skip=False, scale=1
    ):
        super().__init__()
        self.skip = skip
        self.scale = scale
        hidden_dim = int(embedding_dim * mlp_ratio)
        self.act = act_layer()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        out = self.fc2(self.act(self.fc1(x)))
        if self.skip:
            out = out + x
        return self.scale * out


class ViTBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        use_color_adapter=True,
        use_space_adapter=True,
        use_mlp_adapter=True,
    ):
        super().__init__()

        if use_color_adapter:
            self.color_adapter = ColorAttentionAdapter(embed_dim, change=True)
        if use_space_adapter:
            self.space_adapter = Adapter(embed_dim, skip=True)
        if use_mlp_adapter:
            self.mlp_adapter = Adapter(embed_dim, scale=0.5)


class ViTAdapters(nn.Module):

    def __init__(
        self,
        adapter_layer,
        embed_dim,
        use_color_adapter=True,
        use_space_adapter=True,
        use_mlp_adapter=True,
    ):
        super().__init__()

        self.adapter_layer = adapter_layer
        for idx in adapter_layer:
            self.add_module(
                f"adapter_{idx}",
                ViTBlock(
                    embed_dim, use_color_adapter, use_space_adapter, use_mlp_adapter
                ),
            )


class MultiScaleConv(nn.Module):
    def __init__(self, input_dim, output_dim, act_layer=nn.GELU) -> None:
        super().__init__()
        self.act = act_layer()
        self.conv1 = nn.Conv2d(input_dim, output_dim, 1)
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.conv3 = nn.Conv2d(output_dim, output_dim, 3, padding=1)
        self.conv5 = nn.Conv2d(output_dim, output_dim, 5, padding=2)
        self.conv7 = nn.Conv2d(output_dim, output_dim, 7, padding=3)
        self.bn2 = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.conv3(x) + self.conv5(x) + self.conv7(x)
        return self.act(self.bn2(x))
