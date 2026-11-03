"""
attention_modules.py - CBAM (and SE / Coordinate Attention) for YOLOv8.

IMPORTANT
---------
Your original file `Drone-Detection-1/ultralytics/nn/modules/cbam.py` was NOT
in the GitHub repository (the `Drone-Detection-1` folder is an empty
git-submodule pointer to commit 12c69f7, which was never pushed).
This file is a clean re-implementation that follows:
  * the paper (channel attention -> spatial attention, r = 16, 7x7 kernel),
  * the public API your scripts use (`apply_attention_to_yolo`,
    `AttentionLayerWrapper`, `.attn_block.gamma`), as seen in
    `colab_ablation.py` and `scratch/verify_cbam_refactor.py`.

If your real cbam.py differs (for example, MLP biases), parameter counts can
differ by a few hundred.  Re-run a3_param_counts.py with your file to confirm.

How the attention is attached (plain words)
-------------------------------------------
YOLOv8 stores its layers in a list `model.model[0..22]`.  We take layer 12
(and 15), put it inside a small "wrapper" box, and put the box back in the
same position of the list.  The wrapper first runs the original layer and
then runs the attention block on its output:

        output = layer(x)                       # original C2f layer
        refined = output + gamma * (CBAM(output) - output)

`gamma` is ONE learnable number that starts at 0, so on step 0 the network
behaves exactly like the pretrained YOLOv8x.  Training then learns how much
of the attention to mix in.  (This is module *replacement*, not a PyTorch
`register_forward_hook`.)
"""
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# 1. The two halves of CBAM
# ----------------------------------------------------------------------
class ChannelAttention(nn.Module):
    """'WHICH channels matter?'  Paper Eq. (3) with the ReLU (delta) included:

        Mc(F) = sigmoid( W1(ReLU(W0(AvgPool(F)))) + W1(ReLU(W0(MaxPool(F)))) )
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        # Shared MLP written with 1x1 convolutions (same maths as Linear layers).
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),  # W0
            nn.ReLU(inplace=True),                                     # delta
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),  # W1
        )

    def forward(self, x):
        avg = self.mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        mx = self.mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        return x * torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """'WHERE in the image?'  Paper Eq. (5): 7x7 conv on [avg; max] maps."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.amax(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Channel attention followed by spatial attention (Woo et al., 2018)."""

    def __init__(self, channels, reduction=16, kernel_size=7, use_channel=True, use_spatial=True):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction) if use_channel else nn.Identity()
        self.spatial = SpatialAttention(kernel_size) if use_spatial else nn.Identity()

    def forward(self, x):
        return self.spatial(self.channel(x))


# ----------------------------------------------------------------------
# 2. Two other attention blocks for the comparison asked by Reviewer 1
# ----------------------------------------------------------------------
class SqueezeExcitation(nn.Module):
    """SE block (Hu et al., 2018): channel attention from average pooling only."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(torch.mean(x, dim=(2, 3), keepdim=True))


class CoordinateAttention(nn.Module):
    """Coordinate Attention (Hou et al., 2021): pools along height and width separately."""

    def __init__(self, channels, reduction=32):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.bn = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(hidden, channels, 1)
        self.conv_w = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        _, _, h, w = x.shape
        pooled_h = torch.mean(x, dim=3, keepdim=True)                       # N,C,H,1
        pooled_w = torch.mean(x, dim=2, keepdim=True).permute(0, 1, 3, 2)   # N,C,W,1
        y = self.act(self.bn(self.conv1(torch.cat([pooled_h, pooled_w], dim=2))))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w.permute(0, 1, 3, 2)))
        return x * a_h * a_w


def build_attention(mode, channels):
    """Return the attention block for a mode name."""
    if mode == "cbam":
        return CBAM(channels)
    if mode == "channel":
        return CBAM(channels, use_spatial=False)
    if mode == "spatial":
        return CBAM(channels, use_channel=False)
    if mode == "se":
        return SqueezeExcitation(channels)
    if mode == "coord":
        return CoordinateAttention(channels)
    raise ValueError(f"Unknown attention mode: {mode}")


# ----------------------------------------------------------------------
# 3. Wrapping a YOLOv8 layer
# ----------------------------------------------------------------------
class ResidualAttentionBlock(nn.Module):
    """refined = x + gamma * (attention(x) - x), with gamma starting at 0."""

    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.gamma * (self.attention(x) - x)


class AttentionLayerWrapper(nn.Module):
    """Runs the original YOLO layer, then the attention block on its output."""

    def __init__(self, layer, attn_block):
        super().__init__()
        self.layer = layer
        self.attn_block = attn_block
        # Ultralytics' forward loop needs these bookkeeping attributes.
        for name in ("f", "i", "type", "np"):
            setattr(self, name, getattr(layer, name, None))

    def forward(self, x):
        return self.attn_block(self.layer(x))


def output_channels(layer):
    """Number of output channels of an Ultralytics C2f layer (its last conv)."""
    return layer.cv2.conv.out_channels


def apply_attention_to_yolo(model, mode="cbam", target_layers=(12, 15)):
    """Wrap `target_layers` of a YOLO / DetectionModel with attention.

    `model` can be an `ultralytics.YOLO` object or its inner `DetectionModel`.
    Returns the inner DetectionModel.
    """
    net = model.model if hasattr(model, "predictor") else model  # YOLO -> DetectionModel
    if mode in (None, "none"):
        return net
    for idx in target_layers:
        layer = net.model[idx]
        if isinstance(layer, AttentionLayerWrapper):
            continue  # already wrapped
        block = ResidualAttentionBlock(build_attention(mode, output_channels(layer)))
        block.to(next(layer.parameters()).device)
        net.model[idx] = AttentionLayerWrapper(layer, block)
    return net


# ----------------------------------------------------------------------
# 4. A trainer that keeps the attention during model.train()
# ----------------------------------------------------------------------
def make_attention_trainer(mode, target_layers=(12, 15)):
    """Return a DetectionTrainer subclass that adds attention after loading weights.

    Why this is needed: stock `YOLO.train()` REBUILDS the network from its YAML
    file (`trainer.get_model(cfg=model.yaml, weights=model)`), which silently
    throws away any wrapper added beforehand (verified in a1_attachment_check.py).
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    class AttentionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            net = super().get_model(cfg=cfg, weights=weights, verbose=verbose)  # pretrained weights loaded first
            return apply_attention_to_yolo(net, mode=mode, target_layers=target_layers)

    AttentionTrainer.__name__ = f"AttentionTrainer_{mode}_{'_'.join(map(str, target_layers))}"
    return AttentionTrainer
