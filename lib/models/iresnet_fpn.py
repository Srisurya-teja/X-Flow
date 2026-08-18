"""
iResNet50 (InsightFace-style improved ResNet) backbone with FROM's FPN
occlusion-mask branch grafted on.

Input: 3x112x112.  Feature map after layer4: 512 x 7 x 7 (same spatial size the
FROM mask branch expects), so the FPN / mask / regress / fc heads from fpn_112
are reused unchanged.

    forward(x, mask=None) -> (fc_mask, mask, vec, fc)

matching the other FROM backbones, so it drops into train_distributed.py's
Clean / Occ / Mask training modes without any other change.
"""

import torch
import torch.nn as nn

from lib.models.fpn_112 import PyramidFeatures


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    """InsightFace iResNet improved basic block (BN-Conv-BN-PReLU-Conv-BN + residual)."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNetOccFPN(nn.Module):
    """iResNet backbone + FROM FPN mask branch. Returns (fc_mask, mask, vec, fc)."""
    fc_scale = 7 * 7

    def __init__(self, block, layers, num_mask=226, dropout=0.5):
        super().__init__()
        self.inplanes = 64

        # ---- iResNet stem + stages (112 -> 56 -> 28 -> 14 -> 7) ----
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # ---- FROM mask branch: FPN over (layer2, layer3, layer4) -> 512x7x7 sigmoid mask ----
        self.fpn = PyramidFeatures(128, 256, 512)
        self.mask = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(256),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )

        # occlusion grid classifier (mask-prediction loss target in Mask mode)
        self.regress = nn.Sequential(
            nn.BatchNorm1d(512 * self.fc_scale),
            nn.Dropout(p=dropout),
            nn.Linear(512 * self.fc_scale, num_mask, bias=False),
            nn.BatchNorm1d(num_mask),
        )

        # 512-d embedding head (shared for masked and clean feature maps)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(512 * self.fc_scale),
            nn.Dropout(p=dropout),
            nn.Linear(512 * self.fc_scale, 512),
            nn.BatchNorm1d(512),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes, eps=1e-05),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        fmap = self.layer4(x3)

        # Predict the occlusion mask (unless a fixed mask is supplied at test time).
        if not isinstance(mask, torch.Tensor):
            feats = self.fpn([x2, x3, fmap])
            mask = self.mask(feats[0])

        vec = self.regress(mask.view(mask.size(0), -1))
        fmap_mask = fmap * mask
        fc_mask = self.fc(fmap_mask.view(fmap_mask.size(0), -1))
        fc = self.fc(fmap.view(fmap.size(0), -1))
        return fc_mask, mask, vec, fc


def iresnet50_occ(num_mask=226, **kwargs):
    """iResNet50 (layers [3,4,14,3]) + FROM FPN mask branch."""
    return IResNetOccFPN(IBasicBlock, [3, 4, 14, 3], num_mask=num_mask, **kwargs)
