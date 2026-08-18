"""
FPN-based occluded face recognition model for 112x112 input.
Adapted from fpn.py (which was 112x96).

Key change: spatial dims after backbone are 7x7 (was 7x6).
All 7*6 references become 7*7.
"""

import torch
import torch.nn as nn


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)
        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class BlockIR(nn.Module):
    def __init__(self, inplanes, planes, stride, dim_match):
        super(BlockIR, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        if dim_match:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


# ==================== Primary model: 3D mask (512-channel spatial mask) ====================

class LResNet_Occ_112(nn.Module):
    """
    FROM model for 112x112 input.
    Feature map after backbone: 512 x 7 x 7 (was 512 x 7 x 6 for 112x96).
    """
    def __init__(self, block, layers, filter_list, is_gray=False):
        self.inplanes = 64
        super(LResNet_Occ_112, self).__init__()

        if is_gray:
            self.conv1 = nn.Conv2d(1, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_list[0])
        self.prelu1 = nn.PReLU(filter_list[0])
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)

        # Mask branch (FPN-based)
        # After FPN P3: 256 channels at 14x14 spatial.
        # After reduces: 64 channels at 4x4 (approximately).
        # But the mask is applied to fmap (512 x 7 x 7), so mask output should be 512 x 7 x 7.
        self.mask = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(256),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, filter_list[4], kernel_size=3, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.fpn = PyramidFeatures(filter_list[2], filter_list[3], filter_list[4])

        # Regress: predict which grid rectangle is occluded
        # Input: flattened mask (512 * 7 * 7 = 25088)
        self.regress = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 7 * 7),
            nn.Dropout(p=0.5),
            nn.Linear(filter_list[4] * 7 * 7, filter_list[5], bias=False),
            nn.BatchNorm1d(filter_list[5]),
        )

        # FC: produce 512-dim embedding from feature map
        self.fc = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 7 * 7),
            nn.Dropout(p=0.5),
            nn.Linear(filter_list[4] * 7 * 7, 512),
            nn.BatchNorm1d(512),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        layers.append(block(inplanes, planes, stride, False))
        for i in range(1, blocks):
            layers.append(block(planes, planes, stride=1, dim_match=True))
        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        fmap = self.layer4(x3)

        # Generate mask via FPN
        if not isinstance(mask, torch.Tensor):
            features = self.fpn([x2, x3, fmap])
            mask = self.mask(features[0])

        # Predict occlusion grid label
        vec = self.regress(mask.view(mask.size(0), -1))

        # Masked feature embedding
        fmap_mask = fmap * mask
        fc_mask = self.fc(fmap_mask.view(fmap_mask.size(0), -1))

        # Clean feature embedding
        fc = self.fc(fmap.view(fmap.size(0), -1))

        return fc_mask, mask, vec, fc

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


def LResNet50E_IR_Occ_112(is_gray=False, num_mask=101):
    """Factory for 112x112 input FROM model."""
    filter_list = [64, 64, 128, 256, 512, num_mask]
    layers = [3, 4, 14, 3]
    return LResNet_Occ_112(BlockIR, layers, filter_list, is_gray)


# ==================== FC mask variant (mask applied after FC) ====================

class LResNet_Occ_FC_112(nn.Module):
    """FC-level mask variant for 112x112 input."""
    def __init__(self, block, layers, filter_list, is_gray=False):
        self.inplanes = 64
        super(LResNet_Occ_FC_112, self).__init__()

        if is_gray:
            self.conv1 = nn.Conv2d(1, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_list[0])
        self.prelu1 = nn.PReLU(filter_list[0])
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)

        # FPN reduces P3 (14x14) -> mask (7x7) -> flatten -> 512-d mask vector
        self.mask = nn.Sequential(
            nn.BatchNorm1d(64 * 7 * 7),
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.Sigmoid(),
        )
        self.fpn = PyramidFeatures(filter_list[2], filter_list[3], filter_list[4])

        self.reduces = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(256),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(64),
            nn.BatchNorm2d(64)
        )

        self.regress = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, filter_list[5], bias=False),
            nn.BatchNorm1d(filter_list[5]),
        )

        self.fc = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 7 * 7),
            nn.Dropout(p=0.5),
            nn.Linear(filter_list[4] * 7 * 7, 512),
            nn.BatchNorm1d(512),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        layers.append(block(inplanes, planes, stride, False))
        for i in range(1, blocks):
            layers.append(block(planes, planes, stride=1, dim_match=True))
        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        fmap = self.layer4(x3)

        if not isinstance(mask, torch.Tensor):
            features = self.fpn([x2, x3, fmap])
            fmap_reduce = self.reduces(features[0])
            mask = self.mask(fmap_reduce.view(fmap_reduce.size(0), -1))

        vec = self.regress(mask)
        fc = self.fc(fmap.view(fmap.size(0), -1))
        fc_mask = fc * mask

        return fc_mask, mask, vec, fc

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


def LResNet50E_IR_Occ_FC_112(is_gray=False, num_mask=101):
    filter_list = [64, 64, 128, 256, 512, num_mask]
    layers = [3, 4, 14, 3]
    return LResNet_Occ_FC_112(BlockIR, layers, filter_list, is_gray)


# ==================== 2D mask variant (single-channel spatial mask) ====================

class LResNet_Occ_2D_112(nn.Module):
    """2D mask variant for 112x112 input."""
    def __init__(self, block, layers, filter_list, is_gray=False):
        self.inplanes = 64
        self.filter_list = filter_list
        super(LResNet_Occ_2D_112, self).__init__()

        if is_gray:
            self.conv1 = nn.Conv2d(1, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_list[0])
        self.prelu1 = nn.PReLU(filter_list[0])
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)

        self.mask = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(256),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, filter_list[4], kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(filter_list[4]),
            nn.BatchNorm2d(filter_list[4]),
            nn.Conv2d(filter_list[4], 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.fpn = PyramidFeatures(filter_list[2], filter_list[3], filter_list[4])

        self.regress = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 7 * 7),
            nn.Dropout(p=0.5),
            nn.Linear(filter_list[4] * 7 * 7, filter_list[5], bias=False),
            nn.BatchNorm1d(filter_list[5]),
        )

        self.fc = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 7 * 7),
            nn.Dropout(p=0.5),
            nn.Linear(filter_list[4] * 7 * 7, 512),
            nn.BatchNorm1d(512),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        layers.append(block(inplanes, planes, stride, False))
        for i in range(1, blocks):
            layers.append(block(planes, planes, stride=1, dim_match=True))
        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        fmap = self.layer4(x3)

        if not isinstance(mask, torch.Tensor):
            features = self.fpn([x2, x3, fmap])
            mask = self.mask(features[0])
            mask = mask.repeat(1, self.filter_list[4], 1, 1)

        vec = self.regress(mask.view(mask.size(0), -1))
        fmap_mask = fmap * mask
        fc_mask = self.fc(fmap_mask.view(fmap_mask.size(0), -1))
        fc = self.fc(fmap.view(fmap.size(0), -1))

        return fc_mask, mask, vec, fc

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


def LResNet50E_IR_Occ_2D_112(is_gray=False, num_mask=101):
    filter_list = [64, 64, 128, 256, 512, num_mask]
    layers = [3, 4, 14, 3]
    return LResNet_Occ_2D_112(BlockIR, layers, filter_list, is_gray)
