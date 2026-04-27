import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16
from module.MDconv import MDconv
from module.ADatt import ADatt


class ADattWrapper(nn.Module):
    """
    Wrapper for ADatt to work with (B, C, H, W) format
    """

    def __init__(self, dim, num_patches=None, num_heads=8, **kwargs):
        super().__init__()
        if num_patches is None:
            num_patches = 4096

        self.attention = ADatt(
            dim=dim,
            num_patches=num_patches,
            num_heads=num_heads,
            **kwargs
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) format tensor
        Returns:
            out: (B, C, H, W) format tensor
        """
        B, C, H, W = x.shape

        # Reshape to (B, H*W, C)
        x_reshaped = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Apply attention
        out = self.attention(x_reshaped, H, W)  # (B, H*W, C)

        # Reshape back to (B, C, H, W)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return out


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class MDADnet(nn.Module):

    def __init__(self, num_classes=21, pretrained=False, backbone='vgg16'):
        super(MDADnet, self).__init__()
        if backbone == 'vgg16':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        if backbone == 'vgg16':
            # VGG16的通道数
            self.conv1 = MDconv(64)  # feat1: 64 channels
            self.conv2 = MDconv(128)  # feat2: 128 channels
            self.conv3 = MDconv(256)  # feat3: 256 channels
            self.conv4 = MDconv(512)  # feat4: 512 channels
        elif backbone == "resnet50":
            # ResNet50的通道数
            self.conv1 = MDconv(64)  # feat1: 64 channels
            self.conv2 = MDconv(256)  # feat2: 256 channels
            self.conv3 = MDconv(512)  # feat3: 512 channels
            self.conv4 = MDconv(1024)  # feat4: 1024 channels

        # 对于512x512输入图像，特征图尺寸：
        # VGG16: feat1(512x512), feat2(256x256), feat3(128x128), feat4(64x64)
        # ResNet50: feat1(256x256), feat2(128x128), feat3(64x64), feat4(32x32)
        if backbone == 'vgg16':
            # VGG16的通道数和特征图尺寸
            self.attn1 = ADattWrapper(dim=64, num_patches=262144,
                                      num_heads=8)  # feat1: 64 channels, 512x512
            self.attn2 = ADattWrapper(dim=128, num_patches=65536,
                                      num_heads=8)  # feat2: 128 channels, 256x256
            self.attn3 = ADattWrapper(dim=256, num_patches=16384,
                                      num_heads=8)  # feat3: 256 channels, 128x128
            self.attn4 = ADattWrapper(dim=512, num_patches=4096,
                                      num_heads=8)  # feat4: 512 channels, 64x64
        elif backbone == "resnet50":
            # ResNet50的通道数和特征图尺寸
            self.attn1 = ADattWrapper(dim=64, num_patches=65536,
                                      num_heads=8)  # feat1: 64 channels, 256x256
            self.attn2 = ADattWrapper(dim=256, num_patches=16384,
                                      num_heads=8)  # feat2: 256 channels, 128x128
            self.attn3 = ADattWrapper(dim=512, num_patches=4096,
                                      num_heads=8)  # feat3: 512 channels, 64x64
            self.attn4 = ADattWrapper(dim=1024, num_patches=1024,
                                      num_heads=8)  # feat4: 1024 channels, 32x32

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg16":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        feat1 = self.conv1(feat1)
        feat2 = self.conv2(feat2)
        feat3 = self.conv3(feat3)
        feat4 = self.conv4(feat4)

        feat1 = self.attn1(feat1)
        feat2 = self.attn2(feat2)
        feat3 = self.attn3(feat3)
        feat4 = self.attn4(feat4)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True