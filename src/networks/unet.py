# +
#https://github.com/hiepph/unet-lightning/blob/bde5a5d0f7dacb83abe219447fcd2b988f1dbd1c/Unet.py#L28
# -

import torch
import torch.nn as nn
import torch.nn.functional as F
from blurpool import BlurPool
from bisenet import BiSeNet


class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x, skip=None):
        if skip is not None:
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class MaxBlurPool2D(nn.Module):
    def __init__(self, c_out):
        super(MaxBlurPool2D, self).__init__()
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1),
            BlurPool(c_out, stride=2)
        )
    def forward(self, x):
        return self.pool(x)


class BlurUpsample(nn.Module):
    def __init__(self, c_in, c_out, bilinear=True):
        super(BlurUpsample, self).__init__()
        
        if bilinear is True:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.ConvTranspose2d(in_channels=c_in//2, out_channels=c_out//2, kernel_size=2, stride=2)
        
        self.up = nn.Sequential(
            self.upsample,
            BlurPool(c_out, stride=1)    
        )
    
    def forward(self, x):
        return self.up(x)


def down(c_in, c_out):
    return nn.Sequential(
            MaxBlurPool2D(c_in),
            DoubleConv(c_in, c_out)
    )


def up(c_in, c_out):
    return nn.Sequential(
            BlurUpsample(c_in//2, c_in//2),
            DoubleConv(c_in, c_out)
    )


# +
class Unet(nn.Module):
    def __init__(self, c_in=5):
        super(Unet, self).__init__()
        self.n_channels = c_in

        self.in_conv = DoubleConv(self.n_channels, 64)
        self.down1 = down(c_in=64, c_out=128)
        self.down2 = down(c_in=128, c_out=256)
        self.down3 = down(c_in=256, c_out=512)
        self.down4 = down(c_in=512, c_out=512)
        
        self.up1 = BlurUpsample(1024, 512)
        self.up1_conv = DoubleConv(1024, 256)
        self.up2 = BlurUpsample(512, 256)
        self.up2_conv = DoubleConv(512, 128)
        self.up3 = BlurUpsample(256, 128)
        self.up3_conv = DoubleConv(256, 64)
        self.up4 = BlurUpsample(128, 64)
        self.up4_conv = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)
    
    def forward(self, x):
        image = x[:, :3, :, :]
        target_age = x[:, 4:, :, :]
        """ENCODER"""
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        """BOTTLE-NECT"""
        x5 = self.down4(x4)

        """DECODER"""
        x = self.up1(x5)
        x = self.up1_conv(x, x4)
        x = self.up2(x)
        x = self.up2_conv(x, x3)
        x = self.up3(x)
        x = self.up3_conv(x, x2)
        x = self.up4(x)
        x = self.up4_conv(x, x1)
        aging_delta = self.out_conv(x)
        generated_image = image + aging_delta
        return torch.cat([generated_image, target_age], dim=1)
        

# -


class Generator(nn.Module):
    def __init__(self, bisnet_config):
        super(Generator, self).__init()
        self.masknet = BiSeNet(n_classes=19)
        self.masknet.load_state_dict(torch.load(bisnet_config))
        self.masknet.eval()
        self.unet = Unet()
    
    def forward(x, input_age, target_age):
        x_mask = self.masknet(x)[0]
        #TODO
        x_mask_input_age =  mask_to_skin_age_normalize(x_mask, input_age)
        x_mask_target_age =  mask_to_skin_age_normalize(x_mask, target_age)
        x = torch.stack([x, x_mask_input_age, x_mask_target_age], axis = 1)
        return self.unet(x)


# +
# in_tensor = torch.rand(2,5, 512,512)
# conv = DoubleConv(5, 64)
# out = conv(in_tensor)
# out.shape

# +
# pool = MaxBlurPool2D(c_out=64)
# out = pool(out)
# out.shape

# +
# blurup = BlurUpsample(64, 64)
# out_2 = blurup(out)
# out_2.shape

# +
# unet = Unet()

# +
# in_tensor = torch.rand(2,5, 512,512)
# out = unet(in_tensor)

# +
# out.shape
# -
import numpy as np

masknet = BiSeNet(n_classes=19)


masknet.load_state_dict(torch.load("../weights/79999_iter.pth"))

in_tensor = torch.rand(2,3, 512,512)

output_mask = masknet(in_tensor)[0]

filter_skin = [1,2,3,7,8,9,10,11,12,13,14,15]

#filter skin
filtered_mask = output_mask[:, filter_skin, :, :]
#squeeze to 1 image
filtered_mask = filtered_mask.argmax(1)
#normalize to 1
filtered_mask = torch.where(filtered_mask > 1, 1, 0)
filtered_mask = filtered_mask * torch.Tensor([30, 20])

filtered_mask = torch.mul(torch.Tensor([[30], [20]]), filtered_mask)

filtered_mask

filtered_mask.shape

atts = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
        'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

for l, att in enumerate(atts, 1):
    print(l, att)

print(torch.unique(output_mask).shape)


