import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

class enhance_net_denoise(nn.Module):
    """
    Modified Zero-DCE style enhancer with lightweight denoising and fusion.

    - Keeps the original iterative enhancement core (r1..r8) from the original
      `enhance_net_nopool` architecture.
    - Applies a median filter followed by a Gaussian blur to the enhanced image to
      reduce salt-and-pepper and high-frequency amplification noise.
    - Learns a per-pixel fusion mask from the original and denoised-enhanced
      image so that spatial details from the original image can be preserved.

    Return signature mirrors the original: (enhance_image_1, enhance_image, r)
    where `enhance_image` is the final fused & denoised enhanced image.
    """

    def __init__(self, number_f=32, median_kernel=3, gauss_kernel=3, gauss_sigma=1.0):
        super(enhance_net_denoise, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.number_f = number_f
        # same conv stack as original Zero-DCE (kept for compatibility)
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        # produce 8 maps * 3 channels = 24 channels as in original
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

        # fusion conv: takes concatenation of original and denoised-enhanced (6 channels)
        # and predicts a single-channel mask in [0,1]
        self.fusion_net = nn.Sequential(
            nn.Conv2d(6, number_f, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(number_f, number_f // 2, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(number_f // 2, 1, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # denoising params
        self.median_kernel = median_kernel if median_kernel % 2 == 1 else median_kernel + 1
        self.gauss_kernel = gauss_kernel if gauss_kernel % 2 == 1 else gauss_kernel + 1
        self.gauss_sigma = gauss_sigma

    def forward(self, x):
        """
        x: input image tensor in range [0,1] (batch,3,H,W)
        returns (enhance_image_1, enhance_image_final, r)
        """
        device = x.device
        # keep a copy of the original for fusion
        orig = x

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        # split into eight residual maps (each 3 channels)
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        # iterative enhancement as in original Zero-DCE
        x_ = x
        x_ = x_ + r1 * (torch.pow(x_, 2) - x_)
        x_ = x_ + r2 * (torch.pow(x_, 2) - x_)
        x_ = x_ + r3 * (torch.pow(x_, 2) - x_)
        x_ = x_ + r4 * (torch.pow(x_, 2) - x_)
        enhance_image_1 = x_ + r5 * (torch.pow(x_, 2) - x_)
        x_ = enhance_image_1 + r6 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x_ = x_ + r7 * (torch.pow(x_, 2) - x_)
        enhance_image = x_ + r8 * (torch.pow(x_, 2) - x_)

        # Denoising stage: median filter then Gaussian blur (on device)
        # kornia expects input in (B,C,H,W) and operates on float tensors
        try:
            den = kornia.filters.median_blur(enhance_image, (self.median_kernel, self.median_kernel))
        except Exception:
            # if kornia median not available or fails, fallback to identity
            den = enhance_image

        # gaussian blur
        try:
            den = kornia.filters.gaussian_blur2d(den, (self.gauss_kernel, self.gauss_kernel), (self.gauss_sigma, self.gauss_sigma))
        except Exception:
            # fallback: slightly smooth via avg pool/upsample
            den = F.avg_pool2d(den, kernel_size=3, stride=1, padding=1)

        # Learned fusion mask to combine denoised enhanced image with original to preserve spatial detail
        # Concatenate along channels: [orig, den]
        fusion_input = torch.cat([orig, den], dim=1)
        mask = self.fusion_net(fusion_input)  # [B,1,H,W], values in [0,1]

        # Expand mask to 3 channels
        mask3 = mask.repeat(1, 3, 1, 1)
        fused = mask3 * den + (1.0 - mask3) * orig

        # return final fused image along with first-stage enhanced and residuals
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], dim=1)
        return enhance_image_1, fused, r


if __name__ == '__main__':
    # quick smoke test
    net = enhance_net_denoise()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        e1, fused, r = net(x)
    print('e1', e1.shape, 'fused', fused.shape, 'r', r.shape)
