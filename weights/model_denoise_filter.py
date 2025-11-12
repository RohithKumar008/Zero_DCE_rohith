import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia


class enhance_net_denoise(nn.Module):
    """
    Simplified Zero-DCE variant — only includes denoising (median + Gaussian),
    with DenseFusion temporarily disabled.
    """

    def __init__(self, number_f=32, median_kernel=3, gauss_kernel=3, gauss_sigma=1.0):
        super(enhance_net_denoise, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.number_f = number_f

        # --- Enhancement branch (same as original Zero-DCE) ---
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

        # --- Only filtering (no DenseFusion) ---
        self.median_kernel = median_kernel if median_kernel % 2 == 1 else median_kernel + 1
        self.gauss_kernel = gauss_kernel if gauss_kernel % 2 == 1 else gauss_kernel + 1
        self.gauss_sigma = gauss_sigma

    def forward(self, x):
        """
        x: input image tensor in range [0,1] (batch,3,H,W)
        returns (enhance_image_1, filtered_enhanced, r)
        """
        orig = x

        # -----------------------
        # 1️⃣ Enhancement branch
        # -----------------------
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        # Iterative enhancement
        x_ = x
        x_ = x_ + r1 * (torch.pow(x_, 2) - x_)
        x_ = x_ + r2 * (torch.pow(x_, 2) - x_)
        x_ = x_ + r3 * (torch.pow(x_, 2) - x_)
        x_ = x_ + r4 * (torch.pow(x_, 2) - x_)
        enhance_image_1 = x_ + r5 * (torch.pow(x_, 2) - x_)
        x_ = enhance_image_1 + r6 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x_ = x_ + r7 * (torch.pow(x_, 2) - x_)
        enhance_image = x_ + r8 * (torch.pow(x_, 2) - x_)

        # -----------------------
        # 2️⃣ Filtering stage only
        # -----------------------
        try:
            den = kornia.filters.median_blur(enhance_image, (self.median_kernel, self.median_kernel))
        except Exception:
            den = enhance_image

        try:
            den = kornia.filters.gaussian_blur2d(
                den,
                (self.gauss_kernel, self.gauss_kernel),
                (self.gauss_sigma, self.gauss_sigma)
            )
        except Exception:
            den = F.avg_pool2d(den, kernel_size=3, stride=1, padding=1)

        # Output only the filtered result (no fusion)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], dim=1)
        return enhance_image_1, den, r


if __name__ == '__main__':
    net = enhance_net_denoise()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        e1, filtered, r = net(x)
    print('e1', e1.shape, 'filtered', filtered.shape, 'r', r.shape)