import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia


class ResidualRefineBlock(nn.Module):
    """Residual refinement block to restore detail lost after denoising."""
    def __init__(self, ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(ch, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        r = self.act(self.conv1(x))
        r = self.act(self.conv2(r))
        r = self.conv3(r)
        return x + r  # residual addition


class enhance_net_refine(nn.Module):
    """
    Zero-DCE + Adaptive Denoise + Residual Refinement
    Designed to outperform vanilla Zero-DCE on PSNR & SSIM.
    """

    def __init__(self, number_f=32):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        nf = number_f

        # --- Enhancement branch ---
        self.e_conv1 = nn.Conv2d(3, nf, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv5 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(nf * 2, 24, 3, 1, 1)

        # --- Learnable denoise layer (1x1 conv acts as adaptive blur) ---
        self.denoise_conv = nn.Conv2d(3, 3, 1, 1, 0, bias=True)

        # --- Residual refinement ---
        self.refine = ResidualRefineBlock(ch=nf)

        # --- Edge attention for fusion ---
        self.edge_attn = nn.Conv2d(1, 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        orig = x

        # ============= Enhancement core (Zero-DCE) =============
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        x_ = x
        for r in [r1, r2, r3, r4, r5, r6, r7, r8]:
            x_ = x_ + r * (torch.pow(x_, 2) - x_)
        enhance_image = x_

        # ============= Adaptive Denoising =============
        # Light learned smoothing
        den = self.denoise_conv(enhance_image)
        den = kornia.filters.gaussian_blur2d(
            den, (3, 3), (1.0, 1.0)
        )

        # ============= Edge-aware Refinement =============
        edges = kornia.filters.sobel(enhance_image).abs().mean(1, keepdim=True)
        attn = self.sigmoid(self.edge_attn(edges))  # [B,1,H,W]

        refined = self.refine(den)
        # adaptive fusion between denoised and refined using edge attention
        fused = attn * refined + (1 - attn) * den

        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], dim=1)
        return enhance_image, fused, r


if __name__ == '__main__':
    net = enhance_net_refine()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        e1, fused, r = net(x)
    print('e1', e1.shape, 'fused', fused.shape, 'r', r.shape)