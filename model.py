"""
model.py
U-Net Generator + PatchGAN Discriminator (pix2pix-style architecture)
for LISS-IV cloud removal and reconstruction.
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, kernel_size=4, stride=2, padding=1, norm=True, activation="leaky"):
    """Single downsampling conv block: Conv -> (optional) Norm -> Activation."""
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    if activation == "leaky":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    elif activation == "relu":
        layers.append(nn.ReLU(inplace=True))
    # activation=None -> no activation (used for final discriminator layer)
    return nn.Sequential(*layers)


def deconv_block(in_ch, out_ch, kernel_size=4, stride=2, padding=1, dropout=False):
    """Single upsampling block: ConvTranspose -> Norm -> ReLU -> (optional) Dropout."""
    layers = [
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    """
    U-Net generator for image-to-image translation (cloudy -> clean).
    Input/output: (B, 3, 256, 256)
    """

    def __init__(self, in_channels=3, out_channels=3, base_filters=64):
        super().__init__()
        f = base_filters

        # Encoder (downsampling path)
        self.enc1 = conv_block(in_channels, f, norm=False)        # 256 -> 128
        self.enc2 = conv_block(f, f * 2)                          # 128 -> 64
        self.enc3 = conv_block(f * 2, f * 4)                      # 64 -> 32
        self.enc4 = conv_block(f * 4, f * 8)                      # 32 -> 16
        self.enc5 = conv_block(f * 8, f * 8)                      # 16 -> 8
        self.enc6 = conv_block(f * 8, f * 8)                      # 8 -> 4
        self.enc7 = conv_block(f * 8, f * 8)                      # 4 -> 2
        self.enc8 = conv_block(f * 8, f * 8, norm=False)          # 2 -> 1 (bottleneck)

        # Decoder (upsampling path), with skip connections from encoder
        self.dec1 = deconv_block(f * 8, f * 8, dropout=True)              # 1 -> 2
        self.dec2 = deconv_block(f * 8 * 2, f * 8, dropout=True)          # 2 -> 4
        self.dec3 = deconv_block(f * 8 * 2, f * 8, dropout=True)          # 4 -> 8
        self.dec4 = deconv_block(f * 8 * 2, f * 8)                        # 8 -> 16
        self.dec5 = deconv_block(f * 8 * 2, f * 4)                        # 16 -> 32
        self.dec6 = deconv_block(f * 4 * 2, f * 2)                        # 32 -> 64
        self.dec7 = deconv_block(f * 2 * 2, f)                            # 64 -> 128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(f * 2, out_channels, 4, 2, 1),  # 128 -> 256
            nn.Tanh()  # output in [-1, 1], assumes input is normalized to [-1, 1]
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)  # bottleneck

        d1 = self.dec1(e8)
        d1 = torch.cat([d1, e7], dim=1)  # skip connection

        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e6], dim=1)

        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e5], dim=1)

        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e4], dim=1)

        d5 = self.dec5(d4)
        d5 = torch.cat([d5, e3], dim=1)

        d6 = self.dec6(d5)
        d6 = torch.cat([d6, e2], dim=1)

        d7 = self.dec7(d6)
        d7 = torch.cat([d7, e1], dim=1)

        out = self.final(d7)
        return out


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator: classifies overlapping patches as real/fake
    rather than the whole image at once. Takes concatenated (input, target_or_generated)
    as a 6-channel input (3 + 3).
    """

    def __init__(self, in_channels=6, base_filters=64):
        super().__init__()
        f = base_filters

        self.model = nn.Sequential(
            conv_block(in_channels, f, norm=False),       # 256 -> 128
            conv_block(f, f * 2),                          # 128 -> 64
            conv_block(f * 2, f * 4),                      # 64 -> 32
            conv_block(f * 4, f * 8, stride=1),            # 32 -> 31 (stride 1 keeps patch-level granularity)
            nn.Conv2d(f * 8, 1, kernel_size=4, stride=1, padding=1),  # 31 -> 30, single-channel patch scores
        )

    def forward(self, input_img, target_or_generated_img):
        x = torch.cat([input_img, target_or_generated_img], dim=1)
        return self.model(x)


if __name__ == "__main__":
    # Quick sanity check: confirm shapes work end-to-end
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gen = UNetGenerator().to(device)
    disc = PatchGANDiscriminator().to(device)

    dummy_input = torch.randn(2, 3, 256, 256).to(device)  # batch of 2
    fake_output = gen(dummy_input)
    print(f"Generator input shape:  {dummy_input.shape}")
    print(f"Generator output shape: {fake_output.shape}")

    disc_output = disc(dummy_input, fake_output)
    print(f"Discriminator output shape: {disc_output.shape}")

    # Count parameters
    gen_params = sum(p.numel() for p in gen.parameters())
    disc_params = sum(p.numel() for p in disc.parameters())
    print(f"\nGenerator parameters:     {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")