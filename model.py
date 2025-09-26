import torch
import torch.nn as nn

# ------------- Model with Dropout -------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)
    

# Define the attention block
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi

# Define the Attention U-Net with 5 enocnders and 4 decoders
# Dropout was set to 0.1 but can be changed
class AttentionUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, dropout=0.1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2,2)

        # You can vary the dropout rate per depth if needed
        self.Conv1 = ConvBlock(img_ch, 64, dropout=dropout)
        self.Conv2 = ConvBlock(64, 128, dropout=dropout)
        self.Conv3 = ConvBlock(128, 256, dropout=dropout)
        self.Conv4 = ConvBlock(256, 512, dropout=dropout)
        self.Conv5 = ConvBlock(512, 1024, dropout=dropout)

        self.Up5   = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.Att5  = AttentionBlock(512, 512, 256)
        self.UpConv5 = ConvBlock(1024, 512, dropout=dropout)

        self.Up4   = nn.ConvTranspose2d(512, 256, 2, 2)
        self.Att4  = AttentionBlock(256, 256, 128)
        self.UpConv4 = ConvBlock(512, 256, dropout=dropout)

        self.Up3   = nn.ConvTranspose2d(256, 128, 2, 2)
        self.Att3  = AttentionBlock(128, 128, 64)
        self.UpConv3 = ConvBlock(256, 128, dropout=dropout)

        self.Up2   = nn.ConvTranspose2d(128, 64, 2, 2)
        self.Att2  = AttentionBlock(64, 64, 32)
        self.UpConv2 = ConvBlock(128, 64, dropout=dropout)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, 1)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.maxpool(x1))
        x3 = self.Conv3(self.maxpool(x2))
        x4 = self.Conv4(self.maxpool(x3))
        x5 = self.Conv5(self.maxpool(x4))

        d5 = self.Up5(x5)
        x4g = self.Att5(d5, x4)
        d5 = self.UpConv5(torch.cat([x4g, d5], dim=1))

        d4 = self.Up4(d5)
        x3g = self.Att4(d4, x3)
        d4 = self.UpConv4(torch.cat([x3g, d4], dim=1))

        d3 = self.Up3(d4)
        x2g = self.Att3(d3, x2)
        d3 = self.UpConv3(torch.cat([x2g, d3], dim=1))

        d2 = self.Up2(d3)
        x1g = self.Att2(d2, x1)
        d2 = self.UpConv2(torch.cat([x1g, d2], dim=1))

        return self.Conv_1x1(d2)  # logits
