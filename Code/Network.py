import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        l = [32, 64, 128]
        self.l = l
        self.inplanes = l[0]  # 64
        self.dilation = 1

        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=5, stride=2, padding=2,
                               bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.fconv1 = nn.Conv2d(l[0], l[1], kernel_size=5, stride=2, padding=2)

        self.fconv2 = nn.Conv2d(l[1], l[2], kernel_size=3, stride=2)

        self.fc1 = nn.Linear(l[2] * 3 * 3, 10)

        self.fc2 = nn.Linear(in_features=10,
                             out_features=1152)

        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 5, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=2)

    def _make_layer(self, block, planes, blocks, n_inplane, stride=1, dilate=False):
        norm_layer = self.norm_layer
        layers = []
        for _ in range(0, blocks):
            layers.append(block(n_inplane, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)

        x = self.fconv1(x)
        x = torch.relu(x)

        x = self.fconv2(x)
        x = torch.relu(x)

        x = x.view(-1, self.l[2] * 3 * 3)

        u = self.fc1(x)

        x = self.fc2(u)
        x = torch.relu(x)
        x = x.view(-1, self.l[2], 3, 3)
        x = self.deconv1(x)
        x = torch.relu(x)

        x = self.deconv2(x)
        x = torch.relu(x)
        x = self.deconv3(x)

        return x, u
