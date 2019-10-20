#**coding=utf-8**
import torch.nn as nn
import torch.functional
import math

def conv_bn(inp, oup, stride ):
    return nn.Sequential(
        nn.Conv2d(inp, oup,kernel_size= 3, stride= stride, padding= 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
        #ncrelu()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size = 1, stride= 1, padding= 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
        #ncrelu()
		)


class PermutationBlock(nn.Module):
    def __init__(self, groups):
        super(PermutationBlock, self).__init__()
        self.groups = groups

    def forward(self, input):
        n, c, h, w = input.size()
        G = self.groups
        output = input.view(n, G, c // G, h, w).permute(0, 2, 1, 3, 4).contiguous().view(n, c, h, w)
        return output

class InvertedResidual_bn(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_bn, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup
        cur_relu = nn.ReLU6(inplace=True)

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio,kernel_size = 1, stride= 1, padding=0,groups = 2, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            #nn.ReLU6(inplace=True),#remove according to paper
            #permutation
            PermutationBlock(groups=2),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel_size =3, stride= stride, padding=1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, kernel_size =1, stride= 1, padding=0,groups = 2, bias=False),
            nn.BatchNorm2d(oup),
            # permutation
            #PermutationBlock(groups= int(round((oup/2)))),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidual_unb(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_unb, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup
        self.pwconv10 = nn.Sequential(
            nn.Conv2d(inp//4, inp * expand_ratio//4,kernel_size = 1, stride= 1, padding=0, bias=False),
        )
        self.pwconv20 = nn.Sequential(
            nn.Conv2d(inp*3//4, inp * expand_ratio*3//4,kernel_size = 1, stride= 1, padding=0, bias=False),
        )
        self.pwconv11 = nn.Sequential(
            nn.Conv2d(inp * expand_ratio//4, oup//4,kernel_size = 1, stride= 1, padding=0, bias=False),
        )
        self.pwconv21 = nn.Sequential(
            nn.Conv2d(inp * expand_ratio*3//4, oup*3//4,kernel_size = 1, stride= 1, padding=0, bias=False),
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel_size =3, stride= stride, padding=1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
        )
        self.Permu2 = nn.Sequential(
            nn.BatchNorm2d(inp * expand_ratio),
            PermutationBlock(groups=2),
        )
        self.Permu4 = nn.Sequential(
            nn.BatchNorm2d(oup),
            #PermutationBlock(groups= int(round((oup/2)))),
        )


    def forward(self, x):
        x_res = x
        x1 = x[:, :(x.shape[1]//4), :, :]
        x2 = x[:, (x.shape[1]//4):, :, :]
        x1 = self.pwconv10(x1)
        x2 = self.pwconv20(x2)
        x = torch.cat((x1, x2), 1)
        x = self.Permu2(x)
        x = self.dwconv(x)
        x1 = x[:, :(x.shape[1]//4), :, :]
        x2 = x[:, (x.shape[1]//4):, :, :]
        x1 = self.pwconv11(x1)
        x2 = self.pwconv21(x2)
        x = torch.cat((x1, x2), 1)
        x = self.Permu4(x)
        if self.use_res_connect:
            return x_res + x
        else:
            return x


class seesawnet_half(nn.Module):
    def __init__(self):
        super(seesawnet_half, self).__init__()

        input_size = 224
        num_classes = 1000
        s1, s2 = 2, 2
        width_multiplier = 1.

        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s, nc
            [1, 16, 1, 1 ],
            [6, 24, 2, s2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_multiplier)
        self.last_channel = int(1280 * width_multiplier) if width_multiplier > 1.0 else 1280
        #第一层，
        self.features = [conv_bn(inp =3, oup =input_channel, stride = s1)]
        #中间block，一共7个,
        #  Layers from 1 to 7
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_multiplier)
            for i in range(n):
                if i == 0:
                    if input_channel>=160:
                        self.features.append(InvertedResidual_bn(input_channel, output_channel, s, t))
                    else:
                        self.features.append(InvertedResidual_unb(input_channel, output_channel, s, t))
                else:
                    if input_channel>=160:
                        self.features.append(InvertedResidual_bn(input_channel, output_channel, 1, t))
                    else:
                        self.features.append(InvertedResidual_unb(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(input_size // 32, stride=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            #nn.Dropout(p=0.2, inplace=True),
            nn.Linear(self.last_channel,num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()