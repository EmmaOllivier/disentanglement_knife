import torch
from torch import nn
import torch.nn.parallel
import torch.utils.data



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class BasicConv2d_Ins(nn.Module):
    '''
    BasicConv2d module with InstanceNorm
    '''
    def __init__(self, in_planes, out_planes, kernal_size, stride, padding):
        super(BasicConv2d_Ins, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernal_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.InstanceNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x= self.relu(x)
        return x


class block32_Ins(nn.Module):
    def __init__(self, scale=1.0):
        super(block32_Ins, self).__init__()

        self.scale = scale

        self.branch0 = nn.Sequential(BasicConv2d_Ins(64, 16, kernal_size=1, stride=1, padding=0))

        self.branch1 = nn.Sequential(
        BasicConv2d_Ins(64, 16, kernal_size=1, stride=1, padding=0),
        BasicConv2d_Ins(16, 16, kernal_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
        BasicConv2d_Ins(64, 16, kernal_size=1, stride=1, padding=0),
        BasicConv2d_Ins(16, 16, kernal_size=3, stride=1, padding=1),
        BasicConv2d_Ins(16, 16, kernal_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(48, 64, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    '''
    encoder structure: Inception + Instance Normalization
    '''
    def __init__(self, GRAY=False):
        super(Encoder, self).__init__()

        self.dropout = nn.Dropout(p=0.5)
        self.act = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        if GRAY:
            self.conv1 = nn.Sequential(BasicConv2d_Ins(1, 32, kernal_size=5, stride=1, padding=2))
        else:
            self.conv1 = nn.Sequential(BasicConv2d_Ins(3, 32, kernal_size=5, stride=1, padding=2))

        self.conv2 = nn.Sequential(BasicConv2d_Ins(32, 64, kernal_size=5, stride=1, padding=2))
        self.repeat = nn.Sequential(
            block32_Ins(scale=0.17),
            block32_Ins(scale=0.17),
            block32_Ins(scale=0.17),
            block32_Ins(scale=0.17)
        )
        self.conv3 = nn.Sequential(BasicConv2d_Ins(64, 128, kernal_size=5, stride=1, padding=2))
        self.conv4 = nn.Sequential(BasicConv2d_Ins(128, 128, kernal_size=5, stride=1, padding=2))
        self.fc1 = nn.Linear(8 * 8 * 128, 1000)

    def forward(self, x_in):
        # in_chanx128x128 -> 32x128x128
        self.conv1_out = self.conv1(x_in)
        # 32x128x128 -> 32x64x64
        self.ds1_out = self.maxpool(self.conv1_out)
        # 32x64x64 -> 64x64x64
        self.conv2_out = self.conv2(self.ds1_out)
        # 64x64x64 -> 64x32x32
        self.ds2_out = self.maxpool(self.conv2_out)
        # 64x32x32 -> 64x32x32
        self.incep_out = self.repeat(self.ds2_out)
        # 64x32x32 -> 128x32x32
        self.conv3_out = self.conv3(self.incep_out)
        # 128x32x32 -> 128x16x16
        self.ds3_out = self.maxpool(self.conv3_out)
        # 128x16x16 -> 128x16x16
        self.conv4_out = self.conv4(self.ds3_out)
        # 128x16x16 -> 128x8x8
        self.ds4_out = self.maxpool(self.conv4_out)
        self.fc1_out = self.act(self.fc1(self.dropout(self.ds4_out.view(self.ds4_out.size(0), -1))))
        return self.fc1_out


class fc_layer(nn.Module):
    def __init__(self, par=None, p=0.5, cls_num=10575):
        super(fc_layer, self).__init__()
        # activation function
        self.act = nn.ReLU()

        # network structure
        self.fc1 = nn.Linear(8 * 8 * 128, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, cls_num)

        self.dropout = nn.Dropout(p=p)

        # parameters initiation
        if par:
            # to load pre-trained model
            fc_dict = self.state_dict().copy()
            fc_list = list(self.state_dict().keys())

            fc_dict[fc_list[0]] = par['module.fc.weight']
            fc_dict[fc_list[1]] = par['module.fc.bias']

            # load pre-trained parameters into Encoder
            self.load_state_dict(fc_dict)
        else:
            # initiate parameters
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.02)
                elif isinstance(m, nn.ConvTranspose2d):
                    m.weight.data.normal_(0, 0.02)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.02)
                    m.bias.data.fill_(0)

    def forward(self, fea):

        self.fc2_out = self.act(self.fc2(self.dropout(fea)))
        self.fc3_out = self.fc3(self.fc2_out)
        return self.fc3_out


class resblock(nn.Module):
    '''
    residual block
    '''
    def __init__(self, n_chan):
        super(resblock, self).__init__()
        self.infer = nn.Sequential(*[
            nn.Conv2d(n_chan, n_chan, 3, 1, 1),
            nn.ReLU()
        ])

    def forward(self, x_in):
        self.res_out = x_in + self.infer(x_in)
        return self.res_out




class Dis(nn.Module):
    '''
    the class of discriminator to handle classification
    '''
    def __init__(self, fc=None, GRAY=False, cls_num=2):
        super(Dis, self).__init__()

        # initiate encoder
        self.enc = Encoder(GRAY=GRAY)

        # initiate fc layer
        self.fc = Classif(1, False)

    def forward(self, x_in):
        self.fea = self.enc(x_in)
        self.result = self.fc(self.fea)
        return self.fea, self.result



class Classif(torch.nn.Module):
    def __init__(self,nb_class,softmax=True):
        super(Classif,self).__init__()

        self.fc1 = torch.nn.Linear(1000,100)
        self.fc2 = torch.nn.Linear(100,nb_class)
        self.softmax = softmax


    def forward(self,x):
        x = self.fc1(x).relu()
        if self.softmax:
            x = self.fc2(x).softmax(dim=-1)
        else:
            x = self.fc2(x).sigmoid().reshape(-1)
        return x
