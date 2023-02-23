import torch
import os
import numpy as np
import torch.nn as nn
import functools
import torch.nn.functional as F
from torch.autograd import Variable
# from lib.nn import SynchronizedBatchNorm2d as SynBN2d


def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % norm_type)
    return norm_layer


def print_network(net):
    num = 0
    for param in net.parameters():
        num += param.numel()

    print(net)
    print('Total number of parameters: %d' % num)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[], patch=False):
    # netD = None
    use_gpu = 1
    norm_layer = get_norm_layer(norm_type=norm)
    if use_gpu:
        assert (torch.cuda.is_available())
    netD = NoNormDiscriminator(
        input_nc, ndf, n_layers_D, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    if use_gpu:
        netD.cuda(device=gpu_ids[0])
        netD = torch.nn.DataParallel(netD, gpu_ids)
    netD.apply(weights_init)
    return netD


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], skip=False,
             opt=None):
    netG = None
    use_gpu = 1
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    # 模型搭建
    netG = Unet_resize_conv(opt, skip)
    if len(gpu_ids) >= 0:
        netG.cuda(device=gpu_ids[0])
        netG = torch.nn.DataParallel(netG, gpu_ids)
    netG.apply(weights_init)
    return netG


def vgg_preprocess(img, opt):
    tensortype = type(img.data)
    (r, g, b) = torch.chunk(img, 3, dim=1)
    img = torch.cat((b, g, r), dim=1)  # 转换格式,rgb-bgr
    img = (img + 1) * 255 * 0.5  # [-1,1] => [0,255]
    if opt.vgg_mean:
        mean = tensortype(img.data.size())
        mean[:, 0, :, :] = 103.939
        mean[:, 1, :, :] = 116.779
        mean[:, 2, :, :] = 123.680
        img = img.sub(Variable(mean))  # subtract mean
    return img


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]


def pad_tensor(input):
    height, width = input.shape[2], input.shape[3]
    divide = 16
    if width % divide != 0 or height % divide != 0:
        width_res = width % divide
        height_res = height % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0
        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d(
            (pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    return input, pad_left, pad_right, pad_top, pad_bottom


def load_vgg16(model_dir, gpu_ids):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    vgg = Vgg16()
    vgg.cuda()
    vgg.cuda(device=gpu_ids[0])
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    vgg = torch.nn.DataParallel(vgg, gpu_ids)
    return vgg


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X, opt):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        if opt.vgg_choose != "no_maxpool":
            h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        relu4_1 = h
        h = F.relu(self.conv4_2(h), inplace=True)
        relu4_2 = h
        conv4_3 = self.conv4_3(h)
        h = F.relu(conv4_3, inplace=True)
        relu4_3 = h

        if opt.vgg_choose != "no_maxpool":
            if opt.vgg_maxpooling:
                h = F.max_pool2d(h, kernel_size=2, stride=2)

        relu5_1 = F.relu(self.conv5_1(h), inplace=True)
        relu5_2 = F.relu(self.conv5_2(relu5_1), inplace=True)
        conv5_3 = self.conv5_3(relu5_2)
        h = F.relu(conv5_3, inplace=True)
        relu5_3 = h
        if opt.vgg_choose == "conv4_3":
            return conv4_3
        elif opt.vgg_choose == "relu4_2":
            return relu4_2
        elif opt.vgg_choose == "relu4_1":
            return relu4_1
        elif opt.vgg_choose == "relu4_3":
            return relu4_3
        elif opt.vgg_choose == "conv5_3":
            return conv5_3
        elif opt.vgg_choose == "relu5_1":
            return relu5_1
        elif opt.vgg_choose == "relu5_2":
            return relu5_2
        elif opt.vgg_choose == "relu5_3" or "maxpool":
            return relu5_3


class PerceptualLoss(nn.Module):
    def __init__(self, opt):
        super(PerceptualLoss, self).__init__()
        self.opt = opt
        # 归一化
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def compute(self, vgg, img, target):
        img_vgg = vgg_preprocess(img, self.opt)
        target_vgg = vgg_preprocess(target, self.opt)
        img_fea = vgg(img_vgg, self.opt)

        target_fea = vgg(target_vgg, self.opt)

        # normalize
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()  # 当前输入的结果与还原的结果，的相似度
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(
                    real_tensor, requires_grad=False)
            target_tensor = self.real_label_var.cuda()
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:  # self.Tensor(input.size()).fill_(self.fake_label)
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(
                    fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var.cuda()   # ?!!
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Discriminator
class NoNormDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, gpu_ids=[]):
        super(NoNormDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)


# CBAM
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


# G
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ch_out),
            nn.Conv2d(ch_out, ch_out, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ch_out),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ch_out),
            nn.Conv2d(ch_out, ch_out, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Unet_resize_conv(nn.Module):
    def __init__(self, opt, skip):
        super(Unet_resize_conv, self).__init__()

        self.opt = opt
        self.skip = skip
        p = 1
        self.Maxpool = nn.MaxPool2d(2)
        # self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p)
        if opt.self_attention:
            self.cbam1 = CBAM(channel=32)
            self.cbam2 = CBAM(channel=64)
            self.cbam3 = CBAM(channel=128)
            self.cbam4 = CBAM(channel=256)
            self.cbam5 = CBAM(channel=512)

        self.conv1 = conv_block(ch_in=3, ch_out=32)
        self.conv2 = conv_block(ch_in=32, ch_out=64)
        self.conv3 = conv_block(ch_in=64, ch_out=128)
        self.conv4 = conv_block(ch_in=128, ch_out=256)
        self.conv5 = up_conv(ch_in=256, ch_out=512)
        self.deconv5 = nn.Conv2d(512, 256, 3, padding=1)  #
        self.conv6 = up_conv(ch_in=512, ch_out=256)
        self.deconv6 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv7 = up_conv(ch_in=256, ch_out=128)
        self.deconv7 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv8 = up_conv(ch_in=128, ch_out=64)
        self.deconv8 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv9 = up_conv(ch_in=64, ch_out=32)

        self.conv10 = nn.Conv2d(32, 3, 1)
        if self.opt.tanh:
            self.tanh = nn.Tanh()

    def depth_to_space(self, input, block_size):
        block_size_sq = block_size * block_size
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / block_size_sq)
        s_width = int(d_width * block_size)
        s_height = int(d_height * block_size)
        t_1 = output.resize(batch_size, d_height, d_width,
                            block_size_sq, s_depth)
        spl = t_1.split(block_size, 3)
        stack = [t_t.resize(batch_size, d_height, s_width, s_depth)
                 for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).resize(batch_size, s_height, s_width,
                                                                                     s_depth)
        output = output.permute(0, 3, 1, 2)
        return output

    def forward(self, input):
        flag = 0
        if input.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            input = avg(input)
            flag = 1
        # pass
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)

        x1 = self.conv1(input)
        conv1 = self.cbam1(x1)+x1

        x2 = self.Maxpool(conv1)
        x2 = self.conv2(x2)
        conv2 = self.cbam2(x2)+x2

        x3 = self.Maxpool(conv2)
        x3 = self.conv3(x3)
        conv3 = self.cbam3(x3)+x3

        x4 = self.Maxpool(conv3)
        x4 = self.conv4(x4)
        conv4 = self.cbam4(x4)+x4

        x5 = self.Maxpool(conv4)
        x5 = self.conv5(x5)

        d5 = F.upsample(x5, scale_factor=2, mode='bilinear')
        d6 = torch.cat([self.deconv5(d5), conv4], 1)
        d6 = self.conv6(d6)

        d6 = F.upsample(d6, scale_factor=2, mode='bilinear')
        d7 = torch.cat([self.deconv6(d6), conv3], 1)
        d7 = self.conv7(d7)

        d7 = F.upsample(d7, scale_factor=2, mode='bilinear')
        d8 = torch.cat([self.deconv7(d7), conv2], 1)
        d8 = self.conv8(d8)

        d8 = F.upsample(d8, scale_factor=2, mode='bilinear')
        d9 = torch.cat([self.deconv8(d8), conv1], 1)
        d9 = self.conv9(d9)

        latent = self.conv10(d9)

        if self.opt.tanh:
            latent = self.tanh(latent)
        if self.skip:
            if self.opt.linear_add:
                if self.opt.latent_threshold:
                    latent = F.relu(latent)
                elif self.opt.latent_norm:
                    latent = (latent - torch.min(latent)) / \
                        (torch.max(latent) - torch.min(latent))
                input = (input - torch.min(input)) / \
                    (torch.max(input) - torch.min(input))
                output = latent + input * self.opt.skip
                output = output * 2 - 1
            else:
                if self.opt.latent_threshold:
                    latent = F.relu(latent)
                elif self.opt.latent_norm:
                    latent = (latent - torch.min(latent)) / \
                        (torch.max(latent) - torch.min(latent))
                output = latent + input * self.opt.skip  # resnet
        else:
            output = latent

        if self.opt.linear:
            output = output / torch.max(torch.abs(output))

        # 做切割
        output = pad_tensor_back(
            output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(
            latent, pad_left, pad_right, pad_top, pad_bottom)

        if flag == 1:
            output = F.upsample(output, scale_factor=2, mode='bilinear')
        if self.skip:
            return output, latent
        else:
            return output
