import numpy as np
import torch
from collections import OrderedDict
from torch.autograd import Variable
import random

import util.utils as util
import itertools
from util.imagePool import ImagePool
from .baseModel import BaseModel
from . import networks
from PIL import Image


class GanModel(BaseModel):
    def name(self):
        return 'GanModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.opt = opt
        nb = opt.batchsize
        size = opt.finesize

        self.opt = opt
        # 初始化输入空间
        self.inputA = self.Tensor(nb, opt.input_nc, size, size)
        self.inputB = self.Tensor(nb, opt.input_nc, size, size)
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)

        # 设置vgg loss
        self.vgg_loss = networks.PerceptualLoss(opt)
        if self.opt.IN_vgg:
            self.vgg_patch_loss = networks.PerceptualLoss(opt)
            self.vgg_patch_loss.cuda()
        self.vgg_loss.cuda()
        self.vgg = networks.load_vgg16("./models/", self.gpu_id)
        self.vgg.eval()
        for i in self.vgg.parameters():
            i.requires_grad = False

        skip = True if opt.skip > 0 else False
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_id,
                                        skip=skip, opt=opt)
        # 鉴别器
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_id, False)
            if self.opt.patchD:
                self.netD_P = networks.define_D(opt.input_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_patchD, opt.norm, use_sigmoid, self.gpu_id, True)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_net(self.netG_A, 'G_A', which_epoch)
            if self.isTrain:
                self.load_net(self.netD_A, 'D_A', which_epoch)
                if self.opt.patchD:
                    self.load_net(self.netD_P, 'D_P', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fakeB_pool = ImagePool(opt.pool_size)

            # loss
            self.criterionGAN = networks.GANLoss()
            self.criterionCycle = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # optimizer
            self.optimizer_G = torch.optim.Adam(
                self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D_A = torch.optim.Adam(
                self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_P = torch.optim.Adam(
                self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG_A)
            if self.isTrain:
                networks.print_network(self.netD_A)
                if self.opt.patchD:
                    networks.print_network(self.netD_P)
            if opt.isTrain:
                self.netG_A.train()
            else:
                self.netG_A.eval()
            print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        inputA = input['A' if AtoB else 'B']
        inputB = input['B' if AtoB else 'A']
        input_img = input['input_img']
        self.inputA.resize_(inputA.size()).copy_(
            inputA)  # inputA数据赋值给self.inputA
        self.inputB.resize_(inputB.size()).copy_(inputB)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def test(self):
        self.realA = Variable(self.inputA, volatitle=True)
        if self.opt.noise > 0:
            self.noise = Variable(
                torch.cuda.FloatTensor(self.realA.size()).normal_(mean=0, std=self.opt.noise / 255.))

        if self.opt.input_linear:
            self.realA = (self.realA - torch.min(self.realA)) / \
                (torch.max(self.realA) - torch.min(self.realA))
        if self.opt.skip == 1:
            self.fakeB, self.latentRealA = self.netG_A.forward(self.realA)

        else:
            self.fakeB = self.netG_A.forward(self.realA)

        self.realB = Variable(self.inputB, volatile=True)

    def predict(self):
        with torch.no_grad():
            self.realA = Variable(self.inputA)
        if self.opt.noise > 0:
            self.noise = Variable(
                torch.cuda.FloatTensor(self.realA.size()).normal_(mean=0, std=self.opt.noise / 255.))
            self.realA = self.realA + self.noise
        if self.opt.input_linear:
            self.realA = (self.realA - torch.min(self.realA)) / (
                torch.max(self.realA) - torch.min(self.realA))
        if self.opt.skip == 1:
            self.fakeB, self.latentRealA = self.netG_A.forward(self.realA)
        else:
            self.fakeB, self.latent_real_A = self.netG_A.forward(self.realA)
        realA = util.tensor2im(self.realA.data)
        fakeB = util.tensor2im(self.fakeB.data)
        # 保存图片
        # Image.SAVE
        return OrderedDict([('realA', realA), ('fakeB', fakeB)])

    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD.forward(real)
        pred_fake = netD.forward(fake.detach())  # ok的

        lossD_real = self.criterionGAN(pred_real, True)  # .
        lossD_fake = self.criterionGAN(pred_fake, False)
        lossD = (lossD_fake + lossD_real) / 2

        return lossD

    def backward_D_A(self):
        fakeB = self.fakeB_pool.query(self.fakeB)
        fakeB = self.fakeB
        self.lossD_A = self.backward_D_basic(self.netD_A, self.realB, fakeB)
        self.lossD_A.backward()

    def backward_D_P(self):
        lossD_P = self.backward_D_basic(
            self.netD_P, self.real_patch, self.fake_patch)
        for i in range(self.opt.patchD_3):   #
            lossD_P += self.backward_D_basic(self.netD_P,
                                             self.real_patch1[i], self.fake_patch1[i])
        self.lossD_P = lossD_P / float(self.opt.patchD_3 + 1)
        self.lossD_P.backward()

    def backward_G(self, epoch):
        pred_fake = self.netD_A.forward(self.fakeB)
        pred_real = self.netD_A.forward(self.realB)
        self.lossG_A = (self.criterionGAN(pred_real - torch.mean(pred_fake), False) +
                        self.criterionGAN(pred_fake - torch.mean(pred_real), True)) / 2
        loss = 0
        if self.opt.patchD:
            pred_fake_patch = self.netD_P.forward(self.fake_patch)
            pred_real_patch = self.netD_P.forward(self.real_patch)
            loss += (self.criterionGAN(pred_fake_patch - torch.mean(pred_fake_patch), False) +
                     self.criterionGAN(pred_real_patch - torch.mean(pred_real_patch), True)) / 2

        if self.opt.patchD_3 > 0:
            for i, data in enumerate(self.fake_patch1):
                pred_fake_patch1 = self.netD_P.forward(data)
                pred_real_patch1 = self.netD_P.forward(data)
                loss += (self.criterionGAN(pred_fake_patch1 - torch.mean(pred_fake_patch1), True) +
                         self.criterionGAN(pred_real_patch1 - torch.mean(pred_real_patch1), False)) / 2

            self.lossG_A += loss/float(self.opt.patchD_3 + 1)
        # vgg loss
        vgg_w = 1.0
        self.loss_vgg_b = self.vgg_loss.compute(
            self.vgg, self.fakeB, self.realA) * self.opt.vgg
        vgg_loss = 0
        vgg_loss += self.vgg_loss.compute(self.vgg,
                                          self.fake_patch, self.input_patch) * self.opt.vgg

        if self.opt.patchD_3 > 0:
            for i in range(self.opt.patchD_3):
                vgg_loss += self.vgg_loss.compute(
                    self.vgg, self.fake_patch1[i], self.input_patch1[i]) * self.opt.vgg

        self.loss_vgg_b += loss / float(self.opt.patchsize+1)

        self.lossG = self.lossG_A + self.loss_vgg_b*vgg_w
        self.lossG.backward()

    def forward(self):
        self.realA = Variable(self.inputA)
        self.realB = Variable(self.inputB)
        self.realImg = Variable(self.input_img)

        self.fakeB, self.latentRealA = self.netG_A.forward(self.realImg)
        #
        w = self.realA.size(3)
        h = self.realA.size(2)
        w_offset = random.randint(0, max(0, w-self.opt.patchsize - 1))
        h_offset = random.randint(0, max(0, h-self.opt.patchsize - 1))
        self.fake_patch = self.fakeB[:, :, h_offset:h_offset +
                                     self.opt.patchsize, w_offset:w_offset+self.opt.patchsize]
        self.real_patch = self.realB[:, :, h_offset:h_offset +
                                     self.opt.patchsize, w_offset:w_offset+self.opt.patchsize]
        self.input_patch = self.realA[:, :, h_offset:h_offset +
                                      self.opt.patchsize, w_offset:w_offset+self.opt.patchsize]

        if self.opt.patchD_3 > 0:
            self.fake_patch1 = []
            self.real_patch1 = []
            self.input_patch1 = []

            w = self.realA.size(3)
            h = self.realA.size(2)

            for i in range(self.opt.patchD_3):
                w_offset1 = random.randint(
                    0, max(0, w - self.opt.patchsize - 1))
                h_offset1 = random.randint(
                    0, max(0, h - self.opt.patchsize - 1))
# is_cuda??
                self.fake_patch1.append(self.fakeB[:, :, h_offset1:h_offset1+self.opt.patchsize,
                                        w_offset1:w_offset1+self.opt.patchsize])
                self.real_patch1.append(self.realB[:, :, h_offset1:h_offset1+self.opt.patchsize,
                                        w_offset1:w_offset1+self.opt.patchsize])
                self.input_patch1.append(self.realA[:, :, h_offset1:h_offset1+self.opt.patchsize,
                                         w_offset1:w_offset1+self.opt.patchsize])

    def optimize_parameter(self, epoch):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G(epoch)
        self.optimizer_G.step()

        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_P.zero_grad()
        self.backward_D_P()
        self.optimizer_D_A.step()
        self.optimizer_D_P.step()

    def get_current_error(self, epoch):
        D_A = self.lossD_A.item()
        D_P = self.lossD_P.item()
        G_A = self.lossG_A.item()

        vgg = self.loss_vgg_b.item()/self.opt.vgg
        return OrderedDict([('D_A', D_A), ('D_P', D_P), ('G_A', G_A), ('vgg', vgg)])

    def save(self, label):
        self.save_net(self.netG_A, 'G_A', label, self.gpu_id)
        self.save_net(self.netD_P, 'D_P', label, self.gpu_id)
        self.save_net(self.netD_P, 'D_P', label, self.gpu_id)

    def update_lr(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for item in self.optimizer_D_A.param_groups:
            item['lr'] = lr
        for item in self.optimizer_D_P.param_groups:
            item['lr'] = lr
        for item in self.optimizer_G.param_groups:
            item['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
