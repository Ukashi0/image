import os
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import random
import util.imageFolder as imageFolder


class BaseDataLoader():
    def __init__(self):
        pass

    def initialize(self, opt):
        self.opt = opt
        pass
    # def load_data():
    #     return None


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


class dataLoader(BaseDataLoader):
    def name(self):
        return 'dataLoder'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,   # torch TensorDataset format
            batch_size=opt.batchsize,   # mini batch size
            # batch_size=4,
            shuffle=not opt.serial_batches,   # 是否打乱数据
            num_workers=int(opt.nThreads))  # # 多线程读数据

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


def CreateDataLoader(opt):
    data_loader = dataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


def CreateDataset(opt):
    dataset = UnalignedDataset()
    dataset.initialize(opt)  # 进入查看
    return dataset


def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        zoom = 1 + 0.1*random.randint(0, 4)
        osize = [int(400*zoom), int(600*zoom)]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.finesize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.finesize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.finesize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.finesize))
    # elif opt.resize_or_crop == 'no':
    #     osize = [384, 512]
    #     transform_list.append(transforms.Scale(osize, Image.BICUBIC))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dirA = os.path.join(opt.dataroot, opt.phase+'A')
        self.dirB = os.path.join(opt.dataroot, opt.phase+'B')
        self.imgA, self.pathA = imageFolder.store_dataset(self.dirA)
        self.imgB, self.pathB = imageFolder.store_dataset(self.dirB)
        self.sizeA = len(self.pathA)
        self.sizeB = len(self.pathB)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        imgA = self.imgA[index % self.sizeA]
        imgB = self.imgB[index % self.sizeB]
        pathA = self.pathA[index % self.sizeA]
        pathB = self.pathB[index % self.sizeB]
        imgA = self.transform(imgA)
        imgB = self.transform(imgB)
        if self.opt.resize_or_crop == 'no':
            input_img = imgA
        else:
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(imgA.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                imgA = imgA.index_select(2, idx)
                imgB = imgB.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(imgA.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                imgA = imgA.index_select(1, idx)
                imgB = imgB.index_select(1, idx)
            if self.opt.vary == 1 and (not self.opt.no_flip) and random.random() < 0.5:
                times = random.randint(
                    self.opt.low_times, self.opt.high_times) / 100.
                input_img = (imgA + 1) / 2. / times
                input_img = input_img * 2 - 1
            else:
                input_img = imgA
            if self.opt.lighten:
                imgB = (imgB + 1) / 2.
                imgB = (imgB - torch.min(imgB)) / \
                    (torch.max(imgB) - torch.min(imgB))
                imgB = imgB * 2. - 1

        return {'A': imgA, 'B': imgB, 'input_img': input_img,
                'A_paths': pathA, 'B_paths': pathB}

    def __len__(self):
        return max(self.sizeA, self.sizeB)

    def name(self):
        return 'UnalignedDataset'
