import random
from torch.autograd import Variable
import torch

class ImagePool():
    def __init__(self,pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        img = []
        for item in images.data:
            item = torch.unsqueeze(item, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(item)
                img.append(item)
            else:
                p = random.uniform(0,1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1) # 随机选取一张图片
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = item  # ??
                    img.append(tmp)
                else:
                    img.append(item)
        img = Variable(torch.cat(img,0))
        return img
