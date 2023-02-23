import os
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_id = opt.gpu_id
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_id else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimizer_param(self):
        pass

    def get_current_val(self):
        pass

    def get_current_err(self):
        pass

    def save(self, label):
        pass

    def save_net(self, net, net_label, epoch_label, gpu_id):
        save_filename = "%s_net_%s.pth" % (epoch_label, net_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(net.cpu().state_dict(), save_path)
        if len(gpu_id) and torch.cuda.is_available():
            net.cuda(device=gpu_id[0])

    def load_net(self, net, net_label, epoch_label):
        save_filename = "%s_net_%s.pth" % (epoch_label, net_label)
        save_path = os.path.join(self.save_dir, save_filename)
        net.load_state_dict(torch.load(save_path), strict=False)

    def update_learning_rate():
        pass
